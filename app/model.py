"""Small decoder-only GPT implemented from embeddings and causal attention."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 384
    context_length: int = 128
    embedding_dim: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.embedding_dim, self.heads, self.layers) <= 0:
            raise ValueError('Model dimensions must be positive')
        if self.embedding_dim % self.heads:
            raise ValueError('Embedding dimension must be divisible by heads')
        if not 0 <= self.dropout < 1:
            raise ValueError('Dropout must be between zero and one')


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = config.dropout

    def forward(self, x):
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [item.view(batch, length, self.heads, channels // self.heads).transpose(1, 2)
                   for item in (q, k, v)]
        # is_causal enforces the lower triangular mask: future tokens are invisible.
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                    dropout_p=self.dropout if self.training else 0.0)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.projection(output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(), nn.Linear(4 * config.embedding_dim, config.embedding_dim), nn.Dropout(config.dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        length = ids.size(1)
        if not 1 <= length <= self.config.context_length:
            raise ValueError('Input length must fit the context window')
        positions = torch.arange(length, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        logits = self.head(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=120, temperature=0.8, top_k=40, seed=42,
                 top_p=1.0, repetition_penalty=1.0):
        if max_new_tokens < 1 or not math.isfinite(temperature) or temperature <= 0 or top_k < 1:
            raise ValueError('Token count, temperature, and top-k must be positive and finite')
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError('top-p must be between zero (exclusive) and one')
        if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
            raise ValueError('Repetition penalty must be finite and at least one')
        self.eval()
        generator = torch.Generator(device=ids.device).manual_seed(seed)
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.config.context_length:])
            logits = logits[:, -1, :].clone()
            if repetition_penalty != 1:
                for row in range(ids.size(0)):
                    seen = ids[row, -self.config.context_length:].unique()
                    values = logits[row, seen]
                    logits[row, seen] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
            logits = logits / temperature
            cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, float('-inf'))
            if top_p < 1:
                sorted_logits, indices = logits.sort(descending=True, dim=-1)
                cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                mask = torch.zeros_like(remove).scatter(1, indices, remove)
                logits = logits.masked_fill(mask, float('-inf'))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            ids = torch.cat((ids, next_id), dim=1)
        return ids

    def checkpoint(self):
        return {'config': asdict(self.config), 'state_dict': self.state_dict()}
