"""CPU training and fixed held-out evaluation for a small GPT."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch

from app.model import GPT, GPTConfig
from app.tokenizer import BytePairTokenizer


def split_corpus(directory):
    training, validation = [], []
    for name in ['treasure-island.txt', 'alice.txt']:
        text = (Path(directory) / name).read_text(encoding='utf-8')
        split = int(len(text) * 0.9)
        training.append(text[:split])
        validation.append(text[split:])
    return '\n\n'.join(training), '\n\n'.join(validation)


def batch(data, count, context, generator):
    if len(data) <= context:
        raise ValueError('Corpus split must be longer than context length')
    starts = torch.randint(len(data) - context, (count,), generator=generator)
    return (torch.stack([data[i:i+context] for i in starts]),
            torch.stack([data[i+1:i+context+1] for i in starts]))


@torch.no_grad()
def score(model, batches):
    model.eval()
    return sum(model(x, y)[1].item() for x, y in batches) / len(batches)


def train(corpus='corpus', output='checkpoints/demo.pt', steps=3000, seed=42,
          batch_size=16, context=128, embedding_dim=128, layers=2, vocab_size=384,
          eval_interval=250, threads=4, init_from=None, learning_rate=0.001):
    if min(steps, batch_size, context, eval_interval, threads) <= 0:
        raise ValueError('Training counts must be positive')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Learning rate must be finite and positive')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    started = time.monotonic()
    training_text, validation_text = split_corpus(corpus)
    parent_step = 0
    parent_hash = None
    if init_from:
        source = Path(init_from)
        if source.resolve() == Path(output).resolve():
            raise ValueError('Use a different output path to preserve the starting checkpoint')
        previous = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
        for name, text in [('training', training_text), ('validation', validation_text)]:
            if previous[f'{name}_sha256'] != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError('Starting checkpoint corpus does not match this corpus split')
        parent_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if previous['checkpoint_sha256'] != parent_hash:
            raise ValueError('Starting checkpoint hash does not match its training report')
        model, tokenizer = load_checkpoint(source)
        saved = torch.load(source, map_location='cpu', weights_only=True)
        parent_step = saved.get('total_steps', saved['step'])
        config = model.config
        context = config.context_length
        print(f'Continuing weights from step {parent_step}; starting a new optimizer.', flush=True)
    else:
        print('Training byte-pair tokenizer...', flush=True)
        tokenizer = BytePairTokenizer.train(training_text, vocab_size)
        config = GPTConfig(vocab_size=tokenizer.vocab_size, context_length=context,
                           embedding_dim=embedding_dim, layers=layers)
        model = GPT(config)
    train_ids = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    train_rng = torch.Generator().manual_seed(seed)
    eval_rng = torch.Generator().manual_seed(seed + 1)
    validation_batches = [batch(val_ids, batch_size, context, eval_rng) for _ in range(10)]
    training_batches = [batch(train_ids, batch_size, context, eval_rng) for _ in range(10)]
    counts = torch.bincount(train_ids, minlength=tokenizer.vocab_size).float() + 1
    probabilities = counts / counts.sum()
    baseline = sum((-probabilities[y].log()).mean().item() for _, y in validation_batches) / len(validation_batches)
    history = []
    best_loss = float('inf')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(steps + 1):
        if step % eval_interval == 0 or step == steps:
            val_loss = score(model, validation_batches)
            entry = {'step': step, 'train_loss': score(model, training_batches), 'validation_loss': val_loss}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = model.checkpoint()
                checkpoint.update({'merges': tokenizer.merges, 'step': step, 'seed': seed,
                                   'total_steps': parent_step + step})
                torch.save(checkpoint, output)
        if step == steps:
            break
        model.train()
        # Cosine decay prevents a fixed high step size from disrupting learned phrases.
        rate = learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for group in optimizer.param_groups:
            group['lr'] = rate
        x, y = batch(train_ids, batch_size, context, train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    report = {'seed': seed, 'steps': steps, 'config': asdict(config),
              'batch_size': batch_size, 'learning_rate': learning_rate, 'schedule': 'cosine to 10%',
              'parent_steps': parent_step, 'total_steps': parent_step + steps,
              'parent_checkpoint_sha256': parent_hash, 'threads': threads,
              'parameters': sum(p.numel() for p in model.parameters()),
              'torch_version': str(torch.__version__), 'train_tokens': len(train_ids),
              'validation_tokens': len(val_ids), 'validation_scored_tokens': 10 * batch_size * context,
              'unigram_validation_loss': baseline, 'uniform_validation_loss': math.log(tokenizer.vocab_size),
              'best_validation_loss': best_loss, 'best_validation_perplexity': math.exp(best_loss),
              'elapsed_seconds': round(time.monotonic() - started, 2), 'history': history,
              'training_sha256': hashlib.sha256(training_text.encode()).hexdigest(),
              'validation_sha256': hashlib.sha256(validation_text.encode()).hexdigest(),
              'checkpoint_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    tokenizer.save(output.with_suffix('.tokenizer.json'))
    return report


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = GPT(GPTConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, BytePairTokenizer(checkpoint['merges'])
