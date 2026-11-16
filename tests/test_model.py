import torch
import pytest
from app.model import GPT, GPTConfig


def tiny_model():
    torch.manual_seed(7)
    return GPT(GPTConfig(vocab_size=32, context_length=16, embedding_dim=16, heads=2, layers=1, dropout=0))


def test_causal_attention_cannot_see_future():
    model = tiny_model().eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 9, 10]])
    a, _ = model(first)
    b, _ = model(changed_future)
    torch.testing.assert_close(a[:, :3], b[:, :3])
    assert not torch.allclose(a[:, 3:], b[:, 3:])


def test_backpropagation_learns_sequence():
    torch.set_num_threads(1)
    model = tiny_model()
    x = torch.tensor([[1, 2, 3, 4]] * 4)
    y = torch.tensor([[2, 3, 4, 5]] * 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = model(x, y)[1].item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
    assert model(x, y)[1].item() < initial * 0.2
    assert model.token_embedding.weight.grad.abs().sum() > 0


def test_generation_is_seeded_and_handles_long_prompt():
    model = tiny_model()
    prompt = torch.tensor([[1, 2] * 12])
    output = model.generate(prompt, max_new_tokens=5, seed=9)
    assert output.shape == (1, 29)
    assert torch.equal(output[:, :24], prompt)
    assert torch.equal(output, model.generate(prompt, max_new_tokens=5, seed=9))


def test_model_rejects_invalid_shapes_and_architecture():
    with pytest.raises(ValueError):
        GPTConfig(embedding_dim=15, heads=4)
    with pytest.raises(ValueError):
        tiny_model()(torch.ones((1, 17), dtype=torch.long))
