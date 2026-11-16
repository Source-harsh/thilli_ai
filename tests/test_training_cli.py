import json
import pytest
import torch
from app.cli import main
from app.training import train, load_checkpoint


def test_training_checkpoint_and_generation_cli(tmp_path, capsys):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    for name in ['treasure-island.txt', 'alice.txt']:
        (corpus / name).write_text('A ship sailed toward a hidden island. ' * 50, encoding='utf-8')
    checkpoint = tmp_path / 'test.pt'
    report = train(corpus, checkpoint, steps=2, batch_size=2, context=8,
                   embedding_dim=16, layers=1, vocab_size=260, eval_interval=1, threads=1)
    model, tokenizer = load_checkpoint(checkpoint)
    assert report['history'][-1]['step'] == 2
    assert checkpoint.with_suffix('.tokenizer.json').exists()
    assert json.loads(checkpoint.with_suffix('.json').read_text())['steps'] == 2
    assert model.config.vocab_size == tokenizer.vocab_size
    saved = tmp_path / 'story.txt'
    capsys.readouterr()
    main(['generate', 'A ship', '--checkpoint', str(checkpoint), '--tokens', '4', '--output', str(saved)])
    assert capsys.readouterr().out == saved.read_text(encoding='utf-8')
    assert saved.read_text(encoding='utf-8').startswith('A ship')


@pytest.mark.parametrize('prompt', ['', '   '])
def test_cli_rejects_empty_prompt(prompt):
    with pytest.raises(SystemExit) as error:
        main(['generate', prompt])
    assert error.value.code == 2


def test_cli_missing_checkpoint_is_actionable(capsys):
    with pytest.raises(SystemExit):
        main(['generate', 'A ship', '--checkpoint', 'missing-checkpoint.pt'])
    assert 'missing-checkpoint.pt' in capsys.readouterr().err


def test_invalid_generation_settings():
    from app.model import GPT, GPTConfig
    model = GPT(GPTConfig(vocab_size=256, context_length=8, embedding_dim=16, heads=2, layers=1))
    for options in [{'temperature': float('nan')}, {'top_k': 0}, {'max_new_tokens': 0}]:
        with pytest.raises(ValueError):
            model.generate(torch.tensor([[1]]), **options)
