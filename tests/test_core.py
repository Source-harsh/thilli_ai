import pytest
from app.core import NGramModel, load_corpus, tokenize
from app.cli import main


def test_preprocessing_and_character_filter(tmp_path):
    corpus = tmp_path / 'script.txt'
    corpus.write_text("# comment\nGuard: Halt, traveler!\nMage: Don't fear.\n", encoding='utf-8')
    assert load_corpus(corpus, 'guard') == [['halt', ',', 'traveler', '!']]
    assert tokenize("Don't fear.") == ["don't", 'fear', '.']
    with pytest.raises(ValueError):
        load_corpus(corpus, 'missing')


def test_smoothing_and_backoff():
    model = NGramModel(alpha=0.1).fit([['hello', 'traveler'], ['hello', 'friend']])
    probabilities = model.distribution(['hello'])
    assert sum(probabilities.values()) == pytest.approx(1)
    assert all(value > 0 for value in probabilities.values())
    assert probabilities['traveler'] > probabilities['hello']
    assert model.distribution(['unseen', 'hello']) == probabilities
    assert sum(model.distribution(['unknown']).values()) == pytest.approx(1)


def test_generation_reproducible_and_bounded():
    model = NGramModel(alpha=0.000001).fit([['hello', 'brave', 'traveler']])
    assert model.generate('hello', seed=1) == 'Hello brave traveler.'
    assert model.generate('hello', max_words=1, seed=1) == 'Hello brave.'
    assert model.generate('hello', seed=1) == model.generate('hello', seed=1)
    assert '<s>' not in model.generate('unknown', seed=2)


def test_cli_saves_alternatives(tmp_path, capsys):
    target = tmp_path / 'dialogue.txt'
    main(['generate', 'The forest', '--character', 'Mage', '--count', '2', '--seed', '4', '--output', str(target)])
    output = capsys.readouterr().out
    assert target.read_text(encoding='utf-8') == output
    assert len(output.splitlines()) == 2


@pytest.mark.parametrize('options', [['--length', '0'], ['--count', '0'], ['--order', '6'], ['--alpha', 'nan'], ['--character', 'missing']])
def test_cli_reports_invalid_input(options):
    with pytest.raises(SystemExit) as error:
        main(['generate', 'hello', *options])
    assert error.value.code == 2


def test_empty_training_rejected():
    with pytest.raises(ValueError):
        NGramModel().fit([])
    with pytest.raises(ValueError):
        NGramModel().generate('hello')


def test_default_corpus():
    assert len(load_corpus()) == 50
    assert len(load_corpus(character='Guard')) == 10
    assert len(load_corpus(character='AnimeHero')) == 20
