from app.evaluate import evaluate


def test_evaluation_uses_held_out_targets(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('\n'.join(['Hero: red blue.'] * 10), encoding='utf-8')
    result = evaluate(corpus)
    assert result['train_utterances'] == 8
    assert result['test_utterances'] == 2
    baseline = result['models']['unigram_baseline']
    trigram = result['models']['trigram']
    assert baseline['tokens'] == trigram['tokens'] == 8
    assert baseline['correct'] == 2
    assert trigram['correct'] == 8
    assert evaluate(corpus) == result
