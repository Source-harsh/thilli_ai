"""Reproduce a held-out next-token comparison: python -m app.evaluate."""
import argparse
import json
from pathlib import Path
import random

from app.core import DEFAULT_CORPUS, END, START, NGramModel, load_corpus


def evaluate(corpus=DEFAULT_CORPUS, seed=42):
    dialogues = load_corpus(corpus)
    if len(dialogues) < 5:
        raise ValueError('Evaluation requires at least five utterances')
    random.Random(seed).shuffle(dialogues)
    split = int(len(dialogues) * 0.8)
    training, held_out = dialogues[:split], dialogues[split:]
    report = {'seed': seed, 'train_utterances': len(training),
              'test_utterances': len(held_out), 'models': {}}
    for name, order in [('unigram_baseline', 1), ('trigram', 3)]:
        model = NGramModel(n=order, alpha=0.0001).fit(training)
        correct = known = total = 0
        for dialogue in held_out:
            history = [START] * (order - 1)
            for target in dialogue + [END]:
                probabilities = model.distribution(history)
                prediction = max(probabilities, key=probabilities.get)
                correct += prediction == target
                known += target in model.vocabulary
                total += 1
                # Teacher forcing: score each token using its true preceding context.
                history.append(target)
        report['models'][name] = {'correct': correct, 'tokens': total,
                                 'accuracy': round(correct / total, 6),
                                 'vocabulary_coverage': round(known / total, 6)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = evaluate()
    # Train on the full corpus for illustrative samples, separate from held-out scoring.
    model = NGramModel(n=4).fit(load_corpus(character='AnimeHero'))
    report['samples'] = [
        {'prompt': prompt, 'seed': seed, 'order': 4, 'character': 'AnimeHero',
         'output': model.generate(prompt, seed=seed)}
        for prompt, seed in [('My rival', 12), ('Even if', 7), ('The demon king', 2)]]
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(output, encoding='utf-8')
    print(output, end='')


if __name__ == '__main__':
    main()
