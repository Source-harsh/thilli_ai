"""Compare original and improved checkpoints on identical targets and prompts."""
import argparse
import json
from pathlib import Path
from app.evaluate import evaluate
from app.training import load_checkpoint


def compare(before, after, corpus='corpus'):
    old_model, old_tokenizer = load_checkpoint(before)
    new_model, new_tokenizer = load_checkpoint(after)
    if (old_tokenizer.merges != new_tokenizer.merges or
            old_model.config.context_length != new_model.config.context_length):
        raise ValueError('Fair token-level comparison requires the same tokenizer and context length')
    old = evaluate(before, corpus)
    new = evaluate(after, corpus)
    return {'method': 'Identical held-out batches and generation prompts; sampling controls unchanged.',
            'before': old, 'after': new,
            'loss_reduction': old['gpt_loss'] - new['gpt_loss'],
            'perplexity_reduction_percent': 100 * (1 - new['gpt_perplexity'] / old['gpt_perplexity'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', default='checkpoints/demo.pt')
    parser.add_argument('--after', default='checkpoints/improved.pt')
    parser.add_argument('--corpus', default='corpus')
    parser.add_argument('--output', type=Path, default=Path('results/improvement.json'))
    args = parser.parse_args()
    report = compare(args.before, args.after, args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('before', 'after')}, indent=2))


if __name__ == '__main__':
    main()
