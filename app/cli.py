"""Train a custom GPT or generate a story continuation locally."""
import argparse
import json
from pathlib import Path
import torch
from app.training import load_checkpoint, train


def main(argv=None):
    parser = argparse.ArgumentParser(prog='adventure-story-weaver', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train', help='train from scratch on the bundled books')
    training.add_argument('--corpus', type=Path, default=Path('corpus'))
    training.add_argument('--output', type=Path, default=Path('checkpoints/demo.pt'))
    training.add_argument('--steps', type=int, default=3000)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--threads', type=int, default=4)
    generation = commands.add_parser('generate', help='continue a story using a trained checkpoint')
    generation.add_argument('prompt')
    generation.add_argument('--checkpoint', type=Path, default=Path('checkpoints/demo.pt'))
    generation.add_argument('--tokens', type=int, default=120)
    generation.add_argument('--temperature', type=float, default=0.8)
    generation.add_argument('--top-k', type=int, default=40)
    generation.add_argument('--seed', type=int, default=42)
    generation.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'train':
            report = train(args.corpus, args.output, args.steps, args.seed, threads=args.threads)
            print(json.dumps({'best_validation_loss': report['best_validation_loss']}))
        else:
            if not args.prompt.strip():
                raise ValueError('Provide a nonempty story prompt')
            torch.set_num_threads(4)
            model, tokenizer = load_checkpoint(args.checkpoint)
            ids = tokenizer.encode(args.prompt)
            generated = model.generate(torch.tensor([ids]), args.tokens, args.temperature, args.top_k, args.seed)
            # Decode only the new tokens so the exact supplied prompt is preserved.
            output = args.prompt + tokenizer.decode(generated[0, len(ids):].tolist())
            if args.output:
                args.output.write_text(output + '\n', encoding='utf-8')
            print(output)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
