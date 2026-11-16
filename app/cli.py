"""Command-line entry point: python -m app.cli --help."""
import argparse
from pathlib import Path
from app.core import DEFAULT_CORPUS, NGramModel, load_corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate NPC dialogue with an offline n-gram model.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('hello', help='check the application')
    generate = sub.add_parser('generate', help='continue a starting phrase')
    generate.add_argument('prompt', help='starting phrase (use quotes)')
    generate.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    generate.add_argument('--character', help='filter Character: dialogue lines')
    generate.add_argument('--length', type=int, default=30, help='maximum additional words')
    generate.add_argument('--count', type=int, default=1, help='number of alternatives')
    generate.add_argument('--order', type=int, default=3, help='n-gram order, 1 to 5')
    generate.add_argument('--alpha', type=float, default=0.0001, help='positive additive smoothing strength')
    generate.add_argument('--seed', type=int, help='reproduce a generation')
    generate.add_argument('--output', type=Path, help='save UTF-8 dialogue to a file')
    args = parser.parse_args(argv)
    if args.command == 'hello':
        print('Game Dialogue Assistant is ready.')
        return
    if args.length < 1 or args.count < 1:
        parser.error('--length and --count must be positive')
    try:
        model = NGramModel(args.order, args.alpha).fit(load_corpus(args.corpus, args.character))
        lines = [model.generate(args.prompt, args.length, None if args.seed is None else args.seed+i) for i in range(args.count)]
        output = '\n'.join(lines)
        if args.output:
            args.output.write_text(output + '\n', encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)


if __name__ == '__main__':
    main()

