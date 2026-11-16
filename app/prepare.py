"""Remove Gutenberg boilerplate and normalize whitespace without changing case."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def clean_text(raw):
    start = re.search(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*', raw)
    end = re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*', raw)
    if not start or not end or end.start() <= start.end():
        raise ValueError('Missing Gutenberg start/end markers')
    text = raw[start.end():end.start()].replace('\r\n', '\n')
    paragraphs = re.split(r'\n\s*\n', text)
    return '\n\n'.join(re.sub(r'\s+', ' ', p).strip() for p in paragraphs if p.strip()) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('corpus'))
    args = parser.parse_args()
    metadata = []
    for name in ['treasure-island', 'alice']:
        raw = (args.directory / f'{name}.raw.txt').read_bytes()
        clean = clean_text(raw.decode('utf-8-sig'))
        (args.directory / f'{name}.txt').write_text(clean, encoding='utf-8')
        metadata.append({'file': f'{name}.txt', 'characters': len(clean),
                         'raw_sha256': hashlib.sha256(raw).hexdigest(),
                         'clean_sha256': hashlib.sha256(clean.encode('utf-8')).hexdigest()})
    (args.directory / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
