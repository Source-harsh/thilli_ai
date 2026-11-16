from app.tokenizer import BytePairTokenizer
from app.prepare import clean_text
import hashlib
import json
from pathlib import Path


def test_bpe_roundtrip_and_compression(tmp_path):
    text = 'the hidden island and the hidden treasure. ' * 20
    tokenizer = BytePairTokenizer.train(text, 280)
    assert len(tokenizer.encode(text)) < len(text.encode())
    assert tokenizer.decode(tokenizer.encode('Unknown dragon: café 日本語!')) == 'Unknown dragon: café 日本語!'
    path = tmp_path / 'tokenizer.json'
    tokenizer.save(path)
    assert BytePairTokenizer.load(path).encode(text) == tokenizer.encode(text)


def test_cleaning_removes_license_boilerplate():
    raw = 'header\n*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nA dark\nforest.\n\nA cave.\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nlicense'
    assert clean_text(raw) == 'A dark forest.\n\nA cave.\n'


def test_versioned_corpus_matches_manifest():
    directory = Path(__file__).resolve().parents[1] / 'corpus'
    for entry in json.loads((directory / 'manifest.json').read_text()):
        raw = directory / entry['file'].replace('.txt', '.raw.txt')
        clean = directory / entry['file']
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == entry['raw_sha256']
        assert hashlib.sha256(clean.read_bytes()).hexdigest() == entry['clean_sha256']
        assert clean_text(raw.read_text(encoding='utf-8-sig')) == clean.read_text(encoding='utf-8')
