from app.tokenizer import BytePairTokenizer
from app.prepare import clean_text


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
