# Corpus provenance

Downloaded 2026-11-16 from GITenberg's Project Gutenberg archives because the Gutenberg website was under maintenance:

| Text | Author | First published | Raw source |
| --- | --- | --- | --- |
| Treasure Island | Robert Louis Stevenson | 1883 | https://raw.githubusercontent.com/GITenberg/Treasure-Island_120/master/120.txt |
| Alice's Adventures in Wonderland | Lewis Carroll | 1865 | https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11.txt |

These works are public domain in the United States. Raw downloads retain the Project Gutenberg notices and license. `manifest.json` records exact raw and cleaned SHA-256 hashes. No AI-written training stories or pretrained weights are used.

`python -m app.prepare` removes text outside the Gutenberg START/END markers, rejoins wrapped lines inside paragraphs, and preserves case, punctuation, paragraph breaks, and remaining book front matter. Both raw and cleaned text are versioned, so training requires no network download.

Training holds out the final 10% of each book by character position before learning BPE merges. The first 90% of each book trains both tokenizer and model. Validation therefore uses later passages of the same books, not unseen authors or an external adventure-game benchmark.
