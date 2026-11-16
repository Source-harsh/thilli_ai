[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/py3u6aau8k/badge.svg)](https://ask.thili.ai/b/py3u6aau8k)

# Adventure Story Weaver

A small decoder-only GPT, trained from scratch on public-domain adventure and fantasy books, that continues story prompts from a local CLI. It implements byte-level byte-pair encoding, learned token and position embeddings, multi-head causal self-attention, transformer blocks, backpropagation, and autoregressive sampling. No hosted API or pretrained language model is used.

## Run a story

From the repository root, use Python 3.12 or 3.13:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.cli generate "The captain opened the map and" --tokens 100 --seed 42
python -m app.cli generate "Alice stepped through the little door." --temperature 0.7 --top-k 30 --output story.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. A small trained checkpoint is included at `checkpoints/demo.pt`, so no training or network access is required after installing dependencies. Run commands from the repository root. Use `python -m app.cli --help` or `python -m app.cli generate --help` for options.

`--tokens` counts newly generated BPE tokens, not words. The exact starting phrase is preserved, and generation appends new tokens. `--temperature` adjusts randomness; lower values are more conservative. `--top-k` restricts each choice to the most likely tokens. `--seed` repeats a sampling run on the same runtime and checkpoint. Prompts longer than the context window are accepted; only their latest 128 tokens condition the next prediction.

## Train from scratch

```powershell
python -m app.prepare
python -m app.cli train --steps 3000 --output checkpoints/my-model.pt
python -m app.cli generate "Beyond the dark forest" --checkpoint checkpoints/my-model.pt
```

Training runs on CPU, defaults to four threads, and may take several minutes depending on your machine. `--threads` changes CPU parallelism. The demo uses a 384-token byte-BPE vocabulary, 128-token context, 128-dimensional embeddings, two transformer blocks, four attention heads, dropout 0.1, batch size 16, and AdamW with learning rate 0.001 and gradient clipping. Weights start from random initialization. A forward pass learns to predict each sequence's next token using cross-entropy; a triangular attention mask prevents looking ahead.

The tokenizer learns merges on training text only. The final 10% of each book is held out before learning merges or optimizing model weights. Validation uses ten fixed batches with evaluation seed 43. The training log records train and validation loss at step 0, every 250 steps, and the final step. The best validation checkpoint is saved, along with tokenizer JSON, configuration, token counts, corpus hashes, baseline losses, and timing. The checkpoint is for inference; optimizer state is not included for resuming an interrupted run.

## Results and evaluation

```powershell
python -m app.evaluate --output results/evaluation.json
python -m pytest -q
```

See [results/evaluation.json](results/evaluation.json) for measured held-out cross-entropy, perplexity, and unedited story samples. The [training report](checkpoints/demo.json) includes the learning curve, architecture, seed, and checkpoint SHA-256. Evaluation compares the GPT with a uniform token predictor and a smoothed unigram predictor fitted only on training tokens. All models score the same held-out BPE targets; lower loss and perplexity are better. Perplexity measures token prediction, not literary quality.

The included checkpoint was trained for 3,000 steps with seed 42 and scored on 20,480 held-out token positions:

| Predictor | Cross-entropy (nats/token) | Perplexity |
| --- | --- | --- |
| Uniform baseline | 5.9506 | 384.00 |
| Smoothed unigram baseline | 4.8339 | 125.69 |
| Custom GPT | 2.5582 | 12.91 |

Validation loss fell from 5.9829 at initialization to 2.5582. The model demonstrably learns token patterns, but these results do not establish coherent long-form story generation. The [CLI output](results/cli-story.txt) and JSON samples are real, unedited outputs and include imperfect grammar and invented word fragments.

This is an educational model trained on just two books. It can learn spelling and short narrative patterns, but it can also produce malformed words, repeat passages, and lose plot or character consistency. It is not a general-purpose story model. Validation is a later section of the same books and is used to select the checkpoint; it is not an independent final test set. Overlapping random evaluation windows are possible. Exact numerical reproduction across hardware or PyTorch builds is not guaranteed.

## Data and attribution

The corpus consists of *Treasure Island* by Robert Louis Stevenson (1883) and *Alice's Adventures in Wonderland* by Lewis Carroll (1865). These works are public domain in the United States. The [source notes](corpus/SOURCES.md) list download URLs, preprocessing, authors, and split policy. Raw downloads retain their Gutenberg notices and license. [corpus/manifest.json](corpus/manifest.json) records raw and cleaned hashes. The complete raw and cleaned corpus is versioned for offline reruns.

Preprocessing removes Gutenberg boilerplate, rejoins wrapped lines, and preserves case, punctuation, and paragraph boundaries. The reversible byte-level tokenizer handles new words and arbitrary UTF-8 prompts without an unknown-token vocabulary. Sampled byte fragments may decode with replacement characters if generation ends within a multibyte character.

This project began with the [askThili Adventure Story Weaver scaffold](https://ask.thili.ai/b/py3u6aau8k). The implementation, tests, and documentation were developed with AI assistance. The old Game Dialogue Assistant assignment remains on `main`; this branch contains only Adventure Story Weaver. No synthetic training stories or pretrained weights were added.

## Completed milestones

- [x] Assemble and preprocess a public-domain adventure/fantasy corpus.
- [x] Implement byte-pair encoding and learned embeddings.
- [x] Build causal self-attention and transformer blocks.
- [x] Implement backpropagation training and train a custom GPT.
- [x] Implement autoregressive story continuation with temperature and top-k sampling.
- [x] Provide a CLI accepting prompts and saving generated stories.

Tests cover tokenizer compression and Unicode round trips, corpus cleaning, future-token masking, learning through backpropagation, seeded generation, context cropping, checkpoint round trips, a tiny end-to-end training run, CLI file output, and invalid inputs.
