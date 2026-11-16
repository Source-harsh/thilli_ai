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

On macOS/Linux, activate with `source .venv/bin/activate`. A small trained checkpoint is included at `checkpoints/improved.pt` and is selected by default, so no training or network access is required after installing dependencies. The original checkpoint remains at `checkpoints/demo.pt` for comparison. Run commands from the repository root. Use `python -m app.cli --help` or `python -m app.cli generate --help` for options.

`--tokens` counts newly generated BPE tokens, not words. The exact starting phrase is preserved, and generation appends new tokens. `--temperature` adjusts randomness; lower values are more conservative. `--top-k` restricts each choice to the most likely tokens. `--seed` repeats a sampling run on the same runtime and checkpoint. Prompts longer than the context window are accepted; only their latest 128 tokens condition the next prediction.

`--top-p` adds nucleus sampling (default 0.9), keeping a set of likely tokens whose cumulative probability reaches the chosen threshold. `--repetition-penalty` discourages recently used tokens (default 1.05). Use `--top-p 1 --repetition-penalty 1` to disable these controls. They adjust sampling; they do not guarantee correct grammar.

## Improved checkpoint

The default checkpoint retains 750 additional training updates after the original 3,000, for 3,750 total. Training continued from the original weights with a new AdamW optimizer and learning rate 0.0003, decaying on a cosine schedule. The longer planned run was stopped at the user's request; the best saved checkpoint was retained. [Its report](checkpoints/improved.json) records the actual retained step, parent checkpoint hash, corpus hashes, and schedule.

Reproduce the before/after comparison with identical held-out targets and unchanged sampling settings:

```powershell
python -m app.compare
```

[results/improvement.json](results/improvement.json) records both models' measured losses, perplexities, checkpoint hashes, and unedited outputs for the same prompts. The comparison disables the new sampling controls to isolate the effect of further training. The original results below are retained as a baseline. This remains a small educational model with limited grammatical and narrative coherence.

On the same evaluation targets, further training reduces cross-entropy from 2.5582 to 2.4933 nats/token and lowers perplexity by **6.28%**.

Continue training without overwriting your starting checkpoint:

```powershell
python -m app.cli train --init-from checkpoints/improved.pt --output checkpoints/next.pt --steps 2000 --learning-rate 0.0002
```

This starts a fresh optimizer rather than restoring an exact interrupted optimizer state. It requires the matching training report and rejects a changed corpus or a checkpoint hash mismatch. New training uses cosine learning-rate decay; the original baseline used a fixed learning rate.

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

The original checkpoint was trained for 3,000 steps with seed 42 and scored on 20,480 held-out token positions:

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

### Implementation evidence

This repository's default branch is `Adventure-of-story-weaver`. It contains the transformer assignment; `main` preserves the older n-gram assignment. Submit the root repository URL so GitHub opens this assignment by default.

| Assignment milestone | Implementation | Verification |
| --- | --- | --- |
| Public-domain narrative corpus | [Corpus and source attribution](corpus/SOURCES.md), [`clean_text`](app/prepare.py) | [Cleaning and exact corpus hash tests](tests/test_tokenizer.py) |
| Byte-pair tokenizer and embeddings | [`BytePairTokenizer`](app/tokenizer.py), `GPT.token_embedding` and `GPT.position_embedding` in [model.py](app/model.py) | Unicode round trip, compression, serialization, and embedding-gradient tests |
| Causal attention and transformer blocks | `CausalSelfAttention` and `TransformerBlock` in [model.py](app/model.py) | [Future-token masking test](tests/test_model.py) proves earlier predictions cannot see later tokens |
| Train a custom GPT | [`train`](app/training.py): AdamW, cross-entropy, backpropagation, clipping, and validation checkpoint selection | [Original 3,000-step training log](checkpoints/demo.json), [continued-training checkpoint report](checkpoints/improved.json), and [training integration test](tests/test_training_cli.py) |
| Autoregressive continuation | `GPT.generate` in [model.py](app/model.py), with temperature, top-k, nucleus sampling, and repetition controls | [Unedited outputs and before/after measurements](results/improvement.json), seeded sampling and context-window tests |
| Prompt-driven CLI | [`train` and `generate` commands](app/cli.py) | [CLI checkpoint, saved output, and invalid-input tests](tests/test_training_cli.py) |

The neural pipeline is implemented rather than represented by milestone stubs. Output quality remains limited: the saved samples show that learned token patterns do not yet yield consistently coherent stories. The baseline and before/after results quantify token prediction, not successful completion of a human literary-quality assessment.

Tests cover tokenizer compression and Unicode round trips, corpus cleaning, future-token masking, learning through backpropagation, seeded generation, context cropping, checkpoint round trips, a tiny end-to-end training run, CLI file output, and invalid inputs.
