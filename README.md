[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ymqyzkmhwt/badge.svg)](https://ask.thili.ai/b/ymqyzkmhwt)

# Game Dialogue Assistant

An offline Python CLI for prototyping NPC dialogue using a smoothed n-gram language model. No API keys, model downloads, or external service required.

## Run

From this project folder with Python 3.12 or newer:

```powershell
python -m app.cli generate "The forest" --character Mage --seed 4
python -m app.cli generate "Welcome traveler" --character Merchant --count 3 --length 25
python -m app.cli generate "The road" --output dialogue.txt
```

Use `python -m app.cli generate --help` for all options. `--length` caps additional words, excluding the prompt. `--seed` makes results reproducible. `--order` selects an n-gram order from 1 to 5 (default 3). `--alpha` controls additive smoothing (default 0.0001).

## Your own game scripts

Pass `--corpus path/to/script.txt`. Use UTF-8 text with one utterance per line. Optional speaker labels allow `--character` filtering (case insensitive):

```text
Guard: Halt traveler, state your business.
Mage: The forest remembers every footstep.
Merchant: Take a look at my wares.
```

Blank lines and lines beginning with `#` are ignored. Unlabeled lines are accepted when no character filter is selected. Speaker names are removed before training. Use scripts you have permission to use.

The bundled corpus contains 50 original utterances across Guard, Mage, Merchant, and AnimeHero. AnimeHero adds dramatic battle cries, friendship speeches, rival banter, and comedy inspired by anime adventure stories. Try `python -m app.cli generate "Even if" --character AnimeHero --seed 7`. The model tokenizes words and punctuation, counts contexts without crossing utterance boundaries, applies additive smoothing, and backs off to shorter contexts for unfamiliar prompts. Training happens on each invocation. A small n-gram corpus can repeat phrases or produce awkward combinations; a larger consistent corpus improves results. This is a prototype, not a semantic story model.

## Validation

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The generator and evaluation use only the Python standard library. `requirements.txt` pins the test runner and its dependencies. No `.env` configuration is needed. The implementation was validated locally on Python 3.13.2; GitHub Actions runs Python 3.12.

## Results and baseline comparison

Reproduce the saved results from the repository root:

```powershell
python -m app.evaluate --output results/evaluation.json
```

The script shuffles the 50 utterances with seed 42, trains on 40, and evaluates on the remaining 10. Both models use additive smoothing with alpha 0.0001. Each predicts the most probable next token from the true preceding tokens (teacher forcing); ties use alphabetical token order. The unigram baseline ignores context. The trigram uses up to two preceding tokens with suffix backoff.

| Model | Correct / test tokens | Next-token accuracy |
| --- | --- | --- |
| Unigram baseline | 13 / 138 | 9.42% |
| Trigram with backoff | 42 / 138 | 30.43% |

Tokens include punctuation and utterance-end markers. Only 74.64% of test tokens occur in the training vocabulary; unseen tokens count as incorrect. The split is by utterance, not by character or phrase family, so related phrases can appear in both sets. This single small synthetic split measures local prediction, not human-rated coherence or performance on real game scripts.

The [saved evaluation](results/evaluation.json) includes actual outputs with prompts, seeds, and model settings. For example:

```text
Prompt: Even if | character: AnimeHero | order: 4 | seed: 7
Even if my sword breaks, my courage will still have an edge.

Prompt: The demon king | character: AnimeHero | order: 4 | seed: 2
The demon king stole the sun. fine, we will become the dawn.
```

These examples train on all AnimeHero lines and are separate from held-out scoring. The model can copy training phrases, produce duplicate alternatives, combine phrases awkwardly, and leave sentence-internal capitalization lowercase. Higher prediction accuracy does not establish originality or story consistency.

## Data provenance and project origin

The complete dataset is versioned in [app/dialogues.txt](app/dialogues.txt): 10 Guard, 10 Merchant, 10 Mage, and 20 AnimeHero utterances. These are synthetic English examples authored with AI assistance for this project, not collected from published games or anime transcripts. AnimeHero uses general adventure themes rather than quotations from a specific series. There was no scraping or external dataset download. Preprocessing removes speaker labels, lowercases text, keeps word-internal apostrophes, and splits punctuation into tokens; blank lines and comments are skipped.

This project began from the [askThili project scaffold](https://ask.thili.ai/b/ymqyzkmhwt). The initial scaffold supplied milestone instructions, CLI plumbing, CI, and generic helpers. The implemented n-gram model, corpus, generation CLI, evaluation, and project-specific tests were developed with AI assistance. Unused API and storage helpers and their configuration have been removed. The scaffold origin and AI assistance are disclosed here rather than presented as entirely independent work.

## Completed milestones

- [x] Collect and preprocess a corpus of game dialogues.
- [x] Build the n-gram language model with smoothing.
- [x] Implement dialogue generation with sampling and backoff.
- [x] Build a CLI accepting a starting phrase.
- [x] Add length, alternative generation, character filtering, and saving output.

