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

The generator itself uses only the Python standard library. `httpx` supports the starter's optional, unused `app.llm` helper. `.env` is unnecessary for n-gram generation.

## Completed milestones

- [x] Collect and preprocess a corpus of game dialogues.
- [x] Build the n-gram language model with smoothing.
- [x] Implement dialogue generation with sampling and backoff.
- [x] Build a CLI accepting a starting phrase.
- [x] Add length, alternative generation, character filtering, and saving output.

