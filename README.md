# Adventure Story Weaver

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/py3u6aau8k/badge.svg)](https://ask.thili.ai/b/py3u6aau8k)

This project guides you through building a custom GPT model from scratch to generate dynamic story continuations for text adventure games. You will train the model on existing adventure game narratives, applying core concepts like byte-pair tokenization, embeddings, and causal self-attention. The final product will be a CLI tool that takes a story prompt and weaves new narrative segments.

## What you are building

**Goal.** You will build a text adventure story segment generator using a custom GPT model, trained on adventure game narratives, to create dynamic story continuations.

**Done when.** The system can take a story prompt via a command-line interface and generate coherent, dynamic story continuations, demonstrating the custom GPT's ability to apply learned narrative structures.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Assemble a small corpus of text from public domain adventure stories or fantasy novels.**  
      Hint: Gather and clean raw text data for your corpus.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings from the story text.**  
      Hint: Apply byte-pair encoding and create token embeddings from your text.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Construct the causal self-attention layer and integrate it into a transformer block.
- [ ] **4. Set up the training loop and train the GPT model on the narrative data.**  
      Hint: Implement the training loop with backpropagation and optimization for your GPT model.
- [ ] **5. Implement a text generation feature that creates coherent story continuations.**  
      Hint: Use the trained model for autoregressive text generation to create continuations.
- [ ] **6. Design a CLI where users can input a story prompt and receive generated segments.**  
      Hint: Create a user-friendly command-line interface for story input and output.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify parameters for generation, like story mood or character focus.
- Implement a feature to save generated story segments to a file.
- Explore different decoding strategies (e.g., beam search) to vary the output quality.
