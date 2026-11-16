"""Evaluate the published checkpoint and save unedited generated samples."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from app.training import batch, load_checkpoint, score, split_corpus


def evaluate(checkpoint='checkpoints/demo.pt', corpus='corpus'):
    torch.set_num_threads(4)
    model, tokenizer = load_checkpoint(checkpoint)
    training, validation = split_corpus(corpus)
    train_ids = torch.tensor(tokenizer.encode(training), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(validation), dtype=torch.long)
    generator = torch.Generator().manual_seed(43)
    batches = [batch(val_ids, 16, model.config.context_length, generator) for _ in range(10)]
    loss = score(model, batches)
    counts = torch.bincount(train_ids, minlength=tokenizer.vocab_size).float() + 1
    probabilities = counts / counts.sum()
    unigram_loss = sum((-probabilities[y].log()).mean().item() for _, y in batches) / len(batches)
    samples = []
    for prompt in ['The captain opened the map and', 'Alice stepped through the little door.', 'Beyond the dark forest, a hidden castle']:
        ids = tokenizer.encode(prompt)
        generated = model.generate(torch.tensor([ids]), max_new_tokens=100, temperature=0.7, top_k=30, seed=42)
        samples.append({'prompt': prompt, 'seed': 42, 'temperature': 0.7, 'top_k': 30,
                        'new_tokens': 100, 'output': prompt + tokenizer.decode(generated[0, len(ids):].tolist())})
    return {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'evaluation_seed': 43, 'validation_scored_tokens': sum(y.numel() for _, y in batches),
            'gpt_loss': loss, 'gpt_perplexity': math.exp(loss),
            'unigram_loss': unigram_loss, 'unigram_perplexity': math.exp(unigram_loss),
            'uniform_loss': math.log(tokenizer.vocab_size), 'uniform_perplexity': tokenizer.vocab_size,
            'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/demo.pt')
    parser.add_argument('--corpus', default='corpus')
    parser.add_argument('--output', type=Path, default=Path('results/evaluation.json'))
    args = parser.parse_args()
    output = json.dumps(evaluate(args.checkpoint, args.corpus), indent=2) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
