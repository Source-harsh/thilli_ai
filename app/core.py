"""Offline n-gram dialogue model with additive smoothing and suffix backoff."""
from collections import Counter, defaultdict
from pathlib import Path
import math
import random
import re

DEFAULT_CORPUS = Path(__file__).with_name('dialogues.txt')
START, END = '<s>', '</s>'


def tokenize(text):
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*|[.,!?;:]", text.lower())


def load_corpus(path=DEFAULT_CORPUS, character=None):
    dialogues = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        speaker, separator, utterance = line.partition(':')
        if character and (not separator or speaker.strip().casefold() != character.casefold()):
            continue
        tokens = tokenize(utterance if separator else line)
        if tokens:
            dialogues.append(tokens)
    if not dialogues:
        raise ValueError('No dialogue found. Check the corpus and character name.')
    return dialogues


class NGramModel:
    def __init__(self, n=3, alpha=0.0001):
        if n < 1 or n > 5:
            raise ValueError('n must be between 1 and 5')
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        self.n, self.alpha = n, alpha
        self.counts = defaultdict(Counter)
        self.vocabulary = set()

    def fit(self, dialogues):
        self.counts.clear()
        self.vocabulary.clear()
        for dialogue in dialogues:
            if not dialogue:
                continue
            tokens = [START] * (self.n - 1) + dialogue + [END]
            self.vocabulary.update(dialogue + [END])
            for index in range(self.n - 1, len(tokens)):
                for size in range(self.n):
                    context = tuple(tokens[index-size:index]) if size else ()
                    self.counts[context][tokens[index]] += 1
        if not self.vocabulary:
            raise ValueError('Training requires nonempty dialogue')
        return self

    def distribution(self, context):
        if not self.vocabulary:
            raise ValueError('Train the model first')
        history = tuple(context[-(self.n-1):]) if self.n > 1 else ()
        while history and history not in self.counts:
            history = history[1:]
        counts = self.counts[history]
        total = sum(counts.values()) + self.alpha * len(self.vocabulary)
        return {word: (counts[word] + self.alpha) / total for word in sorted(self.vocabulary)}

    def generate(self, prompt, max_words=30, seed=None):
        if max_words < 1:
            raise ValueError('max_words must be positive')
        rng = random.Random(seed)
        history = [START] * (self.n-1) + tokenize(prompt)
        result, words = [], 0
        for _ in range(max_words * 4):
            probabilities = self.distribution(history)
            word = rng.choices(list(probabilities), weights=list(probabilities.values()))[0]
            if word == END:
                break
            history.append(word)
            result.append(word)
            if word not in '.,!?;:':
                words += 1
            if words >= max_words:
                break
        continuation = re.sub(r'\s+([.,!?;:])', r'\1', ' '.join(result))
        text = prompt.strip()
        if continuation:
            text += ('' if not text or continuation[0] in '.,!?;:' else ' ') + continuation
        if text:
            text = text[0].upper() + text[1:]
            if text[-1] not in '.!?':
                text += '.'
        return text

