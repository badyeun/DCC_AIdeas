"""Text-only turn boundaries for the independent ver8 encoder.

The shared source loader has already normalized each utterance and joined the
turns with newlines. This representation deliberately accepts text strings,
never annotations, so labels and speaker metadata cannot become features.
"""
from __future__ import annotations

import unicodedata


def candidate_texts(texts, tokenizer):
    """Insert the tokenizer's existing SEP token between nonempty turns.

    No vocabulary token is added and no maximum dialogue length is imposed.
    An empty dialogue remains empty and obtains the tokenizer's normal outer
    special tokens when it is tokenized.
    """
    separator = getattr(tokenizer, "sep_token", None)
    if not isinstance(separator, str) or not separator:
        raise ValueError("The candidate tokenizer must define an existing sep_token")
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of dialogue strings")
    rendered = []
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("Candidate inputs must be text strings, never annotations")
        turns = [" ".join(unicodedata.normalize("NFC", turn).split()) for turn in text.split("\n")]
        rendered.append(f" {separator} ".join(turn for turn in turns if turn))
    return rendered
