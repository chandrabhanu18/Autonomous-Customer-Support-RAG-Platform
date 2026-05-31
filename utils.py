from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

_WORD_RE = re.compile(r"\b\w+\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text or "")]


def word_count(text: str) -> int:
    return len(tokenize(text))


def split_sentences(text: str) -> list[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    pieces = _SENTENCE_SPLIT_RE.split(stripped)
    return [piece.strip() for piece in pieces if piece.strip()]


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {token.lower() for token in left if token}
    right_set = {token.lower() for token in right if token}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(maximum, minimum):
        return [0.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def deterministic_embedding(text: str, dimensions: int = 1536) -> list[float]:
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dimensions
    values = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index in range(0, len(digest), 4):
            bucket = int.from_bytes(digest[index:index + 4], "little") % dimensions
            values[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
