"""Дедупликация: точная и near-duplicate.

Точная ловит буквальные повторы, near-dup — переформулировки и перестановки
вариантов ответа. В курсовом источнике буквальных повторов нет, а почти-дублей
несколько процентов: одна только точная дедупликация здесь не делает ничего.
"""

from typing import Sequence

from datasketch import MinHash, MinHashLSH

from src.textnorm import shingles


def build_minhash(text: str, shingle_words: int, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    mh.update_batch([s.encode("utf-8") for s in shingles(text, shingle_words)])
    return mh


def jaccard(left: set[str], right: set[str]) -> float:
    """Точный Жаккар для проверки кандидатов, найденных приближённым LSH."""
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def exact_duplicates(keys: Sequence[str]) -> list[int]:
    """Индексы повторных вхождений. Первое вхождение остаётся."""
    seen: set[str] = set()
    dupes: list[int] = []
    for i, key in enumerate(keys):
        if key in seen:
            dupes.append(i)
        else:
            seen.add(key)
    return dupes


def near_duplicates(
    texts: Sequence[str], shingle_words: int, num_perm: int, threshold: float
) -> list[int]:
    """Индексы почти-дублей: жадный проход, первый представитель кластера остаётся.

    MinHash + LSH дают линейное время вместо O(n^2) полного попарного сравнения.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    representatives: dict[str, set[str]] = {}
    dupes: list[int] = []
    for i, text in enumerate(texts):
        text_shingles = shingles(text, shingle_words)
        mh = build_minhash(text, shingle_words, num_perm)
        candidates = lsh.query(mh)
        if any(jaccard(text_shingles, representatives[key]) >= threshold for key in candidates):
            dupes.append(i)
        else:
            key = str(i)
            lsh.insert(key, mh)
            representatives[key] = text_shingles
    return dupes


def cross_near_duplicates(
    left: Sequence[str],
    right: Sequence[str],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> list[tuple[int, int]]:
    """Пары (индекс в left, индекс в right) с Жаккаром выше порога.

    Используется проверкой контаминации: точное совпадение текстов ловит
    копипасту, а протекают обычно парафразы.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    left_shingles: list[set[str]] = []
    for i, text in enumerate(left):
        left_shingles.append(shingles(text, shingle_words))
        lsh.insert(str(i), build_minhash(text, shingle_words, num_perm))
    pairs: list[tuple[int, int]] = []
    for j, text in enumerate(right):
        right_shingles = shingles(text, shingle_words)
        for key in lsh.query(build_minhash(text, shingle_words, num_perm)):
            i = int(key)
            if jaccard(left_shingles[i], right_shingles) >= threshold:
                pairs.append((i, j))
    return pairs
