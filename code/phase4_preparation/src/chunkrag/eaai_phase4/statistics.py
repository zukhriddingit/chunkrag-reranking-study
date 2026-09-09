"""Frozen Phase 4 family-clustered statistical procedures."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def document_weighted_estimand(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("At least one document-level value is required")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def family_sign_flip_statistics(
    values: Sequence[float],
    families: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Flip all members of a family together and retain document weighting."""

    values_array = np.asarray(values, dtype=np.float64)
    family_array = np.asarray(families, dtype=object)
    if len(values_array) != len(family_array):
        raise ValueError("Values and family identifiers differ in length")
    unique_families = sorted(set(str(item) for item in families))
    family_to_index = {family: index for index, family in enumerate(unique_families)}
    document_family_indices = np.asarray([family_to_index[str(item)] for item in families])
    rng = np.random.default_rng(seed)
    output = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(unique_families))
        output[draw] = float(np.mean(values_array * signs[document_family_indices]))
    return output


def family_sign_flip_pvalue(
    values: Sequence[float], families: Sequence[str], *, draws: int, seed: int
) -> float:
    observed = abs(document_weighted_estimand(values))
    null = np.abs(family_sign_flip_statistics(values, families, draws=draws, seed=seed))
    return float((1 + np.sum(null >= observed)) / (int(draws) + 1))


def family_bootstrap_statistics(
    values: Sequence[float],
    families: Sequence[str],
    domains: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Stratified cluster bootstrap; families are never split within a draw.

    Every sampled occurrence of a family carries all its documents. The statistic
    in each replicate is the document-weighted mean of the resulting pseudo-sample.
    """

    values_array = np.asarray(values, dtype=np.float64)
    if not (len(values_array) == len(families) == len(domains)):
        raise ValueError("Values, family identifiers, and domains differ in length")
    domain_families: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (domain, family) in enumerate(zip(domains, families)):
        domain_families[str(domain)][str(family)].append(index)
    rng = np.random.default_rng(seed)
    output = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        repeated_indices: list[int] = []
        for domain in sorted(domain_families):
            family_map = domain_families[domain]
            keys = sorted(family_map)
            sampled = rng.choice(np.asarray(keys, dtype=object), size=len(keys), replace=True)
            for family in sampled:
                repeated_indices.extend(family_map[str(family)])
        output[draw] = float(np.mean(values_array[np.asarray(repeated_indices, dtype=int)]))
    return output


def percentile_interval(samples: Iterable[float], level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(list(samples), dtype=np.float64)
    alpha = (1.0 - float(level)) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper)
