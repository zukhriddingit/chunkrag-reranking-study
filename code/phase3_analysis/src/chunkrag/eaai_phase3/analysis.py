from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Sequence

from chunkrag.eaai_phase3.constants import CONDITIONS
from chunkrag.eaai_phase3.evidence import ndcg_at_k
from chunkrag.eaai_phase3.statistics import paired_estimate


def select_reranker_hyperparameters(
    rows: Iterable[dict[str, Any]],
    *,
    learning_rates: Sequence[float],
    epochs: Sequence[int],
    expected_question_ids: set[str],
    chunkers: Sequence[str],
    k: int = 4,
) -> dict[str, Any]:
    expected_combinations = {
        (float(learning_rate), int(epoch))
        for learning_rate in learning_rates
        for epoch in epochs
    }
    grouped: dict[
        tuple[float, int],
        dict[tuple[str, str], list[dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    seen_candidates: set[tuple[float, int, str, str, str]] = set()
    for raw in rows:
        row = dict(raw)
        combination = (float(row["learning_rate"]), int(row["epochs"]))
        if combination not in expected_combinations:
            raise ValueError(f"Unexpected hyperparameter combination: {combination}")
        question_id = str(row["question_id"])
        chunker = str(row["chunker"])
        if question_id not in expected_question_ids or chunker not in chunkers:
            raise ValueError(f"Unexpected validation cell: {(question_id, chunker)}")
        score = float(row["score"])
        grade = int(row["grade"])
        if not math.isfinite(score) or grade not in {0, 1, 2}:
            raise ValueError("Reranker validation scores or grades are invalid")
        candidate_key = (*combination, question_id, chunker, str(row["chunk_id"]))
        if candidate_key in seen_candidates:
            raise ValueError(f"Duplicate reranker validation candidate: {candidate_key}")
        seen_candidates.add(candidate_key)
        grouped[combination][(question_id, chunker)].append(row)

    expected_cells = {
        (question_id, str(chunker))
        for question_id in expected_question_ids
        for chunker in chunkers
    }
    grid: list[dict[str, Any]] = []
    for learning_rate, epoch in sorted(expected_combinations):
        cells = grouped.get((learning_rate, epoch), {})
        if set(cells) != expected_cells:
            raise ValueError(
                "Reranker validation matrix is incomplete for "
                f"learning_rate={learning_rate}, epochs={epoch}"
            )
        values: list[float] = []
        for cell in sorted(cells):
            ranked = sorted(
                cells[cell],
                key=lambda row: (-float(row["score"]), str(row["chunk_id"])),
            )
            values.append(ndcg_at_k([int(row["grade"]) for row in ranked], k=k))
        grid.append(
            {
                "learning_rate": learning_rate,
                "epochs": epoch,
                "question_chunker_cells": len(values),
                "question_mean_graded_ndcg_at_4": mean(values),
            }
        )
    selected = min(
        grid,
        key=lambda row: (
            -float(row["question_mean_graded_ndcg_at_4"]),
            int(row["epochs"]),
            float(row["learning_rate"]),
        ),
    )
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "metric": "question_mean_graded_ndcg_at_4",
        "selected_learning_rate": selected["learning_rate"],
        "selected_epochs": selected["epochs"],
        "selected_metric": selected["question_mean_graded_ndcg_at_4"],
        "tie_break": ["fewer_epochs", "lower_learning_rate"],
        "grid": grid,
    }


def analyze_confirmatory_rows(
    rows: Iterable[dict[str, Any]],
    *,
    expected_question_ids: set[str],
    chunkers: Sequence[str],
    bootstrap_draws: int,
    bootstrap_seed: int,
    randomization_draws: int,
    randomization_seed: int,
    minimum_effect: float,
    alpha: float,
) -> dict[str, Any]:
    expected_keys = {
        (question_id, chunker, condition)
        for question_id in expected_question_ids
        for chunker in chunkers
        for condition in CONDITIONS
    }
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["question_id"]), str(row["chunker"]), str(row["condition"]))
        if key in index:
            raise ValueError(f"Duplicate confirmatory matrix cell: {key}")
        f1 = float(row["f1"])
        if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
            raise ValueError(f"Invalid F1 for {key}: {f1}")
        index[key] = dict(row)
    if set(index) != expected_keys:
        missing = sorted(expected_keys - set(index))[:5]
        extra = sorted(set(index) - expected_keys)[:5]
        raise ValueError(f"Incomplete or unexpected confirmatory matrix: missing={missing}, extra={extra}")

    question_rows: list[dict[str, Any]] = []
    differences: list[float] = []
    for question_id in sorted(expected_question_ids):
        comparator = mean(float(index[(question_id, chunker, "comparator")]["f1"]) for chunker in chunkers)
        evidence = mean(float(index[(question_id, chunker, "evidence_aware")]["f1"]) for chunker in chunkers)
        difference = evidence - comparator
        differences.append(difference)
        question_rows.append(
            {
                "question_id": question_id,
                "comparator_f1": comparator,
                "evidence_aware_f1": evidence,
                "delta_f1": difference,
            }
        )
    estimate = paired_estimate(
        differences,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        randomization_draws=randomization_draws,
        randomization_seed=randomization_seed,
    ).as_dict()
    p_value = estimate["randomization_p"]
    success = bool(
        estimate["mean_difference"] >= minimum_effect
        and p_value is not None
        and p_value < alpha
    )
    return {
        "schema_version": 1,
        "analysis_classification": "primary_confirmatory",
        "endpoint": "token_f1",
        "direction": "evidence_aware_minus_comparator",
        "sampling_unit": "question",
        "aggregation": "mean_over_four_chunkers_within_question",
        "condition_means": {
            "comparator": mean(row["comparator_f1"] for row in question_rows),
            "evidence_aware": mean(row["evidence_aware_f1"] for row in question_rows),
        },
        "minimum_effect": minimum_effect,
        "alpha": alpha,
        "estimate": estimate,
        "confirmatory_success": success,
        "question_level": question_rows,
    }


def select_comparator(
    rows: Iterable[dict[str, Any]],
    *,
    baseline_ids: Sequence[str],
    tie_break_order: Sequence[str],
    chunkers: Sequence[str],
    expected_question_ids: set[str],
) -> dict[str, Any]:
    if set(baseline_ids) != set(tie_break_order):
        raise ValueError("Comparator tie-break order must contain every baseline exactly once")
    values: dict[str, dict[str, list[float]]] = {
        baseline: defaultdict(list) for baseline in baseline_ids
    }
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        baseline = str(row["condition"])
        key = (str(row["question_id"]), str(row["chunker"]), baseline)
        if baseline not in values or key in seen:
            raise ValueError(f"Invalid or duplicate validation row: {key}")
        seen.add(key)
        values[baseline][str(row["question_id"])].append(float(row["f1"]))
    expected = {
        (question_id, chunker, baseline)
        for question_id in expected_question_ids
        for chunker in chunkers
        for baseline in baseline_ids
    }
    if seen != expected:
        raise ValueError("Validation comparator matrix is incomplete")
    means = {
        baseline: mean(mean(values[baseline][question_id]) for question_id in sorted(expected_question_ids))
        for baseline in baseline_ids
    }
    order = {baseline: index for index, baseline in enumerate(tie_break_order)}
    selected = min(baseline_ids, key=lambda baseline: (-means[baseline], order[baseline]))
    return {
        "schema_version": 1,
        "selected_comparator": selected,
        "endpoint": "question_mean_token_f1_over_four_chunkers",
        "split": "validation",
        "condition_means": means,
        "tie_break_order": list(tie_break_order),
    }
