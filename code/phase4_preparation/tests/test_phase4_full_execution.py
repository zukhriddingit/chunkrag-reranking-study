from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chunkrag.eaai_phase4.full_execution import (
    AllocationLedger,
    BudgetStop,
    ResumableJsonlStore,
    deterministic_group_draws,
    evidence_grade,
    expected_generation_keys,
    family_key,
    frozen_primary_analysis,
    ndcg_at_k,
    select_shared_configuration,
)


class FullExecutionContractTests(unittest.TestCase):
    def test_corrected_ndcg_regression_and_edges(self) -> None:
        pool = [1, 1, 1, 1, 2] + [0] * 15
        self.assertAlmostEqual(ndcg_at_k(pool[:4], 4), 1.0)
        self.assertAlmostEqual(ndcg_at_k(pool, 4), 0.5615579549, places=10)
        self.assertEqual(ndcg_at_k([0] * 20, 4), 0.0)
        self.assertEqual(ndcg_at_k([2, 1, 0], 4), 1.0)
        self.assertEqual(ndcg_at_k([2, 1, 1, 0, 0], 4), 1.0)

    def test_deterministic_sampler_cycles_complete_epochs(self) -> None:
        draws = deterministic_group_draws(7, 20, 20260904)
        self.assertEqual(len(draws), 20)
        self.assertEqual(set(draws[:7]), set(range(7)))
        self.assertEqual(set(draws[7:14]), set(range(7)))
        self.assertEqual(draws, deterministic_group_draws(7, 20, 20260904))

    def test_append_only_resume_and_duplicate_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            store = ResumableJsonlStore(path, ["key"])
            self.assertEqual(store.append({"key": "a", "value": 1}), "appended")
            self.assertEqual(store.append({"key": "a", "value": 1}), "skipped_identical")
            with self.assertRaises(RuntimeError):
                store.append({"key": "a", "value": 2})
            resumed = ResumableJsonlStore(path, ["key"])
            self.assertEqual(len(resumed.rows), 1)

    def test_exact_generation_matrix(self) -> None:
        keys = expected_generation_keys([f"d{i}" for i in range(273)])
        self.assertEqual(len(keys), 8736)
        self.assertIn(("E_20260906", "d1", "sentence_254"), keys)

    def test_evidence_grades(self) -> None:
        self.assertEqual(evidence_grade("Use command alpha 7", "other", "gold", ["command alpha 7"]), 2)
        self.assertEqual(evidence_grade("unrelated", "gold", "gold", ["command alpha 7"]), 1)
        self.assertEqual(evidence_grade("unrelated", "other", "gold", ["command alpha 7"]), 0)

    def test_family_scope_is_domain_specific(self) -> None:
        self.assertEqual(family_key("SSA", "Benefits #12"), "ssa::benefits")
        self.assertNotEqual(family_key("SSA", "Benefits #12"), family_key("VA", "Benefits #1"))

    def test_shared_selection_and_ties(self) -> None:
        rows = []
        for method in ["B", "E"]:
            for lr in [0.00005, 0.0001]:
                for updates in [227, 454, 681]:
                    rows.append({
                        "method": method,
                        "learning_rate": lr,
                        "optimizer_updates": updates,
                        "validation_question_mean_common_evidence_ndcg_at_4": 0.5,
                    })
        selected = select_shared_configuration(rows)["selected"]
        self.assertEqual(selected["optimizer_updates"], 227)
        self.assertEqual(selected["learning_rate"], 0.00005)

    def test_clustered_analysis_is_deterministic(self) -> None:
        rows = []
        for index in range(273):
            rows.append({
                "difference": 0.02 if index % 2 else 0.0,
                "family_key": f"d::{index // 2}",
                "domain": ["dmv", "ssa", "studentaid", "va"][index % 4],
                "seed_differences": {"20260904": 0.01, "20260905": 0.02, "20260906": 0.0},
            })
        first = frozen_primary_analysis(rows)
        second = frozen_primary_analysis(rows)
        self.assertEqual(first, second)
        self.assertEqual(first["documents"], 273)
        self.assertEqual(first["families"], 137)

    def test_budget_ceiling_and_recovery_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocation.json"
            ledger = AllocationLedger(path, ceiling_hours=20.0)
            ledger.begin(stage="freeze", hardware="NVIDIA A100-SXM4-40GB")
            self.assertLess(ledger.heartbeat(stage="freeze"), 1.0)
            with self.assertRaises(BudgetStop):
                ledger.heartbeat(stage="freeze", projected_remaining_seconds=20 * 3600 + 1)
            final = ledger.end(status="test", stage="freeze")
            self.assertEqual(final["sessions"][0]["status"], "test")

    def test_frozen_runner_contains_no_outcome_selection(self) -> None:
        runner = (Path(__file__).parents[1] / "scripts/run_phase4_full.py").read_text(encoding="utf-8")
        selection_start = runner.index("def seal_selection")
        selection_end = runner.index("def run_refits")
        selection_source = runner[selection_start:selection_end]
        self.assertNotIn("generations.jsonl", selection_source)
        self.assertIn("Doc2Dial outcomes exist before configuration selection", selection_source)


if __name__ == "__main__":
    unittest.main()
