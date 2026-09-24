from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from bench.engine.leakage_gate import (
    DEFAULT_THRESHOLDS,
    LEAKAGE_UNIT,
    evaluate_leakage,
    leakage_score,
)
from bench.engine.metrics import fidelity
from bench.engine.protocol import PROTOCOL_VERSION, scoring_profile
from bench.engine.scoring_core import (
    MAX_NORMALIZED_TEXT_CHARACTERS,
    MAX_QUADRATIC_CELLS,
    MAX_SCORING_TEXT_CHARACTERS,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "bench/conformance/scorer-v0.2.json"


class ScorerConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_declares_current_profiles_and_vector_counts(self) -> None:
        self.assertEqual(self.fixture["protocolVersion"], PROTOCOL_VERSION)
        self.assertEqual(self.fixture["scoringProfile"], scoring_profile())
        self.assertEqual(self.fixture["scoringProfile"]["leakageUnit"], LEAKAGE_UNIT)
        self.assertEqual(self.fixture["leakageThresholds"], DEFAULT_THRESHOLDS)
        self.assertEqual(len(self.fixture["fidelityVectors"]), 14)
        self.assertEqual(len(self.fixture["leakageVectors"]), 28)

    def test_fidelity_vectors(self) -> None:
        supported = set(self.fixture["supportedMetricClasses"])
        for vector in self.fixture["fidelityVectors"]:
            self.assertEqual(set(vector["expected"]), supported, vector["id"])
            for metric_class, expected in vector["expected"].items():
                with self.subTest(vector=vector["id"], metric_class=metric_class):
                    self.assertEqual(
                        fidelity(vector["target"], vector["output"], metric_class),
                        expected,
                    )

    def test_leakage_vectors(self) -> None:
        thresholds = self.fixture["leakageThresholds"]
        for vector in self.fixture["leakageVectors"]:
            with self.subTest(vector=vector["id"]):
                actual = evaluate_leakage(
                    vector["prompt"],
                    vector["target"],
                    thresholds=thresholds,
                ).as_dict()
                self.assertEqual(
                    actual,
                    {**vector["expected"], "thresholds": thresholds},
                )

    def test_unknown_metric_classes_fail_closed(self) -> None:
        for metric_class in self.fixture["unsupportedMetricClasses"]:
            with self.subTest(metric_class=metric_class):
                with self.assertRaisesRegex(ValueError, "unsupported metric class"):
                    fidelity("target", "output", metric_class)

    def test_text_limits_fail_closed_before_quadratic_scoring(self) -> None:
        oversized = "x" * (MAX_SCORING_TEXT_CHARACTERS + 1)
        with self.assertRaisesRegex(ValueError, "scorer limit"):
            fidelity("target", oversized, "normalized_edit_similarity")
        with self.assertRaisesRegex(ValueError, "scorer limit"):
            evaluate_leakage(oversized, "target", thresholds=DEFAULT_THRESHOLDS)

        side = int(MAX_QUADRATIC_CELLS**0.5)
        self.assertEqual(
            fidelity("x" * side, "x" * side, "normalized_edit_similarity"),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "quadratic cells"):
            fidelity(
                "x" * (side + 1),
                "x" * (side + 1),
                "normalized_edit_similarity",
            )

        compatibility_expander = "\ufdfa"
        with self.assertRaisesRegex(ValueError, "quadratic cells"):
            evaluate_leakage(
                compatibility_expander * 200,
                compatibility_expander * 200,
                thresholds=DEFAULT_THRESHOLDS,
            )
        expansion_count = MAX_NORMALIZED_TEXT_CHARACTERS // 18 + 1
        with self.assertRaisesRegex(ValueError, "normalized scorer limit"):
            evaluate_leakage(
                compatibility_expander * expansion_count,
                "",
                thresholds=DEFAULT_THRESHOLDS,
            )

    def test_huge_integer_threshold_fails_as_validation_not_overflow(self) -> None:
        thresholds = dict(DEFAULT_THRESHOLDS)
        thresholds["lcsRatio"] = 10**1000
        with self.assertRaisesRegex(ValueError, "finite number in"):
            evaluate_leakage("prompt", "target", thresholds=thresholds)

    def test_fail_closed_policy_has_maximum_diagnostic_score(self) -> None:
        result = evaluate_leakage(
            "safe-looking\u202eprompt",
            "unrelated target",
            thresholds=DEFAULT_THRESHOLDS,
        )
        self.assertEqual(result.failClosedReason, "bidi_control")
        self.assertEqual(leakage_score(result), 1.0)


if __name__ == "__main__":
    unittest.main()
