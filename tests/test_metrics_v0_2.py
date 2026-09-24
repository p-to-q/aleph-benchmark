from __future__ import annotations

import importlib
import re
import sys
import unicodedata
import unittest
from pathlib import Path

from bench.engine import metrics


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DOCUMENT = REPOSITORY_ROOT / "docs/protocol-v0.2.md"
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


class MetricsV02Tests(unittest.TestCase):
    def test_ascii_span_proxy_is_explicit_and_stable(self) -> None:
        self.assertEqual(metrics.token_count("alpha beta-42"), 3)
        self.assertEqual(metrics.token_count("纯中文"), 0)
        self.assertEqual(metrics.token_count("café"), 1)
        self.assertEqual(metrics.token_count(""), 0)

    def test_metrics_module_closes_current_engine_imports(self) -> None:
        for module in (
            "bench.engine.frozen_ladder",
            "bench.engine.manifest",
            "bench.engine.preflight",
        ):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(module))

    def test_fidelity_scoring_obeys_the_frozen_runtime_profile(self) -> None:
        canonical_runtime = (
            sys.version_info[:2] == (3, 13)
            and unicodedata.unidata_version == "15.1.0"
        )
        if canonical_runtime:
            self.assertEqual(metrics.distortion("same", "same", "exact"), 0.0)
            self.assertEqual(metrics.distortion("same", "different", "exact"), 1.0)
        else:
            with self.assertRaisesRegex(RuntimeError, "scoring requires"):
                metrics.distortion("same", "same", "exact")

    def test_monotone_envelope_excludes_leakage_and_non_improvements(self) -> None:
        points = [
            {"tokens": 1, "distortion": 0.0, "fidelity": 1.0, "disqualified": True},
            {"tokens": 2, "distortion": 0.8, "fidelity": 0.2},
            {"tokens": 2, "distortion": 0.4, "fidelity": 0.6},
            {"tokens": 4, "distortion": 0.5, "fidelity": 0.5},
            {"tokens": 6, "distortion": 0.1, "fidelity": 0.9},
            {"tokens": 8, "distortion": 0.1, "fidelity": 0.9},
        ]
        frontier = metrics.monotone_lower_envelope(points)
        self.assertEqual([point["tokens"] for point in frontier], [2, 6])
        self.assertEqual([point["frontierRank"] for point in frontier], [1, 2])
        self.assertNotIn("frontierRank", points[2])

    def test_aurc_uses_the_staircase_and_empty_baseline(self) -> None:
        frontier = [
            {"tokens": 2, "distortion": 0.5},
            {"tokens": 5, "distortion": 0.2},
        ]
        self.assertEqual(metrics.aurc(frontier, normalizer_tokens=10), 0.45)
        self.assertEqual(metrics.aurc([], normalizer_tokens=10), 1.0)

    def test_ecl_and_elicit_use_length_coordinates(self) -> None:
        frontier = [
            {"tokens": 4, "fidelity": 0.8},
            {"tokens": 9, "fidelity": 0.96},
            {"tokens": 12, "fidelity": 0.99},
        ]
        self.assertEqual(metrics.ecl_at_tau(frontier, tau=0.95), 9)
        self.assertIsNone(metrics.ecl_at_tau(frontier, tau=1.0))
        self.assertTrue(metrics.elicit_at_k(frontier, tau=0.95, k=9))
        self.assertFalse(metrics.elicit_at_k(frontier, tau=0.95, k=8))

    def test_bootstrap_summary_is_deterministic(self) -> None:
        first = metrics.ci95([0.1, 0.2, 0.9], seed=7, samples=100)
        second = metrics.ci95([0.1, 0.2, 0.9], seed=7, samples=100)
        self.assertEqual(first, second)
        self.assertEqual(metrics.ci95([], seed=7, samples=100), None)
        self.assertEqual(
            metrics.ci95([0.1234567], seed=7, samples=100),
            {"low": 0.123457, "high": 0.123457},
        )
        self.assertEqual(metrics.summarize([], seed=7, samples=100), (None, None))

    def test_ranking_has_a_deterministic_model_tiebreaker(self) -> None:
        rows = [
            {"model": "z-model", "aurc": 0.3},
            {"model": "b-model", "aurc": 0.2},
            {"model": "a-model", "aurc": 0.2},
        ]
        self.assertEqual(
            metrics.rank_by_metric(rows, "aurc"),
            ["a-model", "b-model", "z-model"],
        )


class ProtocolDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = PROTOCOL_DOCUMENT.read_text(encoding="utf-8")

    def test_claim_boundaries_are_explicit(self) -> None:
        for phrase in (
            "not yet the source-authority cutover",
            "ASCII word-like-span proxy",
            "success-conditioned",
            "There is currently no formal v0.2 public model score",
            "Full runner and installed CLI | not yet available",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.text)

    def test_local_links_are_relative_and_resolve(self) -> None:
        root = REPOSITORY_ROOT.resolve()
        for target in MARKDOWN_LINK.findall(self.text):
            if target.startswith(("https://", "http://")):
                continue
            with self.subTest(target=target):
                self.assertFalse(target.startswith("/"))
                path_text = target.split("#", 1)[0]
                resolved = (PROTOCOL_DOCUMENT.parent / path_text).resolve()
                self.assertTrue(resolved.is_relative_to(root))
                self.assertTrue(resolved.is_file())

    def test_no_source_only_or_maintainer_local_paths_remain(self) -> None:
        self.assertNotIn("docs/benchmark/", self.text)
        self.assertNotIn("/Users/", self.text)
        self.assertNotIn("./aleph-bench run", self.text)
        self.assertNotIn("./aleph-bench package", self.text)


if __name__ == "__main__":
    unittest.main()
