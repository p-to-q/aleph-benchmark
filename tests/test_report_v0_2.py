from __future__ import annotations

import copy
import json
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest import mock

from bench.engine import report
from bench.engine.frozen_ladder import run_benchmark
from bench.engine.schema_validation import SchemaValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V2_DATA_DIR = REPOSITORY_ROOT / "bench/data/v0.2/public/s2"
FROZEN_SCORING_RUNTIME = (
    sys.version_info[:2] == (3, 13) and unicodedata.unidata_version == "15.1.0"
)


class ReportProtocolBoundaryTests(unittest.TestCase):
    def write_json(self, root: Path, value: object) -> Path:
        path = root / "result.json"
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def test_only_versioned_v0_2_schema_is_selectable(self) -> None:
        self.assertEqual(
            report.RESULT_SCHEMA_PATH,
            REPOSITORY_ROOT / "schemas/v0.2/aleph-bench-result.schema.json",
        )
        self.assertTrue(report.RESULT_SCHEMA_PATH.is_file())
        self.assertFalse(
            (REPOSITORY_ROOT / "schemas/aleph-bench-result.schema.json").exists()
        )

    def test_loader_rejects_non_object_before_schema_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_json(Path(tmp), [])
            with mock.patch.object(report, "load_schema") as load_schema:
                with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                    report.load_result(path)
            load_schema.assert_not_called()

    def test_loader_rejects_missing_legacy_and_future_protocols_fail_closed(self) -> None:
        cases = ({}, {"protocolVersion": "0.1-legacy"}, {"protocolVersion": "0.3.0"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, value in enumerate(cases):
                with self.subTest(value=value):
                    path = root / f"result-{index}.json"
                    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                    with mock.patch.object(report, "load_schema") as load_schema:
                        with self.assertRaisesRegex(
                            ValueError,
                            "unsupported Aleph-Bench result protocolVersion",
                        ):
                            report.load_result(path)
                    load_schema.assert_not_called()

    def test_direct_renderer_rejects_missing_legacy_and_future_protocols(self) -> None:
        cases = ({}, {"protocolVersion": "0.1-legacy"}, {"protocolVersion": "0.3.0"})
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "unsupported Aleph-Bench result protocolVersion",
                ):
                    report.render_markdown_report(value)

    def test_loader_and_direct_renderer_require_the_v0_2_schema(self) -> None:
        invalid = {"protocolVersion": "0.2.0"}
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_json(Path(tmp), invalid)
            with self.assertRaises(SchemaValidationError):
                report.load_result(path)
        with self.assertRaises(SchemaValidationError):
            report.render_markdown_report(invalid)

    def test_numeric_and_confidence_interval_formatting_is_stable(self) -> None:
        self.assertEqual(report._ci(None), "n/a")
        self.assertEqual(report._ci({"low": 0.0, "high": 1.0}), "0.0-1.0")
        self.assertEqual(report._num(None), "n/a")
        self.assertEqual(report._num(0), "0")
        self.assertEqual(report._num(0.0), "0")

    def test_hosted_provenance_preserves_capture_identity(self) -> None:
        result = {
            "models": [
                {
                    "model": "provider/model@revision",
                    "evidenceMode": "black_box",
                    "adapterIdentity": {
                        "deploymentId": "deployment-7",
                        "endpointSha256": "a" * 64,
                    },
                    "responseCapture": {
                        "capturedAtMin": "2026-09-24T00:00:00Z",
                        "capturedAtMax": "2026-09-24T00:01:00Z",
                        "providerResponseCount": 899,
                        "cacheHitCount": 1,
                    },
                }
            ]
        }
        rendered = report.hosted_provenance_table(result)
        for value in (
            "provider/model@revision",
            "deployment-7",
            "a" * 64,
            "2026-09-24T00:00:00Z — 2026-09-24T00:01:00Z",
            "899 / 1",
        ):
            with self.subTest(value=value):
                self.assertIn(value, rendered)


@unittest.skipUnless(
    FROZEN_SCORING_RUNTIME,
    "scored smoke generation requires Python 3.13 and Unicode 15.1.0",
)
class ScoredSmokeReportRenderingTests(unittest.TestCase):
    def test_mock_result_loads_and_renders_deterministically_without_mutation(self) -> None:
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("report test attempted network access"),
        ):
            result = run_benchmark(
                data_dir=V2_DATA_DIR,
                models=["mock-frontier"],
                seed=7,
                limit=1,
            )
        original = copy.deepcopy(result)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            loaded = report.load_result(path)
            first = report.render_markdown_report(loaded)
            second = report.render_report_file(path)

        self.assertEqual(first, second)
        self.assertEqual(result, original)
        self.assertEqual(loaded, original)
        self.assertIn("- Protocol: `0.2.0`", first)
        self.assertIn("- Evaluation scope: `smoke`", first)
        self.assertIn("- Evaluated items: `1 / 30`", first)
        self.assertIn("## Model Summary", first)
        self.assertIn("## Per-Item Receipt", first)
        self.assertIn(
            "not a canonical benchmark result or a leaderboard score",
            first,
        )
        self.assertIn(
            "pipeline evidence, not a model leaderboard",
            first,
        )
        self.assertIn(
            "it does not add evidence beyond that file",
            first,
        )
        self.assertNotIn("## Hosted Evidence Provenance", first)

        hosted = copy.deepcopy(result)
        hosted_model_id = "provider/model@revision"
        hosted["config"]["evidenceModes"] = ["black_box"]
        hosted["models"][0].update(
            {
                "model": hosted_model_id,
                "evidenceMode": "black_box",
                "adapterIdentity": {
                    "adapterId": "aleph.hosted-openai-compatible",
                    "adapterVersion": "1",
                    "model": hosted_model_id,
                    "observationMode": "black_box",
                    "temperature": 0.0,
                    "wireProtocol": "openai-chat-completions",
                    "requestPayloadVersion": "1",
                    "endpointSha256": "a" * 64,
                    "maxTokens": 512,
                    "providerModel": hosted_model_id,
                    "deploymentId": "deployment-7",
                    "timeoutSeconds": 60,
                    "maxRetries": 0,
                    "retryDelaySeconds": 0.0,
                },
                "responseCapture": {
                    "responseCount": 30,
                    "providerResponseCount": 30,
                    "cacheHitCount": 0,
                    "capturedAtMin": "2026-09-24T00:00:00Z",
                    "capturedAtMax": "2026-09-24T00:01:00Z",
                },
            }
        )
        hosted["itemRuns"][0]["model"] = hosted_model_id
        hosted_report = report.render_markdown_report(hosted)
        self.assertIn("## Hosted Evidence Provenance", hosted_report)
        self.assertIn("- Evaluation scope: `smoke`", hosted_report)
        self.assertIn("- Evaluated items: `1 / 30`", hosted_report)
        self.assertIn(
            "not a canonical benchmark result or a leaderboard score",
            hosted_report,
        )
        self.assertNotIn("pipeline evidence, not a model leaderboard", hosted_report)


if __name__ == "__main__":
    unittest.main()
