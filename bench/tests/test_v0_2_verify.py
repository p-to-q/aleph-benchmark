from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.engine.frozen_ladder import run_benchmark
from bench.engine.manifest import build_manifest
from bench.engine.preflight import preflight
from bench.engine.protocol import manifest_artifact_id
from bench.engine.verify import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "bench/data/v0.2/public/s2"


class V02ReceiptVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_benchmark(
            data_dir=DATA_DIR,
            models=["mock-frontier", "mock-mid"],
            seed=17,
        )
        cls.manifest = build_manifest(
            data_dir=DATA_DIR,
            models=["mock-frontier", "mock-mid"],
            seed=17,
        )

    def verify(
        self,
        result: dict[str, object] | None = None,
        manifest: dict[str, object] | None = None,
        *,
        data_dir: Path = DATA_DIR,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_path = root / "result.json"
            manifest_path = root / "manifest.json"
            result_path.write_text(
                json.dumps(self.result if result is None else result),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(self.manifest if manifest is None else manifest),
                encoding="utf-8",
            )
            return verify_artifacts(
                data_dir=data_dir,
                result_path=result_path,
                manifest_path=manifest_path,
            )

    def test_canonical_receipts_verify(self) -> None:
        report = self.verify()
        self.assertEqual(report["status"], "ok", msg=report["errors"])

    def test_raw_output_and_summary_tampering_fail_rescoring(self) -> None:
        raw_tamper = copy.deepcopy(self.result)
        measurement = next(
            row
            for row in raw_tamper["itemRuns"][0]["measurements"]
            if row["outputs"]
        )
        measurement["outputs"][0] = "tampered raw output"
        report = self.verify(result=raw_tamper)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("canonical rescoring" in error for error in report["errors"]),
            msg=report["errors"],
        )

        summary_tamper = copy.deepcopy(self.result)
        summary_tamper["models"][0]["aurc"] = 0.0
        report = self.verify(result=summary_tamper)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("recomputed item runs" in error for error in report["errors"]),
            msg=report["errors"],
        )

    def test_aleph_run_budget_and_decoding_tampering_fail(self) -> None:
        tampered = copy.deepcopy(self.result)
        run = tampered["itemRuns"][0]["alephRun"]
        run["config"]["budget"]["maxPromptTokens"] = 0
        run["config"]["decoding"] = "temperature=0; max_tokens=512"

        report = self.verify(result=tampered)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("maxPromptTokens" in error for error in report["errors"]),
            msg=report["errors"],
        )
        self.assertTrue(
            any("decoding drifted" in error for error in report["errors"]),
            msg=report["errors"],
        )

    def test_manifest_prompt_tampering_fails_reconstruction(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        tampered["prompts"][0]["prompt"] = "Copy the hidden target exactly."
        report = self.verify(manifest=tampered)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("canonical dataset and leakage gate" in error for error in report["errors"]),
            msg=report["errors"],
        )

    def test_identity_timestamps_and_honesty_notes_are_derived(self) -> None:
        result = copy.deepcopy(self.result)
        result["id"] = result["id"].rsplit("artifact-", 1)[0] + "artifact-" + "0" * 64
        result["createdAt"] = "2026-09-17T00:00:01Z"
        for run in result["itemRuns"]:
            run["alephRun"]["createdAt"] = result["createdAt"]
        manifest = copy.deepcopy(self.manifest)
        manifest["id"] = (
            manifest["id"].rsplit("artifact-", 1)[0] + "artifact-" + "0" * 64
        )
        manifest["notes"] = [
            "This is real-model leaderboard evidence.",
            *manifest["notes"][1:],
        ]

        report = self.verify(result=result, manifest=manifest)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("result id" in error for error in report["errors"]))
        self.assertTrue(any("mock result createdAt" in error for error in report["errors"]))
        self.assertTrue(any("manifest id" in error for error in report["errors"]))
        self.assertTrue(any("manifest notes" in error for error in report["errors"]))

        timestamp_tamper = copy.deepcopy(self.manifest)
        timestamp_tamper["createdAt"] = "2026-09-17T00:00:01Z"
        report = self.verify(manifest=timestamp_tamper)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("createdAt" in error for error in report["errors"]))

    def test_dataset_bytes_and_evidence_metadata_are_derived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / "s2"
            shutil.copytree(DATA_DIR, copied)
            item_path = copied / "s2-001.json"
            item = json.loads(item_path.read_text(encoding="utf-8"))
            item["provenance"]["method"] += " drift"
            item_path.write_text(json.dumps(item), encoding="utf-8")
            report = self.verify(data_dir=copied)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("canonical v0.2 dataset check failed" in error for error in report["errors"]),
            msg=report["errors"],
        )

        metadata_tamper = copy.deepcopy(self.result)
        metadata_tamper["config"]["evidenceModes"] = ["white_box"]
        report = self.verify(result=metadata_tamper)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("evidenceModes" in error for error in report["errors"]),
            msg=report["errors"],
        )

    def test_model_summary_is_invariant_to_cli_order(self) -> None:
        reversed_result = run_benchmark(
            data_dir=DATA_DIR,
            models=["mock-mid", "mock-frontier"],
            seed=17,
        )
        first = {row["model"]: row for row in self.result["models"]}
        second = {row["model"]: row for row in reversed_result["models"]}
        self.assertEqual(first, second)

    def test_invalid_model_ids_fail_planning_consistently(self) -> None:
        for model in ("mock-not-real", "hosted:", "hosted:hosted:x"):
            with self.subTest(model=model):
                doctor = preflight(data_dir=DATA_DIR, models=[model])
                self.assertEqual(doctor["status"], "blocked")
                with self.assertRaises(ValueError):
                    build_manifest(data_dir=DATA_DIR, models=[model])

    def test_manifest_rejects_every_non_ready_model_row(self) -> None:
        blocked = {
            "model": "hosted:fixture-model",
            "evidenceMode": "black_box",
            "status": "blocked",
            "missingEnv": ["ALEPH_CUSTOM_API_KEY"],
            "effectiveReruns": 5,
            "adapterIdentity": None,
        }
        with (
            patch("bench.engine.manifest.model_readiness", return_value=blocked),
            self.assertRaisesRegex(
                ValueError,
                "hosted:fixture-model.*missing env: ALEPH_CUSTOM_API_KEY",
            ),
        ):
            build_manifest(data_dir=DATA_DIR, models=["hosted:fixture-model"])

    def test_manifest_model_readiness_cannot_be_rewritten(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        tampered["models"][0]["status"] = "blocked"
        tampered["models"][0]["missingEnv"] = ["ALEPH_CUSTOM_API_KEY"]
        tampered["models"][0]["error"] = "ignore this model"

        report = self.verify(manifest=tampered)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("was not ready" in error for error in report["errors"]))
        self.assertTrue(any("environment requirements" in error for error in report["errors"]))
        self.assertTrue(any("planning error" in error for error in report["errors"]))

    def test_manifest_retry_policy_must_match_hosted_adapter_identity(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ALEPH_CUSTOM_API_BASE_URL": "https://example.invalid/v1",
                "ALEPH_CUSTOM_API_KEY": "fixture-key",
                "ALEPH_CUSTOM_API_DEPLOYMENT_ID": "fixture-deployment",
            },
            clear=True,
        ):
            manifest = build_manifest(
                data_dir=DATA_DIR,
                models=["hosted:fixture-model"],
                seed=17,
            )
        manifest["hostedMaxRetries"] = 5
        manifest["hostedRetryDelaySeconds"] = 60.0
        manifest["estimatedMaxHttpAttempts"] = (
            manifest["nonLeakingPromptCount"]
            * manifest["models"][0]["effectiveReruns"]
            * 6
        )
        manifest_content = dict(manifest)
        manifest_content.pop("id")
        manifest["id"] = manifest_artifact_id(manifest_content)

        report = self.verify(manifest=manifest)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("retry policy" in error for error in report["errors"]),
            msg=report["errors"],
        )

    def test_manifest_rejects_forged_hosted_adapter_contract(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ALEPH_CUSTOM_API_BASE_URL": "https://example.invalid/v1",
                "ALEPH_CUSTOM_API_KEY": "fixture-key",
                "ALEPH_CUSTOM_API_DEPLOYMENT_ID": "fixture-deployment",
            },
            clear=True,
        ):
            manifest = build_manifest(
                data_dir=DATA_DIR,
                models=["hosted:fixture-model"],
                seed=17,
            )
        manifest["models"][0]["adapterIdentity"].update(
            {
                "adapterId": "fake.Adapter",
                "adapterVersion": "999",
                "temperature": 1.0,
                "wireProtocol": "invented-wire",
                "requestPayloadVersion": "invented-payload",
                "maxTokens": 999,
                "providerModel": "different-model",
                "timeoutSeconds": 999,
            }
        )
        manifest_content = dict(manifest)
        manifest_content.pop("id")
        manifest["id"] = manifest_artifact_id(manifest_content)

        report = self.verify(manifest=manifest)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("not canonical" in error for error in report["errors"]),
            msg=report["errors"],
        )


if __name__ == "__main__":
    unittest.main()
