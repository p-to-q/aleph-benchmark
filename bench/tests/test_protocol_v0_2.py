from __future__ import annotations

from contextlib import ExitStack
import ctypes
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.engine import frozen_ladder as frozen_ladder_module
from bench.engine import metrics as metrics_module
from bench.engine import platform_package_v0_2 as platform_package_module
from bench.engine import scoring_core as scoring_core_module
from bench.engine.adapters.base import ModelAdapter
from bench.engine.adapters.hosted_black_box import (
    HostedBlackBoxAdapter,
    HostedBlackBoxError,
)
from bench.engine.frozen_ladder import evaluate_item, load_items, run_benchmark
from bench.engine.manifest import build_manifest
from bench.engine.platform_package_v0_2 import (
    PACKAGE_TREE_HASH_ALGORITHM,
    UnsupportedPlatformError,
    check_v0_2_package,
    package_tree_sha256,
    write_v0_2_package,
)
from bench.engine.protocol import (
    FROZEN_DATASET_HASH_ALGORITHM,
    FROZEN_DATASET_ID,
    FROZEN_DATASET_ITEM_COUNT,
    FROZEN_DATASET_SHA256,
    PROTOCOL_VERSION,
    RESPONSE_CAPTURE_VERSION,
    load_protocol_config,
)
from bench.engine.response_cache import MAX_CACHE_RECORD_BYTES, ResponseCacheAdapter
from bench.engine.preflight import preflight
from bench.engine.schema_validation import SchemaValidationError, load_schema, validate
from bench.engine.scoring_core import MAX_SCORING_TEXT_CHARACTERS


ROOT = Path(__file__).resolve().parents[2]
V2_DATA_DIR = ROOT / "bench/data/v0.2/public/s2"
V2_SCHEMA_DIR = ROOT / "schemas/v0.2"
RAW_OUTPUTS = [
    "  Cafe\u0301\r\n",
    "👩\u200d💻\n",
    "尾随空白  ",
    "fourth raw output",
    "fifth raw output",
]
CANONICAL_SCORING_RUNTIME = (
    sys.version_info[:2] == (3, 13) and unicodedata.unidata_version == "15.1.0"
)
EXPECTED_PACKAGE_TREE_SHA256 = (
    "be213ca07782f108815bf94264b4d87f48570c29cbe8511f6346e31296bf0cd8"
)


class RawSequenceAdapter(ModelAdapter):
    def __init__(self) -> None:
        super().__init__(
            model_id="raw-sequence",
            observation_mode="fixture",
            temperature=0.7,
        )

    def generate(
        self,
        prompt: str,
        item: dict[str, object],
        ladder_prompt: dict[str, object],
        *,
        seed: int,
        rerun_index: int,
    ) -> str:
        del prompt, item, ladder_prompt, seed
        return RAW_OUTPUTS[rerun_index]


class LiteralAdapter(ModelAdapter):
    def __init__(self, output: object) -> None:
        super().__init__(model_id="literal", observation_mode="fixture")
        self.output = output

    def generate(
        self,
        prompt: str,
        item: dict[str, object],
        ladder_prompt: dict[str, object],
        *,
        seed: int,
        rerun_index: int,
    ) -> str:
        del prompt, item, ladder_prompt, seed, rerun_index
        return self.output  # type: ignore[return-value]


class ProtocolV02Tests(unittest.TestCase):
    def test_huge_json_integer_validation_does_not_overflow(self) -> None:
        huge = 10**1000
        validate(huge, {"type": "integer"})
        with self.assertRaises(SchemaValidationError):
            validate(huge, {"type": "number", "maximum": 1})

        config = json.loads(
            (ROOT / "bench/config/frozen_ladder-v0.2.json").read_text(encoding="utf-8")
        )
        config["tau"] = huge
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tau must be between"):
                load_protocol_config(path)

    def test_frozen_config_drift_fails_before_adapter_construction(self) -> None:
        config_path = ROOT / "bench/config/frozen_ladder-v0.2.json"
        canonical = json.loads(config_path.read_text(encoding="utf-8"))
        mutations = {
            "protocolVersion": "0.2.1",
            "track": "O",
            "split": "fresh",
            "stratum": "S3",
            "datasetId": "different-dataset",
            "datasetItemCount": 31,
            "datasetSha256": "0" * 64,
            "datasetHashAlgorithm": "different-algorithm",
            "tau": 0.9,
            "k": 17,
            "reruns": 6,
            "bootstrapSamples": 501,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for field, value in mutations.items():
                with self.subTest(field=field):
                    drifted = json.loads(json.dumps(canonical))
                    drifted[field] = value
                    path = root / f"{field}.json"
                    path.write_text(
                        json.dumps(drifted, indent=2) + "\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "config bytes changed"):
                        load_protocol_config(path)

            for field, value in {
                "lcsRatio": 0.6,
                "targetTrigramRecall": 0.4,
                "verbatimSpanUnits": 15,
                "skeletonLcsRatio": 0.7,
                "skeletonTargetTrigramRecall": 0.7,
                "skeletonVerbatimSpanUnits": 31,
            }.items():
                with self.subTest(threshold=field):
                    drifted = json.loads(json.dumps(canonical))
                    drifted["leakageThresholds"][field] = value
                    path = root / f"threshold-{field}.json"
                    path.write_text(
                        json.dumps(drifted, indent=2) + "\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "config bytes changed"):
                        load_protocol_config(path)

            drifted = json.loads(json.dumps(canonical))
            drifted["decoding"]["hosted"] += "; drifted=true"
            path = root / "runtime-drift.json"
            path.write_text(json.dumps(drifted, indent=2) + "\n", encoding="utf-8")
            with patch("bench.engine.protocol.PROTOCOL_CONFIG_PATH", path):
                doctor = preflight(data_dir=V2_DATA_DIR, models=["mock-frontier"])
                self.assertEqual(doctor["status"], "blocked")
                self.assertTrue(
                    any("config bytes changed" in error for error in doctor["errors"])
                )
                with self.assertRaisesRegex(ValueError, "config bytes changed"):
                    build_manifest(
                        data_dir=V2_DATA_DIR,
                        models=["mock-frontier"],
                    )
                with patch("bench.engine.frozen_ladder.adapter_for_model") as adapter:
                    with self.assertRaisesRegex(ValueError, "config bytes changed"):
                        run_benchmark(
                            data_dir=V2_DATA_DIR,
                            models=["mock-frontier"],
                        )
                    adapter.assert_not_called()

            drifted = json.loads(json.dumps(canonical))
            drifted["unexpected"] = True
            path = root / "extra-key.json"
            path.write_text(json.dumps(drifted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"unexpected=\['unexpected'\]"):
                load_protocol_config(path)

    def test_v0_2_result_and_items_validate_while_legacy_is_rejected(self) -> None:
        v2_item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        v2_schema = load_schema(V2_SCHEMA_DIR / "aleph-bench-item.schema.json")
        validate(v2_item, v2_schema)
        rejected_items = {
            "legacy": {**v2_item, "protocolVersion": "0.1.0"},
            "missing-version": {
                key: value for key, value in v2_item.items() if key != "protocolVersion"
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            item_path = Path(tmp) / "s2-001.json"
            for name, rejected_item in rejected_items.items():
                with self.subTest(name=name):
                    with self.assertRaises(SchemaValidationError):
                        validate(rejected_item, v2_schema)
                    item_path.write_text(json.dumps(rejected_item), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "rejected dataset item"):
                        load_items(item_path.parent, limit=1)

        result = run_benchmark(
            data_dir=V2_DATA_DIR,
            models=["mock-frontier"],
            limit=2,
        )
        self.assertEqual(result["protocolVersion"], PROTOCOL_VERSION)
        result_schema = load_schema(V2_SCHEMA_DIR / "aleph-bench-result.schema.json")
        validate(result, result_schema)
        self.assertEqual(result["evaluationScope"], "smoke")
        self.assertEqual(result["requestedItemLimit"], 2)
        self.assertEqual(result["evaluatedItemCount"], 2)
        self.assertIn("-smoke-2-", result["id"])

        mislabeled = json.loads(json.dumps(result))
        mislabeled["evaluationScope"] = "canonical"
        with self.assertRaises(SchemaValidationError):
            validate(mislabeled, result_schema)
        with self.assertRaisesRegex(ValueError, "exceeds the available dataset size"):
            run_benchmark(
                data_dir=V2_DATA_DIR,
                models=["mock-frontier"],
                limit=FROZEN_DATASET_ITEM_COUNT + 1,
            )

    def test_artifact_ids_cover_content_and_model_order_is_canonical(self) -> None:
        frontier = run_benchmark(
            data_dir=V2_DATA_DIR,
            models=["mock-frontier"],
            seed=0,
            limit=1,
        )
        small = run_benchmark(
            data_dir=V2_DATA_DIR,
            models=["mock-small"],
            seed=0,
            limit=1,
        )
        reseeded = run_benchmark(
            data_dir=V2_DATA_DIR,
            models=["mock-frontier"],
            seed=1,
            limit=1,
        )
        self.assertNotEqual(frontier["id"], small["id"])
        self.assertNotEqual(frontier["id"], reseeded["id"])
        self.assertNotEqual(
            frontier["itemRuns"][0]["alephRun"]["id"],
            reseeded["itemRuns"][0]["alephRun"]["id"],
        )

        ordered = build_manifest(
            data_dir=V2_DATA_DIR,
            models=["mock-frontier", "mock-small"],
        )
        reversed_order = build_manifest(
            data_dir=V2_DATA_DIR,
            models=["mock-small", "mock-frontier"],
        )
        self.assertEqual(ordered, reversed_order)

        with patch.dict(
            os.environ,
            {
                "ALEPH_CUSTOM_API_BASE_URL": "https://example.invalid/v1",
                "ALEPH_CUSTOM_API_KEY": "fixture-key",
                "ALEPH_CUSTOM_API_DEPLOYMENT_ID": "fixture-deployment",
                "ALEPH_CUSTOM_API_MAX_RETRIES": "0",
            },
            clear=True,
        ):
            no_retry = build_manifest(
                data_dir=V2_DATA_DIR,
                models=["hosted:fixture-model"],
            )
        with patch.dict(
            os.environ,
            {
                "ALEPH_CUSTOM_API_BASE_URL": "https://example.invalid/v1",
                "ALEPH_CUSTOM_API_KEY": "fixture-key",
                "ALEPH_CUSTOM_API_DEPLOYMENT_ID": "fixture-deployment",
                "ALEPH_CUSTOM_API_MAX_RETRIES": "5",
            },
            clear=True,
        ):
            five_retries = build_manifest(
                data_dir=V2_DATA_DIR,
                models=["hosted:fixture-model"],
            )
        self.assertNotEqual(no_retry["id"], five_retries["id"])

    def test_loader_rejects_mixed_versions_and_unknown_metrics_before_limit(self) -> None:
        valid = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "s2-001.json").write_text(
                json.dumps(valid), encoding="utf-8"
            )
            mixed = dict(valid)
            mixed["id"] = "s2-002"
            mixed.pop("protocolVersion")
            (data_dir / "s2-002.json").write_text(
                json.dumps(mixed), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "rejected dataset item"):
                load_items(data_dir, limit=1)

            mixed["protocolVersion"] = PROTOCOL_VERSION
            mixed["metricClass"] = "semantic"
            (data_dir / "s2-002.json").write_text(
                json.dumps(mixed), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unsupported metric class"):
                load_items(data_dir, limit=1)

            mixed["metricClass"] = "normalized_edit_similarity"
            mixed["target"] = {"text": "\u200d", "label": "empty after normalization"}
            (data_dir / "s2-002.json").write_text(
                json.dumps(mixed), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "empty after leakage normalization"):
                load_items(data_dir, limit=1)

    def test_hosted_adapter_preserves_raw_string_and_rejects_non_string(self) -> None:
        adapter = HostedBlackBoxAdapter(
            "fixture-model",
            base_url="https://invalid.example",
            api_key="fixture-key",
            deployment_id="fixture-deployment",
            max_retries=0,
        )
        raw = " \tCafe\u0301\r\n👩\u200d💻  "
        with patch.object(
            adapter,
            "_post_chat_completion",
            return_value={"choices": [{"message": {"content": raw}}]},
        ):
            observed = adapter.generate(
                "prompt",
                {"target": {"text": "target"}},
                {"id": "prompt-1"},
                seed=0,
                rerun_index=0,
            )
        self.assertEqual(observed, raw)

        with patch.object(
            adapter,
            "_post_chat_completion",
            return_value={"choices": [{"message": {"content": None}}]},
        ):
            with self.assertRaisesRegex(HostedBlackBoxError, "must be a string"):
                adapter.generate(
                    "prompt",
                    {"target": {"text": "target"}},
                    {"id": "prompt-1"},
                    seed=0,
                    rerun_index=0,
                )

        with patch.object(
            adapter,
            "_post_chat_completion",
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "x" * (MAX_SCORING_TEXT_CHARACTERS + 1)
                        }
                    }
                ]
            },
        ):
            with self.assertRaisesRegex(HostedBlackBoxError, "exceeded"):
                adapter.generate(
                    "prompt",
                    {"target": {"text": "target"}},
                    {"id": "prompt-1"},
                    seed=0,
                    rerun_index=0,
                )

    def test_response_cache_round_trips_raw_output_and_versions_identity(self) -> None:
        raw = "  Cafe\u0301\r\n👩\u200d💻  "
        item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        ladder = item["frozenLadder"][2]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            first = ResponseCacheAdapter(LiteralAdapter(raw), cache_dir, seed=7)
            observed = first.generate(
                ladder["prompt"], item, ladder, seed=7, rerun_index=0
            )
            self.assertEqual(observed, raw)
            cache_files = list(cache_dir.rglob("*.json"))
            self.assertEqual(len(cache_files), 1)
            record = json.loads(cache_files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["output"], raw)
            self.assertEqual(record["protocolVersion"], PROTOCOL_VERSION)
            self.assertEqual(record["responseCaptureVersion"], RESPONSE_CAPTURE_VERSION)

            cached = ResponseCacheAdapter(LiteralAdapter("different"), cache_dir, seed=7)
            self.assertEqual(
                cached.generate(ladder["prompt"], item, ladder, seed=7, rerun_index=0),
                raw,
            )
            self.assertEqual(cached.cache_hits, 1)

    def test_response_cache_does_not_cross_hosted_transport_policies(self) -> None:
        item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        ladder = item["frozenLadder"][2]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            first_hosted = HostedBlackBoxAdapter(
                "fixture-model",
                base_url="https://example.invalid/v1",
                api_key="fixture-key",
                deployment_id="fixture-deployment",
                max_retries=2,
                retry_delay_seconds=1,
            )
            first = ResponseCacheAdapter(first_hosted, cache_dir, seed=7)
            with patch.object(
                first_hosted,
                "_post_chat_completion",
                return_value={"choices": [{"message": {"content": "first"}}]},
            ):
                self.assertEqual(
                    first.generate(
                        ladder["prompt"], item, ladder, seed=7, rerun_index=0
                    ),
                    "first",
                )

            changed_hosted = HostedBlackBoxAdapter(
                "fixture-model",
                base_url="https://example.invalid/v1",
                api_key="fixture-key",
                deployment_id="fixture-deployment",
                max_retries=5,
                retry_delay_seconds=1,
            )
            changed = ResponseCacheAdapter(changed_hosted, cache_dir, seed=7)
            with patch.object(
                changed_hosted,
                "_post_chat_completion",
                return_value={"choices": [{"message": {"content": "second"}}]},
            ) as post:
                self.assertEqual(
                    changed.generate(
                        ladder["prompt"], item, ladder, seed=7, rerun_index=0
                    ),
                    "second",
                )
                post.assert_called_once()
            self.assertEqual(changed.cache_hits, 0)
            self.assertEqual(changed.cache_misses, 1)

    def test_response_cache_rejects_symlink_entries_without_overwriting_target(self) -> None:
        item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        ladder = item["frozenLadder"][2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = ResponseCacheAdapter(LiteralAdapter("new output"), root / "cache", seed=7)
            identity = cache._request_identity(
                ladder["prompt"], item, ladder, seed=7, rerun_index=0
            )
            entry = cache._path_for_key(cache._cache_key(identity))
            entry.parent.mkdir(parents=True)
            cache.cache_dir.chmod(0o700)
            entry.parent.chmod(0o700)
            victim = root / "victim.txt"
            victim.write_text("preserve\n", encoding="utf-8")
            entry.symlink_to(victim)

            with self.assertRaisesRegex(ValueError, "regular file, not a link"):
                cache.generate(
                    ladder["prompt"], item, ladder, seed=7, rerun_index=0
                )
            self.assertEqual(victim.read_text(encoding="utf-8"), "preserve\n")

    def test_response_cache_uses_private_regular_files(self) -> None:
        item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        ladder = item["frozenLadder"][2]
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache = ResponseCacheAdapter(LiteralAdapter("output"), cache_dir, seed=7)
            cache.generate(ladder["prompt"], item, ladder, seed=7, rerun_index=0)
            self.assertEqual(stat.S_IMODE(cache_dir.stat().st_mode), 0o700)
            for directory in (path for path in cache_dir.rglob("*") if path.is_dir()):
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            for path in (path for path in cache_dir.rglob("*") if path.is_file()):
                self.assertFalse(path.is_symlink())
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_response_cache_rejects_oversized_records_before_json_parse(self) -> None:
        item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        ladder = item["frozenLadder"][2]
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResponseCacheAdapter(
                LiteralAdapter("new output"), Path(tmp) / "cache", seed=7
            )
            identity = cache._request_identity(
                ladder["prompt"], item, ladder, seed=7, rerun_index=0
            )
            entry = cache._path_for_key(cache._cache_key(identity))
            entry.parent.mkdir(mode=0o700, parents=True)
            cache.cache_dir.chmod(0o700)
            entry.parent.chmod(0o700)
            entry.write_bytes(b" " * (MAX_CACHE_RECORD_BYTES + 1))
            entry.chmod(0o600)

            with self.assertRaisesRegex(ValueError, "cache entry exceeds"):
                cache.generate(
                    ladder["prompt"], item, ladder, seed=7, rerun_index=0
                )

    def test_response_cache_refuses_existing_shared_directory_without_chmod(self) -> None:
        item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        ladder = item["frozenLadder"][2]
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared"
            shared.mkdir(mode=0o755)
            shared.chmod(0o755)
            cache = ResponseCacheAdapter(LiteralAdapter("output"), shared, seed=7)

            with self.assertRaisesRegex(ValueError, "must have mode 0700"):
                cache.generate(
                    ladder["prompt"], item, ladder, seed=7, rerun_index=0
                )

            self.assertEqual(stat.S_IMODE(shared.stat().st_mode), 0o755)
            self.assertEqual(list(shared.iterdir()), [])

    def test_response_cache_refuses_symlinked_ancestor(self) -> None:
        item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        ladder = item["frozenLadder"][2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_parent = root / "real"
            real_parent.mkdir(mode=0o700)
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            cache = ResponseCacheAdapter(
                LiteralAdapter("output"), linked_parent / "cache", seed=7
            )

            with self.assertRaisesRegex(ValueError, "path component"):
                cache.generate(
                    ladder["prompt"], item, ladder, seed=7, rerun_index=0
                )

            self.assertFalse((real_parent / "cache").exists())

    def test_measurements_retain_every_raw_rerun(self) -> None:
        item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        run = evaluate_item(item, RawSequenceAdapter(), seed=0)
        gated = [row for row in run["measurements"] if row["disqualified"]]
        scored = [row for row in run["measurements"] if not row["disqualified"]]
        self.assertTrue(all(row["outputs"] == [] for row in gated))
        self.assertTrue(all(row["outputs"] == RAW_OUTPUTS for row in scored))
        self.assertTrue(all(row["rerunCount"] == 5 for row in scored))

    def test_aleph_run_budget_and_decoding_match_candidates(self) -> None:
        result = run_benchmark(
            data_dir=V2_DATA_DIR,
            models=["mock-frontier"],
            seed=23,
        )
        for item_run in result["itemRuns"]:
            aleph_run = item_run["alephRun"]
            tokens = [candidate["tokens"] for candidate in aleph_run["candidates"]]
            self.assertEqual(
                aleph_run["config"]["budget"]["maxPromptTokens"],
                max(tokens),
            )
            self.assertTrue(
                all(
                    token <= aleph_run["config"]["budget"]["maxPromptTokens"]
                    for token in tokens
                )
            )
            self.assertEqual(aleph_run["config"]["decoding"], "temperature=0")

    def test_non_string_adapter_output_fails_before_scoring(self) -> None:
        item = json.loads((V2_DATA_DIR / "s2-001.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(TypeError, "non-string output"):
            evaluate_item(item, LiteralAdapter(7), seed=0)

    def test_v0_2_package_is_deterministic_and_runs_shared_conformance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first_manifest = write_v0_2_package(first)
            second_manifest = write_v0_2_package(second)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["datasetId"], FROZEN_DATASET_ID)
            self.assertEqual(
                first_manifest["datasetItemCount"], FROZEN_DATASET_ITEM_COUNT
            )
            self.assertEqual(
                first_manifest["datasetSha256"], FROZEN_DATASET_SHA256
            )
            self.assertEqual(
                first_manifest["datasetHashAlgorithm"],
                FROZEN_DATASET_HASH_ALGORITHM,
            )
            self.assertEqual(first_manifest["scoringProfile"]["pythonVersion"], "3.13")
            self.assertEqual(
                first_manifest["scoringProfile"]["unicodeDatabaseVersion"],
                "15.1.0",
            )
            validate(
                first_manifest,
                load_schema(V2_SCHEMA_DIR / "aleph-bench-platform-package.schema.json"),
            )
            self.assertEqual(
                (first / "package-manifest.json").read_bytes(),
                (json.dumps(first_manifest, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                ),
            )
            self.assertEqual(
                (first / "kaggle/_scoring.py").read_bytes(),
                (ROOT / "bench/engine/scoring_core.py").read_bytes(),
            )
            first_files = {
                path.relative_to(first).as_posix(): path.read_bytes()
                for path in first.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second).as_posix(): path.read_bytes()
                for path in second.rglob("*")
                if path.is_file()
            }
            self.assertEqual(len(first_files), 10)
            self.assertEqual(first_files, second_files)
            first_tree_sha256 = package_tree_sha256(first)
            self.assertEqual(first_tree_sha256, package_tree_sha256(second))
            self.assertEqual(
                PACKAGE_TREE_HASH_ALGORITHM,
                "sha256-length-framed-path-type-mode-and-content-v1",
            )
            self.assertEqual(first_tree_sha256, EXPECTED_PACKAGE_TREE_SHA256)
            for directory in [first, *(path for path in first.rglob("*") if path.is_dir())]:
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o755)
            for path in (path for path in first.rglob("*") if path.is_file()):
                expected_mode = (
                    0o755
                    if path.relative_to(first).as_posix() == "kaggle/run_conformance.py"
                    else 0o644
                )
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected_mode)
            package_check = check_v0_2_package(first / "package-manifest.json")
            self.assertEqual(package_check["status"], "ok")
            self.assertEqual(package_check["packageTreeSha256"], first_tree_sha256)
            self.assertEqual(
                package_check["expectedPackageTreeSha256"], first_tree_sha256
            )
            files_before_runner = {
                path.relative_to(first).as_posix(): path.read_bytes()
                for path in first.rglob("*")
                if path.is_file()
            }
            completed = subprocess.run(
                [sys.executable, str(first / "kaggle/run_conformance.py")],
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            if CANONICAL_SCORING_RUNTIME:
                self.assertEqual(completed.returncode, 0)
                self.assertEqual(report["status"], "ok")
                self.assertEqual(report["fidelityChecks"], 42)
                self.assertEqual(report["leakageChecks"], 28)
                self.assertEqual(report["unsupportedMetricChecks"], 4)
                self.assertEqual(
                    report["scoringProfile"], first_manifest["scoringProfile"]
                )
            else:
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(report["status"], "runtime_incompatible")
                self.assertEqual(report["requiredPythonVersion"], "3.13")
                self.assertEqual(
                    report["observedPythonVersion"],
                    f"{sys.version_info.major}.{sys.version_info.minor}",
                )
                self.assertEqual(report["requiredUnicodeDatabaseVersion"], "15.1.0")
                self.assertEqual(
                    report["observedUnicodeDatabaseVersion"],
                    unicodedata.unidata_version,
                )
                self.assertEqual(report["fidelityChecks"], 0)
                self.assertEqual(report["leakageChecks"], 0)
                self.assertEqual(report["unsupportedMetricChecks"], 0)
                self.assertTrue(report["errors"])
            files_after_runner = {
                path.relative_to(first).as_posix(): path.read_bytes()
                for path in first.rglob("*")
                if path.is_file()
            }
            self.assertEqual(files_after_runner, files_before_runner)

            (first / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            tampered = check_v0_2_package(first / "package-manifest.json")
            self.assertEqual(tampered["status"], "failed")
            self.assertIn("unexpected package file: unexpected.txt", tampered["errors"])

    def test_v0_2_package_build_and_check_do_not_execute_scoring(self) -> None:
        prohibited_scoring_names = (
            "validate_scoring_runtime",
            "fidelity",
            "evaluate_leakage",
            "normalize_unicode",
            "normalize_lexical_text",
            "_normalize_leakage_text",
            "fail_closed_codepoint_reason",
            "unicode_skeleton_codepoints",
            "unicode_lexical_units",
        )
        prohibited_metric_names = (
            "token_count",
            "distortion",
            "monotone_lower_envelope",
            "aurc",
            "ecl_at_tau",
            "elicit_at_k",
            "ci95",
            "summarize",
            "rank_by_metric",
        )

        def prohibited(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("package assembly executed a scoring operation")

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            for name in prohibited_scoring_names:
                stack.enter_context(
                    patch.object(scoring_core_module, name, side_effect=prohibited)
                )
                if hasattr(platform_package_module, name):
                    stack.enter_context(
                        patch.object(
                            platform_package_module, name, side_effect=prohibited
                        )
                    )
            for name in prohibited_metric_names:
                stack.enter_context(
                    patch.object(metrics_module, name, side_effect=prohibited)
                )
            stack.enter_context(
                patch.object(
                    frozen_ladder_module, "load_items", side_effect=prohibited
                )
            )
            if hasattr(platform_package_module, "load_items"):
                stack.enter_context(
                    patch.object(
                        platform_package_module, "load_items", side_effect=prohibited
                    )
                )

            package = Path(tmp) / "package"
            manifest = write_v0_2_package(package)
            report = check_v0_2_package(package / "package-manifest.json")

        self.assertEqual(manifest["scoringProfile"]["pythonVersion"], "3.13")
        self.assertEqual(
            manifest["scoringProfile"]["unicodeDatabaseVersion"], "15.1.0"
        )
        self.assertEqual(report["status"], "ok")

    @unittest.skipIf(
        CANONICAL_SCORING_RUNTIME,
        "runtime-incompatible behavior requires a non-canonical scorer runtime",
    )
    def test_packaged_conformance_rejects_runtime_before_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            write_v0_2_package(package)
            fixture_path = package / "conformance/scorer-v0.2.json"
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            fixture["fidelityVectors"][0]["expected"]["exact"] = False
            fixture_path.write_text(
                json.dumps(fixture, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            files_before_runner = {
                path.relative_to(package).as_posix(): path.read_bytes()
                for path in package.rglob("*")
                if path.is_file()
            }

            completed = subprocess.run(
                [sys.executable, str(package / "kaggle/run_conformance.py")],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 2)
            report = json.loads(completed.stdout)
            self.assertEqual(report["status"], "runtime_incompatible")
            self.assertEqual(report["requiredPythonVersion"], "3.13")
            self.assertEqual(
                report["observedPythonVersion"],
                f"{sys.version_info.major}.{sys.version_info.minor}",
            )
            self.assertEqual(report["requiredUnicodeDatabaseVersion"], "15.1.0")
            self.assertEqual(
                report["observedUnicodeDatabaseVersion"],
                unicodedata.unidata_version,
            )
            self.assertEqual(report["fidelityChecks"], 0)
            self.assertEqual(report["leakageChecks"], 0)
            self.assertEqual(report["unsupportedMetricChecks"], 0)
            self.assertTrue(report["errors"])
            files_after_runner = {
                path.relative_to(package).as_posix(): path.read_bytes()
                for path in package.rglob("*")
                if path.is_file()
            }
            self.assertEqual(files_after_runner, files_before_runner)

    def test_v0_2_package_requires_canonical_dataset_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copied_data = Path(tmp) / "data"
            shutil.copytree(V2_DATA_DIR, copied_data)
            item_path = copied_data / "s2-001.json"
            item_path.write_bytes(item_path.read_bytes() + b"\n")
            destination = Path(tmp) / "package"
            with patch(
                "bench.engine.platform_package_v0_2.ITEMS_SOURCE_DIR", copied_data
            ):
                with self.assertRaisesRegex(ValueError, "does not match frozen"):
                    write_v0_2_package(destination)
            self.assertFalse(destination.exists())

    def test_v0_2_dataset_digest_and_schema_share_one_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copied_data = Path(tmp) / "data"
            shutil.copytree(V2_DATA_DIR, copied_data)
            item_path = copied_data / "s2-001.json"
            expected_item = json.loads(item_path.read_text(encoding="utf-8"))
            original_digest = platform_package_module._dataset_sha256

            def mutate_after_digest(
                snapshots: list[tuple[str, bytes]],
            ) -> str:
                digest = original_digest(snapshots)
                item_path.write_text("{}\n", encoding="utf-8")
                return digest

            with patch.object(
                platform_package_module, "ITEMS_SOURCE_DIR", copied_data
            ), patch.object(
                platform_package_module,
                "_dataset_sha256",
                side_effect=mutate_after_digest,
            ):
                items = platform_package_module._load_v0_2_items()

            self.assertEqual(len(items), FROZEN_DATASET_ITEM_COUNT)
            self.assertEqual(items[0], expected_item)
            self.assertEqual(item_path.read_text(encoding="utf-8"), "{}\n")

    def test_v0_2_package_rejects_existing_destination_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "package"
            destination.mkdir()
            sentinel = destination / "sentinel.txt"
            sentinel.write_text("preserve\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                write_v0_2_package(destination)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve\n")

            broken_link = Path(tmp) / "broken-link"
            broken_link.symlink_to(Path(tmp) / "missing-target", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                write_v0_2_package(broken_link)
            self.assertTrue(broken_link.is_symlink())

    def test_v0_2_package_requires_safe_parent_and_capabilities_before_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_parent = root / "missing" / "package"
            with patch.object(
                platform_package_module,
                "_atomic_rename_implementation",
                side_effect=UnsupportedPlatformError("injected unsupported host"),
            ) as capability, patch.object(
                platform_package_module, "_create_staging_directory"
            ) as create_staging:
                with self.assertRaisesRegex(
                    UnsupportedPlatformError, "injected unsupported host"
                ):
                    write_v0_2_package(missing_parent)
            capability.assert_called_once_with()
            create_staging.assert_not_called()
            self.assertFalse((root / "missing").exists())

            parent = root / "supported-parent"
            parent.mkdir()

            def unavailable_primitive(*_args: object) -> int:
                ctypes.set_errno(errno.ENOSYS)
                return -1

            with patch.object(
                platform_package_module,
                "_atomic_rename_implementation",
                return_value=(unavailable_primitive, 1),
            ), patch.object(
                platform_package_module, "_create_staging_directory"
            ) as create_staging:
                with self.assertRaisesRegex(
                    UnsupportedPlatformError, "capability probe"
                ):
                    write_v0_2_package(parent / "package")
            create_staging.assert_not_called()
            self.assertEqual(list(parent.iterdir()), [])

            with self.assertRaisesRegex(ValueError, "ancestor is unavailable"):
                write_v0_2_package(missing_parent)
            self.assertFalse((root / "missing").exists())

            file_parent = root / "not-a-directory"
            file_parent.write_text("preserve\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a directory"):
                write_v0_2_package(file_parent / "package")
            self.assertEqual(file_parent.read_text(encoding="utf-8"), "preserve\n")

            unsafe_parent = root / "unsafe-parent"
            unsafe_parent.mkdir()
            unsafe_parent.chmod(0o777)
            with patch.object(
                platform_package_module, "_create_staging_directory"
            ) as create_staging:
                with self.assertRaisesRegex(
                    ValueError, "group/world writable without sticky protection"
                ):
                    write_v0_2_package(unsafe_parent / "package")
            create_staging.assert_not_called()
            self.assertEqual(list(unsafe_parent.iterdir()), [])

            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                write_v0_2_package(linked_parent / "package")
            self.assertFalse((real_parent / "package").exists())

            for destination in (
                "",
                os.sep,
                f"{root}{os.sep}package{os.sep}",
                f"{root}{os.sep}.",
                f"{root}{os.sep}..",
            ):
                with self.subTest(destination=destination):
                    with self.assertRaisesRegex(ValueError, "destination"):
                        write_v0_2_package(destination)  # type: ignore[arg-type]

    def test_v0_2_package_detects_lexical_parent_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            detached = root / "detached-parent"
            parent.mkdir()
            destination = parent / "package"
            original_create = platform_package_module._create_staging_directory

            def create_then_replace_parent(
                parent_fd: int,
            ) -> tuple[str, int, tuple[int, int]]:
                created = original_create(parent_fd)
                parent.rename(detached)
                parent.mkdir()
                return created

            with patch.object(
                platform_package_module,
                "_create_staging_directory",
                side_effect=create_then_replace_parent,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "destination parent identity changed"
                ):
                    write_v0_2_package(destination)

            self.assertFalse(destination.exists())
            self.assertEqual(list(detached.iterdir()), [])
            self.assertEqual(list(parent.iterdir()), [])

    def test_v0_2_package_rejects_unencodable_leaf_without_fd_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid_destination = root / "\ud800"
            with self.assertRaisesRegex(ValueError, "filesystem encoding"):
                write_v0_2_package(invalid_destination)
            self.assertEqual(list(root.iterdir()), [])

            retained_fd = os.open(
                root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            with patch.object(
                platform_package_module,
                "_open_directory_components",
                return_value=retained_fd,
            ):
                with self.assertRaisesRegex(ValueError, "filesystem encoding"):
                    platform_package_module._open_parent_descriptor(
                        f"{os.sep}\ud800"
                    )
            with self.assertRaises(OSError) as closed:
                os.fstat(retained_fd)
            self.assertEqual(closed.exception.errno, errno.EBADF)

            with patch.object(
                platform_package_module.os,
                "fpathconf",
                return_value=-1,
            ):
                parent_fd, leaf = platform_package_module._open_parent_descriptor(
                    str(root / "package")
                )
            try:
                self.assertEqual(leaf, "package")
            finally:
                os.close(parent_fd)

    def test_v0_2_package_destination_race_preserves_competing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "package"

            def competing_publish(
                parent_fd: int,
                source_name: str,
                destination_name: str,
                **_identity: object,
            ) -> None:
                self.assertNotEqual(source_name, destination_name)
                os.mkdir(destination_name, mode=0o700, dir_fd=parent_fd)
                destination_fd = os.open(
                    destination_name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                try:
                    sentinel_fd = os.open(
                        "winner.txt",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        mode=0o600,
                        dir_fd=destination_fd,
                    )
                    try:
                        os.write(sentinel_fd, b"competing writer\n")
                    finally:
                        os.close(sentinel_fd)
                finally:
                    os.close(destination_fd)
                raise FileExistsError("injected competing destination")

            with patch.object(
                platform_package_module,
                "_publish_directory_noreplace",
                side_effect=competing_publish,
            ):
                with self.assertRaisesRegex(
                    FileExistsError, "injected competing destination"
                ):
                    write_v0_2_package(destination)

            self.assertEqual(
                (destination / "winner.txt").read_bytes(), b"competing writer\n"
            )
            self.assertEqual([path.name for path in root.iterdir()], [destination.name])

    def test_v0_2_package_native_noreplace_preserves_both_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "source.txt").write_text("source\n", encoding="utf-8")
            (destination / "winner.txt").write_text("winner\n", encoding="utf-8")
            source_before = source.stat()
            destination_before = destination.stat()
            parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            source_fd = os.open(
                source.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                with self.assertRaises(FileExistsError):
                    platform_package_module._publish_directory_noreplace(
                        parent_fd,
                        source.name,
                        destination.name,
                        source_fd=source_fd,
                        expected_identity=(source_before.st_dev, source_before.st_ino),
                    )
            finally:
                os.close(source_fd)
                os.close(parent_fd)

            source_after = source.stat()
            destination_after = destination.stat()
            self.assertEqual(
                (source_after.st_dev, source_after.st_ino),
                (source_before.st_dev, source_before.st_ino),
            )
            self.assertEqual(
                (destination_after.st_dev, destination_after.st_ino),
                (destination_before.st_dev, destination_before.st_ino),
            )
            self.assertEqual(
                (source / "source.txt").read_text(encoding="utf-8"), "source\n"
            )
            self.assertEqual(
                (destination / "winner.txt").read_text(encoding="utf-8"),
                "winner\n",
            )

    def test_v0_2_package_failure_preserves_replaced_staging_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "package"
            victim = root / "victim"
            victim.mkdir()
            sentinel = victim / "sentinel.txt"
            sentinel.write_text("preserve\n", encoding="utf-8")
            captured: dict[str, str] = {}

            def replace_staging_name(
                parent_fd: int,
                source_name: str,
                destination_name: str,
                **_identity: object,
            ) -> None:
                del destination_name
                orphan_name = f"{source_name}.orphan"
                os.rename(
                    source_name,
                    orphan_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.symlink("victim", source_name, dir_fd=parent_fd)
                captured["source"] = source_name
                captured["orphan"] = orphan_name
                raise FileExistsError("injected namespace replacement")

            with patch.object(
                platform_package_module,
                "_publish_directory_noreplace",
                side_effect=replace_staging_name,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "injected namespace replacement; staging cleanup failed",
                ):
                    write_v0_2_package(destination)

            replaced_name = root / captured["source"]
            orphan = root / captured["orphan"]
            self.assertTrue(replaced_name.is_symlink())
            self.assertEqual(os.readlink(replaced_name), "victim")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve\n")
            self.assertTrue(orphan.is_dir())
            self.assertEqual(stat.S_IMODE(orphan.stat().st_mode), 0o700)
            self.assertTrue((orphan / "package-manifest.json").is_file())
            self.assertFalse(destination.exists())

    def test_v0_2_package_creation_failure_cleans_name_bound_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "package"
            real_fchmod = platform_package_module.os.fchmod
            failed_once = False

            def fail_first_private_chmod(file_descriptor: int, mode: int) -> None:
                nonlocal failed_once
                if mode == 0o700 and not failed_once:
                    failed_once = True
                    raise OSError(errno.EIO, "injected staging setup failure")
                real_fchmod(file_descriptor, mode)

            with patch.object(
                platform_package_module.os,
                "fchmod",
                side_effect=fail_first_private_chmod,
            ):
                with self.assertRaisesRegex(
                    OSError, "injected staging setup failure"
                ):
                    write_v0_2_package(destination)

            self.assertTrue(failed_once)
            self.assertEqual(list(root.iterdir()), [])

    def test_v0_2_package_read_only_check_allows_shared_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared_parent = root / "shared"
            shared_parent.mkdir()
            package = shared_parent / "package"
            write_v0_2_package(package)
            shared_parent.chmod(0o777)
            try:
                report = check_v0_2_package(package / "package-manifest.json")
                self.assertEqual(report["status"], "ok", report["errors"])
                self.assertEqual(
                    platform_package_module.package_tree_sha256(package),
                    EXPECTED_PACKAGE_TREE_SHA256,
                )
            finally:
                shared_parent.chmod(0o755)

    def test_v0_2_package_checker_enforces_canonical_manifest_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            manifest = write_v0_2_package(package)
            (package / "package-manifest.json").write_text(
                json.dumps(manifest, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            report = check_v0_2_package(package / "package-manifest.json")
            self.assertEqual(report["status"], "failed")
            self.assertIn(
                "package-manifest.json is not canonical deterministic JSON",
                report["errors"],
            )

    def test_v0_2_package_checker_rejects_non_closed_world_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            extra_dir_package = root / "extra-dir"
            write_v0_2_package(extra_dir_package)
            (extra_dir_package / "unexpected").mkdir()
            report = check_v0_2_package(
                extra_dir_package / "package-manifest.json"
            )
            self.assertIn("unexpected package directory: unexpected", report["errors"])
            self.assertIsNone(report["packageTreeSha256"])

            extra_file_package = root / "extra-file"
            write_v0_2_package(extra_file_package)
            (extra_file_package / "unexpected.bin").write_bytes(b"do not read\n")
            real_read_file_at = platform_package_module._read_file_at

            def reject_unexpected_read(
                root_fd: int,
                relative: str,
                *,
                max_bytes: int | None = None,
            ) -> bytes:
                if relative == "unexpected.bin":
                    raise AssertionError("checker read an unexpected package file")
                return real_read_file_at(
                    root_fd, relative, max_bytes=max_bytes
                )

            with patch.object(
                platform_package_module,
                "_read_file_at",
                side_effect=reject_unexpected_read,
            ):
                report = check_v0_2_package(
                    extra_file_package / "package-manifest.json"
                )
            self.assertIn(
                "unexpected package file: unexpected.bin", report["errors"]
            )
            self.assertIsNone(report["packageTreeSha256"])

            oversized_package = root / "oversized-expected-file"
            write_v0_2_package(oversized_package)
            readme = oversized_package / "README.md"
            readme.write_bytes(readme.read_bytes() + b"x")
            report = check_v0_2_package(
                oversized_package / "package-manifest.json"
            )
            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any("README.md" in error and "byte limit" in error for error in report["errors"]),
                report["errors"],
            )

            symlink_package = root / "symlink-entry"
            write_v0_2_package(symlink_package)
            (symlink_package / "unexpected-link").symlink_to("README.md")
            report = check_v0_2_package(symlink_package / "package-manifest.json")
            self.assertIn(
                "unexpected package symlink: unexpected-link", report["errors"]
            )

            fifo_package = root / "fifo-entry"
            write_v0_2_package(fifo_package)
            os.mkfifo(fifo_package / "unexpected-fifo", 0o644)
            report = check_v0_2_package(fifo_package / "package-manifest.json")
            self.assertIn(
                "unexpected package non-regular entry: unexpected-fifo",
                report["errors"],
            )

            target_package = root / "target"
            write_v0_2_package(target_package)
            linked_package = root / "linked-root"
            linked_package.symlink_to(target_package, target_is_directory=True)
            report = check_v0_2_package(linked_package / "package-manifest.json")
            self.assertIn("package root must not be a symlink", report["errors"])

            hardlink_package = root / "hardlink-entry"
            write_v0_2_package(hardlink_package)
            outside = root / "outside-readme"
            readme = hardlink_package / "README.md"
            outside.write_bytes(readme.read_bytes())
            readme.unlink()
            os.link(outside, readme)
            report = check_v0_2_package(
                hardlink_package / "package-manifest.json"
            )
            self.assertIn(
                "package file link count mismatch: README.md expected 1, found 2",
                report["errors"],
            )

            raced_package = root / "post-layout-hardlink-race"
            write_v0_2_package(raced_package)
            raced_readme = raced_package / "README.md"
            raced_outside = root / "outside-raced-readme"
            raced_outside.write_bytes(raced_readme.read_bytes())
            original_inspect = platform_package_module._inspect_package_layout_fd

            def inspect_then_replace_with_hardlink(
                root_fd: int, allowed_files: set[str]
            ) -> tuple[list[str], set[str]]:
                inspected = original_inspect(root_fd, allowed_files)
                raced_readme.unlink()
                os.link(raced_outside, raced_readme)
                return inspected

            with patch.object(
                platform_package_module,
                "_inspect_package_layout_fd",
                side_effect=inspect_then_replace_with_hardlink,
            ):
                report = check_v0_2_package(
                    raced_package / "package-manifest.json"
                )
            self.assertEqual(report["status"], "failed")
            self.assertIsNone(report["packageTreeSha256"])
            self.assertTrue(
                any(
                    "README.md" in error and "single-link" in error
                    for error in report["errors"]
                ),
                report["errors"],
            )
            self.assertEqual(raced_readme.stat().st_nlink, 2)

    def test_v0_2_package_checker_enforces_exact_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            write_v0_2_package(package)
            package.chmod(0o700)
            (package / "README.md").chmod(0o600)
            (package / "schemas").chmod(0o700)
            (package / "kaggle/run_conformance.py").chmod(0o644)
            report = check_v0_2_package(package / "package-manifest.json")
            self.assertIn(
                "package directory mode mismatch: . expected 0755, found 0700",
                report["errors"],
            )
            self.assertIn(
                "package file mode mismatch: README.md expected 0644, found 0600",
                report["errors"],
            )
            self.assertIn(
                "package directory mode mismatch: schemas expected 0755, found 0700",
                report["errors"],
            )
            self.assertIn(
                "package file mode mismatch: kaggle/run_conformance.py expected 0755, found 0644",
                report["errors"],
            )

    @unittest.skipUnless(
        CANONICAL_SCORING_RUNTIME,
        "fixture-drift checks require the canonical scorer runtime",
    )
    def test_packaged_conformance_rejects_bool_nonfinite_and_field_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_fixture = json.loads(
                (ROOT / "bench/conformance/scorer-v0.2.json").read_text(
                    encoding="utf-8"
                )
            )

            cases = {
                "bool": lambda fixture: fixture["fidelityVectors"][0]["expected"].__setitem__(
                    "exact", False
                ),
                "nonfinite": lambda fixture: fixture["fidelityVectors"][0][
                    "expected"
                ].__setitem__("normalized_edit_similarity", float("nan")),
                "field-drift": lambda fixture: fixture["leakageVectors"][0][
                    "expected"
                ].pop("unit"),
            }
            for name, mutate in cases.items():
                with self.subTest(name=name):
                    package = root / name
                    write_v0_2_package(package)
                    fixture = json.loads(json.dumps(source_fixture))
                    mutate(fixture)
                    (package / "conformance/scorer-v0.2.json").write_text(
                        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    completed = subprocess.run(
                        [sys.executable, str(package / "kaggle/run_conformance.py")],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 1)
                    report = json.loads(completed.stdout)
                    self.assertEqual(report["status"], "failed")
                    self.assertTrue(report["errors"])


if __name__ == "__main__":
    unittest.main()
