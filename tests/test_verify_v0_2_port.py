from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench.engine.frozen_ladder import run_benchmark
from bench.engine.manifest import build_manifest
from bench.engine import verify as verifier
from bench.engine.schema_validation import SchemaValidationError, load_schema, validate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "bench/data/v0.2/public/s2"


class VerifierProtocolBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_benchmark(
            data_dir=DATA_DIR,
            models=["mock-frontier"],
            seed=29,
        )
        cls.manifest = build_manifest(
            data_dir=DATA_DIR,
            models=["mock-frontier"],
            seed=29,
        )

    def verify(
        self,
        *,
        result: dict[str, object] | None = None,
        manifest: dict[str, object] | None = None,
        data_dir: Path = DATA_DIR,
        audit_path: Path | None = None,
        bundle_path: Path | None = None,
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
            return verifier.verify_artifacts(
                data_dir=data_dir,
                result_path=result_path,
                manifest_path=manifest_path,
                audit_path=audit_path,
                bundle_path=bundle_path,
            )

    def test_canonical_receipts_return_a_digest_bound_success_report(self) -> None:
        report = self.verify()

        self.assertEqual(report["status"], "ok", msg=report["errors"])
        self.assertEqual(report["datasetSha256"], verifier.FROZEN_DATASET_SHA256)
        self.assertRegex(report["resultSha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(report["manifestSha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            report["resultAlephRunContractCount"], len(self.result["itemRuns"])
        )

    def test_missing_and_unknown_protocol_versions_fail_closed(self) -> None:
        for version in (None, "9.9.9"):
            with self.subTest(version=version):
                result = copy.deepcopy(self.result)
                manifest = copy.deepcopy(self.manifest)
                if version is None:
                    result.pop("protocolVersion")
                    manifest.pop("protocolVersion")
                else:
                    result["protocolVersion"] = version
                    manifest["protocolVersion"] = version

                report = self.verify(result=result, manifest=manifest)

                self.assertEqual(report["status"], "failed")
                self.assertTrue(
                    any(
                        "unsupported protocol versions" in error
                        for error in report["errors"]
                    ),
                    msg=report["errors"],
                )

    def test_legacy_audit_and_bundle_arguments_are_rejected_without_reading(self) -> None:
        report = self.verify(
            audit_path=Path("does-not-exist-audit.json"),
            bundle_path=Path("does-not-exist-bundle.json"),
        )

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "v0.2 verification does not accept a legacy audit artifact",
            report["errors"],
        )
        self.assertIn(
            "v0.2 verification does not accept a legacy evidence bundle",
            report["errors"],
        )

    def test_nested_aleph_run_is_validated_from_the_v0_2_schema_root(self) -> None:
        result = copy.deepcopy(self.result)
        result_schema = load_schema(
            REPOSITORY_ROOT / "schemas/v0.2/aleph-bench-result.schema.json"
        )
        aleph_run_schema = result_schema["$defs"]["alephRun"]
        validate(
            result["itemRuns"][0]["alephRun"],
            aleph_run_schema,
            root=result_schema,
        )
        result["itemRuns"][0]["alephRun"]["unexpected"] = True
        with self.assertRaises(SchemaValidationError):
            validate(
                result["itemRuns"][0]["alephRun"],
                aleph_run_schema,
                root=result_schema,
            )

        report = self.verify(result=result)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("unexpected keys" in error for error in report["errors"]),
            msg=report["errors"],
        )

    def test_malformed_dataset_returns_failed_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / "s2"
            shutil.copytree(DATA_DIR, copied)
            (copied / "s2-001.json").write_text("{}", encoding="utf-8")

            report = self.verify(data_dir=copied)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("canonical v0.2 dataset check failed" in row for row in report["errors"]),
            msg=report["errors"],
        )

    def test_result_path_replacement_during_verification_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_path = root / "result.json"
            manifest_path = root / "manifest.json"
            replacement = root / "replacement.json"
            result_path.write_text(json.dumps(self.result), encoding="utf-8")
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            replacement.write_text("{}", encoding="utf-8")
            original = verifier._verify_v0_2_manifest_receipts

            def replace_after_manifest_check(**kwargs: object) -> None:
                original(**kwargs)
                os.replace(replacement, result_path)

            with mock.patch.object(
                verifier,
                "_verify_v0_2_manifest_receipts",
                side_effect=replace_after_manifest_check,
            ):
                report = verifier.verify_artifacts(
                    data_dir=DATA_DIR,
                    result_path=result_path,
                    manifest_path=manifest_path,
                )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("BenchResult path changed after reading" in row for row in report["errors"]),
            msg=report["errors"],
        )


class VerifierInputBoundaryTests(unittest.TestCase):
    def test_deeply_nested_json_returns_failed_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_path = root / "result.json"
            manifest_path = root / "manifest.json"
            result_path.write_bytes(b"[" * 20_000 + b"0" + b"]" * 20_000)
            manifest_path.write_text("{}", encoding="utf-8")

            report = verifier.verify_artifacts(
                data_dir=DATA_DIR,
                result_path=result_path,
                manifest_path=manifest_path,
            )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("invalid BenchResult JSON" in row for row in report["errors"]),
            msg=report["errors"],
        )

    def test_escaped_unpaired_surrogate_returns_failed_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_path = root / "result.json"
            manifest_path = root / "manifest.json"
            result_path.write_bytes(b'{"notes":["\\ud800"]}')
            manifest_path.write_text("{}", encoding="utf-8")

            report = verifier.verify_artifacts(
                data_dir=DATA_DIR,
                result_path=result_path,
                manifest_path=manifest_path,
            )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any("surrogate code points are forbidden" in row for row in report["errors"]),
            msg=report["errors"],
        )

    def test_runtime_incompatibility_is_a_deterministic_verification_error(self) -> None:
        errors: list[str] = []
        try:
            verifier.validate_scoring_runtime()
        except RuntimeError:
            verifier._verify_v0_2_result_receipts(items=[], result={}, errors=errors)
        else:
            with mock.patch.object(
                verifier,
                "validate_scoring_runtime",
                side_effect=RuntimeError("fixture incompatible runtime"),
            ):
                verifier._verify_v0_2_result_receipts(
                    items=[], result={}, errors=errors
                )
        self.assertEqual(len(errors), 1)
        self.assertTrue(
            errors[0].startswith("result receipt replay runtime incompatible:"),
            msg=errors,
        )

    def test_symlink_hardlink_fifo_and_oversize_inputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original = root / "original.json"
            original.write_text("{}", encoding="utf-8")

            symlink = root / "symlink.json"
            symlink.symlink_to(original)
            with self.assertRaisesRegex(ValueError, "could not open"):
                verifier._read_regular_path(
                    symlink, role="result", max_bytes=verifier.MAX_RESULT_BYTES
                )

            hardlink = root / "hardlink.json"
            os.link(original, hardlink)
            with self.assertRaisesRegex(ValueError, "exactly one hard link"):
                verifier._read_regular_path(
                    original, role="result", max_bytes=verifier.MAX_RESULT_BYTES
                )

            fifo = root / "fifo.json"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "regular file"):
                verifier._read_regular_path(
                    fifo, role="result", max_bytes=verifier.MAX_RESULT_BYTES
                )

            oversize = root / "oversize.json"
            with oversize.open("wb") as stream:
                stream.truncate(verifier.MAX_MANIFEST_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "verification limit"):
                verifier._read_regular_path(
                    oversize,
                    role="manifest",
                    max_bytes=verifier.MAX_MANIFEST_BYTES,
                )

    def test_input_metadata_change_during_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "result.json"
            path.write_text("{}", encoding="utf-8")
            real_fstat = os.fstat
            calls = 0

            def changed_fstat(fd: int) -> os.stat_result | SimpleNamespace:
                nonlocal calls
                calls += 1
                observed = real_fstat(fd)
                if calls == 1:
                    return observed
                return SimpleNamespace(
                    st_dev=observed.st_dev,
                    st_ino=observed.st_ino,
                    st_mode=observed.st_mode,
                    st_nlink=observed.st_nlink,
                    st_size=observed.st_size,
                    st_mtime_ns=observed.st_mtime_ns + 1,
                    st_ctime_ns=observed.st_ctime_ns,
                )

            with (
                mock.patch.object(verifier.os, "fstat", side_effect=changed_fstat),
                self.assertRaisesRegex(ValueError, "changed while it was being read"),
            ):
                verifier._read_regular_path(
                    path, role="result", max_bytes=verifier.MAX_RESULT_BYTES
                )

    def test_dataset_directory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            symlink = Path(temporary_directory) / "s2"
            symlink.symlink_to(DATA_DIR, target_is_directory=True)
            errors: list[str] = []

            items, digest, binding = verifier._load_dataset(symlink, errors)

            self.assertEqual(items, [])
            self.assertIsNone(digest)
            self.assertIsNone(binding)
            self.assertTrue(any("could not open dataset directory" in row for row in errors))

    def test_dataset_cardinality_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index in range(verifier.FROZEN_DATASET_ITEM_COUNT + 1):
                (root / f"{index:03d}.json").write_text("{}", encoding="utf-8")
            errors: list[str] = []
            items, digest, binding = verifier._load_dataset(root, errors)
            self.assertEqual(items, [])
            self.assertIsNone(digest)
            self.assertIsNone(binding)
            self.assertTrue(any("found more than" in row for row in errors))

    def test_surrogateescape_dataset_filename_is_rejected(self) -> None:
        if os.name != "posix":
            self.skipTest("surrogateescape filename coverage requires POSIX")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            encoded_path = os.fsencode(root) + b"/\xff.json"
            try:
                descriptor = os.open(
                    encoded_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
            except OSError as exc:
                self.skipTest(f"filesystem rejected a surrogateescape filename: {exc}")
            os.close(descriptor)
            errors = []
            items, digest, binding = verifier._load_dataset(root, errors)
            self.assertEqual(items, [])
            self.assertIsNone(digest)
            self.assertIsNone(binding)
            self.assertTrue(any("not valid UTF-8" in row for row in errors))

    def test_dataset_entry_replacement_during_scan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / "s2"
            shutil.copytree(DATA_DIR, copied)
            target = copied / "s2-001.json"
            replacement = Path(temporary_directory) / "replacement.tmp"
            replacement.write_text("{}", encoding="utf-8")
            original = verifier._decode_json
            replaced = False

            def replace_first_item(raw: bytes, *, role: str) -> object:
                nonlocal replaced
                value = original(raw, role=role)
                if not replaced and role == "dataset item s2-001.json":
                    os.replace(replacement, target)
                    replaced = True
                return value

            errors: list[str] = []
            with mock.patch.object(
                verifier, "_decode_json", side_effect=replace_first_item
            ):
                _, _, binding = verifier._load_dataset(copied, errors)

            self.assertIsNotNone(binding)
            verifier._recheck_dataset_binding(binding, errors)
            self.assertTrue(
                any("changed after reading" in row for row in errors),
                msg=errors,
            )


if __name__ == "__main__":
    unittest.main()
