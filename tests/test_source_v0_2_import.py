from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPOSITORY_ROOT / "scripts/check-source-v0.2-import.py"
SPEC = importlib.util.spec_from_file_location("source_v0_2_import", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


class InventoryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = (REPOSITORY_ROOT / checker.INVENTORY_PATH).read_bytes()
        cls.manifest, cls.copy_rows = checker._parse_inventory_bytes(cls.raw)
        cls.receipt_raw = (REPOSITORY_ROOT / checker.E3A_RECEIPT_PATH).read_bytes()
        cls.e3a_transformations = checker._parse_e3a_receipt_bytes(
            cls.receipt_raw, cls.manifest
        )
        cls.e3b_receipt_raw = (
            REPOSITORY_ROOT / checker.E3B_RECEIPT_PATH
        ).read_bytes()
        cls.e3b_transformations = checker._parse_e3b_receipt_bytes(
            cls.e3b_receipt_raw, cls.manifest
        )
        cls.e3c_receipt_raw = (
            REPOSITORY_ROOT / checker.E3C_RECEIPT_PATH
        ).read_bytes()
        cls.e3c_transformations = checker._parse_e3c_receipt_bytes(
            cls.e3c_receipt_raw, cls.manifest
        )
        cls.e3d_receipt_raw = (
            REPOSITORY_ROOT / checker.E3D_RECEIPT_PATH
        ).read_bytes()
        cls.e3d_transformations = checker._parse_e3d_receipt_bytes(
            cls.e3d_receipt_raw, cls.manifest
        )
        cls.e3e_receipt_raw = (
            REPOSITORY_ROOT / checker.E3E_RECEIPT_PATH
        ).read_bytes()
        cls.e3e_transformations = checker._parse_e3e_receipt_bytes(
            cls.e3e_receipt_raw, cls.manifest
        )
        cls.e4a_inventory_raw = (
            REPOSITORY_ROOT / checker.E4A_INVENTORY_PATH
        ).read_bytes()
        cls.e4a_inventory, cls.e4a_rows = checker._parse_e4a_inventory_bytes(
            cls.e4a_inventory_raw
        )
        cls.e4a_receipt_raw = (
            REPOSITORY_ROOT / checker.E4A_RECEIPT_PATH
        ).read_bytes()
        cls.e4a_transformations = checker._parse_e4a_receipt_bytes(
            cls.e4a_receipt_raw,
            cls.e4a_rows,
        )
        cls.e4b_inventory_raw = (
            REPOSITORY_ROOT / checker.E4B_INVENTORY_PATH
        ).read_bytes()
        cls.e4b_inventory, cls.e4b_rows = checker._parse_e4b_inventory_bytes(
            cls.e4b_inventory_raw
        )
        cls.e4b_receipt_raw = (
            REPOSITORY_ROOT / checker.E4B_RECEIPT_PATH
        ).read_bytes()
        cls.e4b_receipt = checker._parse_e4b_receipt_bytes(
            cls.e4b_receipt_raw
        )

    def test_reviewed_inventory_and_copy_selection(self) -> None:
        self.assertEqual(len(self.copy_rows), 78)
        self.assertEqual(sum(row["bytes"] for row in self.copy_rows), 842_943)
        self.assertTrue(
            all(row["source"] == row["destination"] for row in self.copy_rows)
        )

    def test_duplicate_json_key_is_rejected(self) -> None:
        text = self.raw.decode("ascii")
        marker = '  "artifactKind": "aleph_bench_extraction_inventory",\n'
        tampered = text.replace(marker, marker + marker, 1).encode("ascii")
        with self.assertRaisesRegex(checker.ImportCheckError, "duplicate JSON key"):
            checker._parse_inventory_bytes(tampered, expected_raw_sha256=None)

    def test_boolean_format_version_is_rejected(self) -> None:
        text = self.raw.decode("ascii")
        tampered = text.replace('  "formatVersion": 1,', '  "formatVersion": true,', 1)
        with self.assertRaisesRegex(checker.ImportCheckError, "must be an integer"):
            checker._parse_inventory_bytes(
                tampered.encode("ascii"), expected_raw_sha256=None
            )

    def test_canonical_manifest_tamper_is_rejected(self) -> None:
        value = json.loads(self.raw)
        value["source"]["commit"] = "0" * 40
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw inventory SHA-256"):
            checker._parse_inventory_bytes(tampered)

    def test_reviewed_e3a_receipt_and_transformations(self) -> None:
        self.assertEqual(
            [entry["destination"] for entry in self.e3a_transformations],
            ["bench/engine/metrics.py", "docs/protocol-v0.2.md"],
        )
        self.assertEqual(
            self.e3a_transformations[0]["transformation"]["algorithm"],
            "utf8-replace-once-v1",
        )

    def test_e3a_receipt_tamper_is_rejected(self) -> None:
        value = json.loads(self.receipt_raw)
        value["transformations"][0]["output"]["bytes"] += 1
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw E3a receipt"):
            checker._parse_e3a_receipt_bytes(tampered, self.manifest)

    def test_reviewed_e3b_receipt_and_transformation(self) -> None:
        self.assertEqual(
            [entry["destination"] for entry in self.e3b_transformations],
            ["bench/engine/report.py"],
        )
        transformation = self.e3b_transformations[0]["transformation"]
        self.assertEqual(
            transformation["algorithm"],
            "utf8-replace-once-sequence-v1",
        )
        self.assertEqual(len(transformation["replacements"]), 5)

    def test_e3b_receipt_tamper_is_rejected(self) -> None:
        value = json.loads(self.e3b_receipt_raw)
        value["standaloneBaseCommit"] = "0" * 40
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw E3b receipt"):
            checker._parse_e3b_receipt_bytes(tampered, self.manifest)

    def test_e3b_replacement_contract_tamper_is_rejected_without_digest_pin(self) -> None:
        value = json.loads(self.e3b_receipt_raw)
        value["transformations"][0]["transformation"]["replacements"].pop()
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "five replacements"):
            checker._parse_e3b_receipt_bytes(
                tampered,
                self.manifest,
                expected_raw_sha256=None,
            )

    def test_reviewed_e3c_receipt_and_transformations(self) -> None:
        self.assertEqual(
            [entry["destination"] for entry in self.e3c_transformations],
            list(checker.E3C_SOURCE_PATHS),
        )
        self.assertEqual(
            [
                entry["transformation"]["algorithm"]
                for entry in self.e3c_transformations
            ],
            [checker.E3C_TRANSFORMATION_ALGORITHM] * 2,
        )
        receipt = json.loads(self.e3c_receipt_raw)
        self.assertEqual(
            receipt["packageTree"],
            {
                "algorithm": checker.E3C_PACKAGE_TREE_HASH_ALGORITHM,
                "fileCount": checker.E3C_PACKAGE_TREE_FILES,
                "sha256": checker.E3C_PACKAGE_TREE_SHA256,
            },
        )

    def test_e3c_receipt_tamper_is_rejected(self) -> None:
        value = json.loads(self.e3c_receipt_raw)
        value["packageTree"]["sha256"] = "0" * 64
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw E3c receipt"):
            checker._parse_e3c_receipt_bytes(tampered, self.manifest)

    def test_e3c_splice_contract_tamper_is_rejected_without_digest_pin(self) -> None:
        value = json.loads(self.e3c_receipt_raw)
        edit = value["transformations"][0]["transformation"]["edits"][0]
        edit["outputCharOffset"] += 1
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(
            checker.ImportCheckError, "source/output offsets are inconsistent"
        ):
            checker._parse_e3c_receipt_bytes(
                tampered,
                self.manifest,
                expected_raw_sha256=None,
            )

    def test_reviewed_e3d_receipt_and_transformation(self) -> None:
        self.assertEqual(
            [entry["destination"] for entry in self.e3d_transformations],
            [checker.E3D_SOURCE_PATH],
        )
        self.assertEqual(
            self.e3d_transformations[0]["transformation"]["algorithm"],
            checker.E3C_TRANSFORMATION_ALGORITHM,
        )

    def test_e3d_receipt_tamper_is_rejected(self) -> None:
        value = json.loads(self.e3d_receipt_raw)
        value["standaloneBaseCommit"] = "0" * 40
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw E3d receipt"):
            checker._parse_e3d_receipt_bytes(tampered, self.manifest)

    def test_e3d_splice_contract_tamper_is_rejected_without_digest_pin(self) -> None:
        value = json.loads(self.e3d_receipt_raw)
        edit = value["transformations"][0]["transformation"]["edits"][0]
        edit["outputCharOffset"] += 1
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(
            checker.ImportCheckError, "source/output offsets are inconsistent"
        ):
            checker._parse_e3d_receipt_bytes(
                tampered,
                self.manifest,
                expected_raw_sha256=None,
            )

    def test_reviewed_e3e_receipt_and_transformation(self) -> None:
        self.assertEqual(
            [entry["destination"] for entry in self.e3e_transformations],
            [checker.E3E_SOURCE_PATH],
        )
        entry = self.e3e_transformations[0]
        self.assertEqual(
            entry["transformation"]["algorithm"],
            checker.E3C_TRANSFORMATION_ALGORITHM,
        )
        self.assertEqual(entry["source"]["bytes"], 16_766)
        self.assertEqual(
            entry["source"]["sha256"],
            "5a4a67bd9979d84c2ca6e402df55967c0c86186e246bdde056d517af9ccd9c51",
        )

    def test_reviewed_e4a_inventory_and_receipt(self) -> None:
        self.assertEqual(len(self.e4a_rows), 17)
        self.assertEqual(
            sum(row["disposition"] == "copy" for row in self.e4a_rows),
            15,
        )
        self.assertEqual(
            tuple(row["source"] for row in self.e4a_rows),
            checker.E4A_SOURCE_PATHS,
        )
        self.assertEqual(
            [entry["destination"] for entry in self.e4a_transformations],
            list(checker.E4A_PORT_PATHS),
        )

    def test_e4a_inventory_tamper_is_rejected(self) -> None:
        value = json.loads(self.e4a_inventory_raw)
        value["summary"]["sourceFiles"] += 1
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw E4a delta"):
            checker._parse_e4a_inventory_bytes(tampered)

    def test_e4a_inventory_contract_tamper_fails_without_digest_pin(self) -> None:
        value = json.loads(self.e4a_inventory_raw)
        value["files"][0]["disposition"] = "copy"
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "disposition differs"):
            checker._parse_e4a_inventory_bytes(
                tampered,
                expected_raw_sha256=None,
            )

    def test_e4a_receipt_tamper_is_rejected(self) -> None:
        value = json.loads(self.e4a_receipt_raw)
        value["installedTree"]["sha256"] = "0" * 64
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw E4a receipt"):
            checker._parse_e4a_receipt_bytes(tampered, self.e4a_rows)

    def test_e4a_transform_contract_tamper_fails_without_digest_pin(self) -> None:
        value = json.loads(self.e4a_receipt_raw)
        value["transformations"][1]["transformation"]["authorityProjection"][
            "replacement"
        ]["to"] = "docs/other.md"
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(
            checker.ImportCheckError,
            "authority-checker transformation differs",
        ):
            checker._parse_e4a_receipt_bytes(
                tampered,
                self.e4a_rows,
                expected_raw_sha256=None,
            )

    def test_reviewed_e4b_inventory_and_receipt(self) -> None:
        self.assertEqual(len(self.e4b_rows), 11)
        self.assertEqual(
            tuple(
                row["source"]
                for row in self.e4b_rows
                if row["disposition"] == "copy"
            ),
            checker.E4B_COPY_PATHS,
        )
        self.assertEqual(
            [
                row["source"]
                for row in self.e4b_rows
                if row["disposition"] == "exclude"
            ],
            [checker.E4B_EXCLUDED_PATH],
        )
        self.assertEqual(self.e4b_receipt["copyFiles"], 10)
        self.assertEqual(
            self.e4b_receipt["sourceImmediateParentCommit"],
            checker.E4B_SOURCE_IMMEDIATE_PARENT_COMMIT,
        )
        self.assertEqual(
            self.e4b_receipt["sourceReviewedDependencyCommit"],
            checker.E4A_SOURCE_COMMIT,
        )

    def test_e4b_inventory_tamper_is_rejected(self) -> None:
        value = json.loads(self.e4b_inventory_raw)
        value["summary"]["copyFiles"] += 1
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw E4b delta"):
            checker._parse_e4b_inventory_bytes(tampered)

    def test_e4b_inventory_contract_tamper_fails_without_digest_pin(self) -> None:
        value = json.loads(self.e4b_inventory_raw)
        value["files"][0]["destination"] = "bench/README.md"
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "exclusion contract"):
            checker._parse_e4b_inventory_bytes(
                tampered,
                expected_raw_sha256=None,
            )

    def test_e4b_receipt_tamper_is_rejected(self) -> None:
        value = json.loads(self.e4b_receipt_raw)
        value["installedTree"]["sha256"] = "0" * 64
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw E4b receipt"):
            checker._parse_e4b_receipt_bytes(tampered)

    def test_e4b_receipt_contract_tamper_fails_without_digest_pin(self) -> None:
        value = json.loads(self.e4b_receipt_raw)
        value["sourceReviewedDependencyCommit"] = "0" * 40
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "receipt contract"):
            checker._parse_e4b_receipt_bytes(
                tampered,
                expected_raw_sha256=None,
            )

    def test_e3e_receipt_tamper_is_rejected(self) -> None:
        value = json.loads(self.e3e_receipt_raw)
        value["transformations"][0]["output"]["bytes"] += 1
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(checker.ImportCheckError, "raw E3e receipt"):
            checker._parse_e3e_receipt_bytes(tampered, self.manifest)

    def test_e3e_source_metadata_tamper_is_rejected_without_digest_pin(self) -> None:
        value = json.loads(self.e3e_receipt_raw)
        value["transformations"][0]["source"]["gitBlobSha1"] = "0" * 40
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(
            checker.ImportCheckError, "runner transformation differs from inventory"
        ):
            checker._parse_e3e_receipt_bytes(
                tampered,
                self.manifest,
                expected_raw_sha256=None,
            )

    def test_e3e_splice_contract_tamper_is_rejected_without_digest_pin(self) -> None:
        value = json.loads(self.e3e_receipt_raw)
        edit = value["transformations"][0]["transformation"]["edits"][0]
        edit["outputCharOffset"] += 1
        tampered = checker._canonical_file_json(value)
        with self.assertRaisesRegex(
            checker.ImportCheckError, "source/output offsets are inconsistent"
        ):
            checker._parse_e3e_receipt_bytes(
                tampered,
                self.manifest,
                expected_raw_sha256=None,
            )

    def test_path_attacks_are_rejected(self) -> None:
        attacks = (
            "/absolute",
            "C:/drive",
            "../escape",
            "a/../escape",
            "./relative",
            "a//b",
            "a/",
            "a\\b",
            "a\nb",
            "a\x00b",
            ".git/config",
            "safe/.GIT/config",
            "café",
        )
        for value in attacks:
            with self.subTest(value=value):
                with self.assertRaises(checker.ImportCheckError):
                    checker._validate_repository_path(value, role="test")

    def test_portable_and_prefix_collisions_are_rejected(self) -> None:
        for paths in (("A/file", "a/file"), ("tree", "tree/child")):
            with self.subTest(paths=paths):
                with self.assertRaises(checker.ImportCheckError):
                    checker._validate_unique_paths(paths, role="test")


class RootClosureTests(unittest.TestCase):
    def materialize(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        for path in checker.EXPECTED_ROOT_FILES:
            (root / path).touch()
        for path in checker.EXPECTED_ROOT_DIRECTORIES:
            (root / path).mkdir()
        (root / "platform/m0-mock").mkdir()
        return temporary, root

    def test_expected_root_entries_pass(self) -> None:
        temporary, root = self.materialize()
        with temporary:
            checker._verify_root_entries(root)

    def test_top_level_import_shadow_is_rejected(self) -> None:
        for filename in ("random.py", "statistics.py"):
            with self.subTest(filename=filename):
                temporary, root = self.materialize()
                with temporary:
                    (root / filename).write_text("shadow = True\n")
                    with self.assertRaisesRegex(checker.ImportCheckError, "extra"):
                        checker._verify_root_entries(root)

    def test_standard_library_package_shadow_is_rejected(self) -> None:
        for filename in ("__init__.py", "__init__.pyc"):
            with self.subTest(filename=filename):
                temporary, root = self.materialize()
                with temporary:
                    (root / "platform" / filename).write_bytes(b"shadow\n")
                    with self.assertRaisesRegex(checker.ImportCheckError, "shadow"):
                        checker._verify_root_entries(root)

    def test_root_symlink_is_rejected(self) -> None:
        temporary, root = self.materialize()
        with temporary:
            readme = root / "README.md"
            readme.unlink()
            readme.symlink_to("NOTICE")
            with self.assertRaisesRegex(checker.ImportCheckError, "symlink"):
                checker._verify_root_entries(root)


class InstalledTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw = (REPOSITORY_ROOT / checker.INVENTORY_PATH).read_bytes()
        cls.manifest, cls.copy_rows = checker._parse_inventory_bytes(raw)
        receipt_raw = (REPOSITORY_ROOT / checker.E3A_RECEIPT_PATH).read_bytes()
        e3a_transformations = checker._parse_e3a_receipt_bytes(
            receipt_raw, cls.manifest
        )
        e3b_receipt_raw = (
            REPOSITORY_ROOT / checker.E3B_RECEIPT_PATH
        ).read_bytes()
        e3b_transformations = checker._parse_e3b_receipt_bytes(
            e3b_receipt_raw, cls.manifest
        )
        e3c_receipt_raw = (
            REPOSITORY_ROOT / checker.E3C_RECEIPT_PATH
        ).read_bytes()
        e3c_transformations = checker._parse_e3c_receipt_bytes(
            e3c_receipt_raw, cls.manifest
        )
        e3d_receipt_raw = (
            REPOSITORY_ROOT / checker.E3D_RECEIPT_PATH
        ).read_bytes()
        e3d_transformations = checker._parse_e3d_receipt_bytes(
            e3d_receipt_raw, cls.manifest
        )
        e3e_receipt_raw = (
            REPOSITORY_ROOT / checker.E3E_RECEIPT_PATH
        ).read_bytes()
        e3e_transformations = checker._parse_e3e_receipt_bytes(
            e3e_receipt_raw, cls.manifest
        )
        e4a_inventory_raw = (
            REPOSITORY_ROOT / checker.E4A_INVENTORY_PATH
        ).read_bytes()
        cls.e4a_inventory, cls.e4a_rows = checker._parse_e4a_inventory_bytes(
            e4a_inventory_raw
        )
        e4a_receipt_raw = (
            REPOSITORY_ROOT / checker.E4A_RECEIPT_PATH
        ).read_bytes()
        cls.e4a_transformations = checker._parse_e4a_receipt_bytes(
            e4a_receipt_raw,
            cls.e4a_rows,
        )
        e4b_inventory_raw = (
            REPOSITORY_ROOT / checker.E4B_INVENTORY_PATH
        ).read_bytes()
        cls.e4b_inventory, cls.e4b_rows = checker._parse_e4b_inventory_bytes(
            e4b_inventory_raw
        )
        e4b_receipt_raw = (
            REPOSITORY_ROOT / checker.E4B_RECEIPT_PATH
        ).read_bytes()
        cls.e4b_receipt = checker._parse_e4b_receipt_bytes(e4b_receipt_raw)
        cls.transformations = (
            e3a_transformations
            + e3b_transformations
            + e3c_transformations
            + e3d_transformations
            + e3e_transformations
        )

    def materialize(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        for row in self.copy_rows:
            source = REPOSITORY_ROOT / row["source"]
            destination = root / row["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o755 if row["mode"] == "100755" else 0o644)
        for entry in self.transformations:
            source = REPOSITORY_ROOT / entry["destination"]
            destination = root / entry["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
        for path in checker.SUPPORT_MANAGED_PATHS:
            source = REPOSITORY_ROOT / path
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
        for row in self.e4a_rows:
            source = REPOSITORY_ROOT / row["destination"]
            destination = root / row["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
        for row in self.e4b_rows:
            if row["disposition"] != "copy":
                continue
            source = REPOSITORY_ROOT / row["destination"]
            destination = root / row["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
        shutil.copyfile(REPOSITORY_ROOT / "NOTICE", root / "NOTICE")
        (root / "NOTICE").chmod(0o644)
        return temporary, root

    def assert_rejected(self, mutate) -> None:
        temporary, root = self.materialize()
        with temporary:
            mutate(root)
            with self.assertRaises(checker.ImportCheckError):
                checker._verify_installed(
                    root,
                    self.manifest,
                    self.copy_rows,
                    self.transformations,
                    self.e4a_rows,
                    self.e4b_rows,
                )

    def assert_e4a_rejected(self, mutate) -> None:
        temporary, root = self.materialize()
        with temporary:
            mutate(root)
            with self.assertRaises(checker.ImportCheckError):
                checker._verify_e4a_installed(
                    root,
                    self.e4a_rows,
                    self.e4a_transformations,
                )

    def assert_e4b_rejected(self, mutate) -> None:
        temporary, root = self.materialize()
        with temporary:
            mutate(root)
            with self.assertRaises(checker.ImportCheckError):
                checker._verify_e4b_installed(root, self.e4b_rows)

    def test_complete_tree_passes(self) -> None:
        temporary, root = self.materialize()
        with temporary:
            checker._verify_installed(
                root,
                self.manifest,
                self.copy_rows,
                self.transformations,
                self.e4a_rows,
                self.e4b_rows,
            )
            checker._verify_e4a_installed(
                root,
                self.e4a_rows,
                self.e4a_transformations,
            )
            checker._verify_e4b_installed(root, self.e4b_rows)

    def test_missing_file_is_rejected(self) -> None:
        self.assert_rejected(lambda root: (root / "bench/__init__.py").unlink())

    def test_extra_file_is_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (root / "bench/extra.py").write_text("extra\n")
        )

    def test_extra_vendor_file_is_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (root / "bench/vendor/unreviewed.bin").write_bytes(b"x")
        )

    def test_missing_or_extra_e4b_file_is_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (root / checker.E4B_COPY_PATHS[0]).unlink()
        )
        self.assert_rejected(
            lambda root: (root / "bench/portable/data/extra.json").write_text("{}\n")
        )

    def test_extra_schema_outside_v0_2_is_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (root / "schemas/legacy.schema.json").write_text("{}\n")
        )

    def test_executable_mode_drift_is_rejected(self) -> None:
        self.assert_rejected(lambda root: (root / "aleph-bench").chmod(0o644))

    def test_group_only_execute_does_not_satisfy_executable_mode(self) -> None:
        self.assert_rejected(lambda root: (root / "aleph-bench").chmod(0o410))

    def test_execute_bits_on_non_executable_output_are_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (
                root / "bench/engine/platform_package_v0_2.py"
            ).chmod(0o655)
        )

    def test_execute_bits_on_runner_output_are_rejected(self) -> None:
        self.assert_rejected(lambda root: (root / checker.E3E_SOURCE_PATH).chmod(0o755))

    def test_missing_runner_test_is_rejected_by_managed_tree(self) -> None:
        self.assert_rejected(
            lambda root: (root / "tests/test_run_v0_2_port.py").unlink()
        )

    def test_symlink_is_rejected(self) -> None:
        def mutate(root: Path) -> None:
            path = root / "bench/__init__.py"
            path.unlink()
            path.symlink_to("elsewhere")

        self.assert_rejected(mutate)

    def test_e4a_symlink_is_rejected(self) -> None:
        def mutate(root: Path) -> None:
            path = root / "bench/vendor/portable-unicode-inputs-v1.json"
            path.unlink()
            path.symlink_to("README.md")

        self.assert_e4a_rejected(mutate)

    def test_e4a_hardlink_is_rejected(self) -> None:
        def mutate(root: Path) -> None:
            path = root / "bench/vendor/portable-unicode-inputs-v1.json"
            outside = root / "outside-manifest.json"
            shutil.move(path, outside)
            os.link(outside, path)

        self.assert_e4a_rejected(mutate)

    def test_e4a_mode_and_byte_drift_are_rejected(self) -> None:
        self.assert_e4a_rejected(
            lambda root: (
                root / "bench/vendor/portable-unicode-inputs-v1.json"
            ).chmod(0o755)
        )

        def mutate_wheel(root: Path) -> None:
            path = root / self.e4a_rows[12]["destination"]
            data = path.read_bytes()
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])

        self.assert_e4a_rejected(mutate_wheel)

    def test_e4a_ports_and_metrics_projection_are_exact(self) -> None:
        self.assert_e4a_rejected(
            lambda root: (root / ".gitattributes").write_text("drift\n")
        )
        self.assert_e4a_rejected(
            lambda root: (
                root / "scripts/check-v0-2-scoring-authority.py"
            ).write_text("drift\n")
        )
        self.assert_e4a_rejected(
            lambda root: (root / checker.E4A_METRICS_PATH).write_bytes(
                (root / checker.E4A_METRICS_PATH)
                .read_bytes()
                .replace(b"Lower is better", b"Lower is bettor", 1)
            )
        )

    def test_e4b_symlink_hardlink_mode_and_byte_drift_are_rejected(self) -> None:
        target_path = "bench/portable/data/portable-string-semantics-v1.manifest.json"

        def symlink(root: Path) -> None:
            target = root / target_path
            target.unlink()
            target.symlink_to("portable-string-semantics-v1.schema.json")

        self.assert_e4b_rejected(symlink)

        def hardlink(root: Path) -> None:
            target = root / target_path
            outside = root / "outside-e4b-manifest.json"
            shutil.move(target, outside)
            os.link(outside, target)

        self.assert_e4b_rejected(hardlink)
        self.assert_e4b_rejected(lambda root: (root / target_path).chmod(0o755))

        def byte_drift(root: Path) -> None:
            target = root / target_path
            data = target.read_bytes()
            target.write_bytes(data[:-2] + b" \n")

        self.assert_e4b_rejected(byte_drift)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO requires POSIX")
    def test_e4b_fifo_is_rejected_without_blocking(self) -> None:
        def mutate(root: Path) -> None:
            target = root / checker.E4B_COPY_PATHS[0]
            target.unlink()
            os.mkfifo(target)

        self.assert_e4b_rejected(mutate)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO requires POSIX")
    def test_fifo_is_rejected(self) -> None:
        def mutate(root: Path) -> None:
            path = root / "bench/__init__.py"
            path.unlink()
            os.mkfifo(path)

        self.assert_rejected(mutate)

    def test_non_copy_destination_is_rejected(self) -> None:
        def mutate(root: Path) -> None:
            path = root / "bench/engine/verify.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not reviewed\n")

        self.assert_rejected(mutate)

    def test_e3a_output_tamper_is_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (root / "bench/engine/metrics.py").write_text(
                "not the reviewed port\n"
            )
        )

    def test_e3b_output_tamper_is_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (root / "bench/engine/report.py").write_text(
                "not the reviewed port\n"
            )
        )

    def test_e3b_output_symlink_is_rejected(self) -> None:
        def mutate(root: Path) -> None:
            path = root / "bench/engine/report.py"
            path.unlink()
            path.symlink_to("metrics.py")

        self.assert_rejected(mutate)

    def test_e3c_outputs_tamper_is_rejected(self) -> None:
        for relative in checker.E3C_SOURCE_PATHS:
            with self.subTest(relative=relative):
                self.assert_rejected(
                    lambda root, relative=relative: (root / relative).write_text(
                        "not the reviewed E3c port\n"
                    )
                )

    def test_e3c_output_symlinks_are_rejected(self) -> None:
        for relative in checker.E3C_SOURCE_PATHS:
            with self.subTest(relative=relative):
                def mutate(root: Path, relative: str = relative) -> None:
                    path = root / relative
                    path.unlink()
                    path.symlink_to("__init__.py")

                self.assert_rejected(mutate)

    def test_e3e_output_tamper_is_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (root / checker.E3E_SOURCE_PATH).write_text(
                "not the reviewed E3e port\n"
            )
        )

    def test_e3e_output_symlink_is_rejected(self) -> None:
        def mutate(root: Path) -> None:
            path = root / checker.E3E_SOURCE_PATH
            path.unlink()
            path.symlink_to("engine/__init__.py")

        self.assert_rejected(mutate)

    def test_missing_protocol_document_is_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (root / "docs/protocol-v0.2.md").unlink()
        )

    def test_notice_tamper_is_rejected(self) -> None:
        self.assert_rejected(
            lambda root: (root / "NOTICE").write_text("changed\n")
        )


class GitObjectVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary.name)
        git(self.repository, "init", "--quiet")
        git(self.repository, "config", "user.name", "E2 Test")
        git(self.repository, "config", "user.email", "e2@example.invalid")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def commit_file(self, payload: bytes, *, mode: int = 0o644) -> tuple[str, dict]:
        path = self.repository / "bench/sample.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(mode)
        git(self.repository, "add", "bench/sample.bin")
        git(self.repository, "commit", "--quiet", "-m", hashlib.sha256(payload).hexdigest())
        commit = git(self.repository, "rev-parse", "HEAD")
        record = {
            "bytes": len(payload),
            "destination": "bench/sample.bin",
            "disposition": "copy",
            "gitBlobSha1": git_blob_sha1(payload),
            "mode": "100755" if mode & 0o111 else "100644",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source": "bench/sample.bin",
        }
        return commit, record

    def test_source_rows_are_verified_from_git_objects(self) -> None:
        commit, record = self.commit_file(b"object bytes\n")
        checker._verify_commit(self.repository, commit)
        payloads = checker._verify_rows_at_commit(
            self.repository, commit, [record]
        )
        self.assertEqual(payloads[record["source"]], b"object bytes\n")

    def test_reviewed_dependency_must_be_an_ancestor(self) -> None:
        ancestor, _ = self.commit_file(b"ancestor\n")
        descendant, _ = self.commit_file(b"descendant\n")
        checker._verify_ancestor(
            self.repository,
            ancestor,
            descendant,
            role="test dependency",
        )
        with self.assertRaisesRegex(
            checker.ImportCheckError,
            "test dependency is not an ancestor",
        ):
            checker._verify_ancestor(
                self.repository,
                descendant,
                ancestor,
                role="test dependency",
            )

    def test_ancestor_check_uses_fail_closed_git_command(self) -> None:
        ancestor = "a" * 40
        descendant = "b" * 40
        with mock.patch.object(checker, "_git", return_value=b"") as git_call:
            checker._verify_ancestor(
                self.repository,
                ancestor,
                descendant,
                role="E4b reviewed dependency",
            )
        git_call.assert_called_once_with(
            self.repository,
            ["merge-base", "--is-ancestor", ancestor, descendant],
        )

    def test_missing_source_and_mode_drift_are_rejected(self) -> None:
        commit, record = self.commit_file(b"object bytes\n")
        missing = dict(record, source="bench/missing.bin")
        with self.assertRaises(checker.ImportCheckError):
            checker._verify_rows_at_commit(self.repository, commit, [missing])
        wrong_mode = dict(record, mode="100755")
        with self.assertRaises(checker.ImportCheckError):
            checker._verify_rows_at_commit(self.repository, commit, [wrong_mode])

    def test_source_symlink_is_rejected(self) -> None:
        path = self.repository / "bench/sample.bin"
        path.parent.mkdir(parents=True)
        path.symlink_to("target")
        git(self.repository, "add", "bench/sample.bin")
        git(self.repository, "commit", "--quiet", "-m", "symlink")
        commit = git(self.repository, "rev-parse", "HEAD")
        payload = b"target"
        record = {
            "bytes": len(payload),
            "destination": "bench/sample.bin",
            "disposition": "copy",
            "gitBlobSha1": git_blob_sha1(payload),
            "mode": "100644",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source": "bench/sample.bin",
        }
        with self.assertRaisesRegex(checker.ImportCheckError, "regular blob"):
            checker._verify_rows_at_commit(self.repository, commit, [record])

    def test_replace_refs_and_hostile_git_environment_are_ignored(self) -> None:
        good_commit, record = self.commit_file(b"good\n")
        bad_commit, _ = self.commit_file(b"bad\n")
        git(self.repository, "replace", good_commit, bad_commit)
        hostile = {
            "GIT_DIR": str(self.repository / "missing-git-dir"),
            "GIT_WORK_TREE": str(self.repository / "missing-work-tree"),
            "GIT_OBJECT_DIRECTORY": str(self.repository / "missing-objects"),
            "GIT_INDEX_FILE": str(self.repository / "missing-index"),
            "GIT_NAMESPACE": "hostile",
        }
        with mock.patch.dict(os.environ, hostile, clear=False):
            payloads = checker._verify_rows_at_commit(
                self.repository, good_commit, [record]
            )
        self.assertEqual(payloads[record["source"]], b"good\n")

    def test_metrics_port_allows_only_the_reviewed_path_replacement(self) -> None:
        installed = (REPOSITORY_ROOT / "bench/engine/metrics.py").read_bytes()
        source = installed.replace(
            checker.METRICS_NEW_DOC_PATH,
            checker.METRICS_OLD_DOC_PATH,
            1,
        )
        manifest_raw = (REPOSITORY_ROOT / checker.INVENTORY_PATH).read_bytes()
        manifest, _ = checker._parse_inventory_bytes(manifest_raw)
        receipt_raw = (REPOSITORY_ROOT / checker.E3A_RECEIPT_PATH).read_bytes()
        transformations = checker._parse_e3a_receipt_bytes(receipt_raw, manifest)
        checker._verify_e3a_transformations_at_source(
            {
                "bench/engine/metrics.py": source,
                "bench/README.md": b"source documentation is separately rewritten\n",
            },
            REPOSITORY_ROOT,
            transformations,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "bench/engine/metrics.py"
            metrics.parent.mkdir(parents=True)
            drifted = installed.replace(
                b"Lower is better",
                b"Lower is bettor",
                1,
            )
            self.assertNotEqual(drifted, installed)
            self.assertEqual(len(drifted), len(installed))
            metrics.write_bytes(drifted)
            with self.assertRaisesRegex(
                checker.ImportCheckError,
                "one reviewed path replacement",
            ):
                checker._verify_e3a_transformations_at_source(
                    {
                        "bench/engine/metrics.py": source,
                        "bench/README.md": b"source documentation\n",
                    },
                    root,
                    transformations,
                )

    def test_report_port_allows_only_the_reviewed_replacement_sequence(self) -> None:
        installed = (REPOSITORY_ROOT / checker.E3B_SOURCE_PATH).read_bytes()
        manifest_raw = (REPOSITORY_ROOT / checker.INVENTORY_PATH).read_bytes()
        manifest, _ = checker._parse_inventory_bytes(manifest_raw)
        receipt_raw = (REPOSITORY_ROOT / checker.E3B_RECEIPT_PATH).read_bytes()
        transformations = checker._parse_e3b_receipt_bytes(receipt_raw, manifest)
        replacements = transformations[0]["transformation"]["replacements"]
        source = installed
        for replacement in reversed(replacements):
            before = replacement["to"].encode("utf-8")
            after = replacement["from"].encode("utf-8")
            self.assertEqual(source.count(before), 1)
            source = source.replace(before, after, 1)

        checker._verify_e3b_transformations_at_source(
            {checker.E3B_SOURCE_PATH: source},
            REPOSITORY_ROOT,
            transformations,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / checker.E3B_SOURCE_PATH
            destination.parent.mkdir(parents=True)
            drifted = installed.replace(b"Result Report", b"Result Retort", 1)
            self.assertNotEqual(drifted, installed)
            self.assertEqual(len(drifted), len(installed))
            destination.write_bytes(drifted)
            with self.assertRaisesRegex(
                checker.ImportCheckError,
                "reviewed replacement sequence",
            ):
                checker._verify_e3b_transformations_at_source(
                    {checker.E3B_SOURCE_PATH: source},
                    root,
                    transformations,
                )

    def test_e3c_ports_round_trip_to_their_reviewed_sources(self) -> None:
        manifest_raw = (REPOSITORY_ROOT / checker.INVENTORY_PATH).read_bytes()
        manifest, _ = checker._parse_inventory_bytes(manifest_raw)
        receipt_raw = (REPOSITORY_ROOT / checker.E3C_RECEIPT_PATH).read_bytes()
        transformations = checker._parse_e3c_receipt_bytes(
            receipt_raw, manifest
        )
        reconstructed = checker._reconstruct_e3c_sources(
            REPOSITORY_ROOT, transformations
        )
        self.assertEqual(set(reconstructed), set(checker.E3C_SOURCE_PATHS))
        checker._verify_e3c_transformations_at_source(
            reconstructed,
            REPOSITORY_ROOT,
            transformations,
        )
        source_drift = dict(reconstructed)
        source_drift[checker.E3C_SOURCE_PATHS[0]] = (
            b"#" + source_drift[checker.E3C_SOURCE_PATHS[0]][1:]
        )
        with self.assertRaisesRegex(
            checker.ImportCheckError, "does not reconstruct pinned source"
        ):
            checker._verify_e3c_transformations_at_source(
                source_drift,
                REPOSITORY_ROOT,
                transformations,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in checker.E3C_SOURCE_PATHS:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPOSITORY_ROOT / relative, destination)
            platform_path = root / checker.E3C_SOURCE_PATHS[0]
            installed = platform_path.read_bytes()
            marker = b'PACKAGE_VERSION = "0.2.0"'
            replacement = b'PACKAGE_VERSION = "0.2.O"'
            self.assertEqual(installed.count(marker), 1)
            drifted = installed.replace(marker, replacement, 1)
            self.assertNotEqual(drifted, installed)
            self.assertEqual(len(drifted), len(installed))
            platform_path.write_bytes(drifted)
            with self.assertRaisesRegex(
                checker.ImportCheckError, "installed output metadata differs"
            ):
                checker._reconstruct_e3c_sources(root, transformations)

    def test_e3d_verifier_round_trips_to_its_reviewed_source(self) -> None:
        manifest_raw = (REPOSITORY_ROOT / checker.INVENTORY_PATH).read_bytes()
        manifest, _ = checker._parse_inventory_bytes(manifest_raw)
        receipt_raw = (REPOSITORY_ROOT / checker.E3D_RECEIPT_PATH).read_bytes()
        transformations = checker._parse_e3d_receipt_bytes(
            receipt_raw, manifest
        )
        reconstructed = checker._reconstruct_e3d_source(
            REPOSITORY_ROOT, transformations
        )
        checker._verify_e3d_transformation_at_source(
            {checker.E3D_SOURCE_PATH: reconstructed},
            REPOSITORY_ROOT,
            transformations,
        )
        with self.assertRaisesRegex(
            checker.ImportCheckError, "does not reconstruct its pinned source"
        ):
            checker._verify_e3d_transformation_at_source(
                {checker.E3D_SOURCE_PATH: b"#" + reconstructed[1:]},
                REPOSITORY_ROOT,
                transformations,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / checker.E3D_SOURCE_PATH
            destination.parent.mkdir(parents=True)
            installed = (REPOSITORY_ROOT / checker.E3D_SOURCE_PATH).read_bytes()
            destination.write_bytes(installed.replace(b"MAX_RESULT_BYTES", b"MAX_RESULU_BYTES", 1))
            with self.assertRaisesRegex(
                checker.ImportCheckError, "installed verifier output metadata differs"
            ):
                checker._reconstruct_e3d_source(root, transformations)

    def test_e3e_runner_round_trips_to_its_reviewed_source(self) -> None:
        manifest_raw = (REPOSITORY_ROOT / checker.INVENTORY_PATH).read_bytes()
        manifest, _ = checker._parse_inventory_bytes(manifest_raw)
        receipt_raw = (REPOSITORY_ROOT / checker.E3E_RECEIPT_PATH).read_bytes()
        transformations = checker._parse_e3e_receipt_bytes(
            receipt_raw, manifest
        )
        reconstructed = checker._reconstruct_e3e_source(
            REPOSITORY_ROOT, transformations
        )
        checker._verify_e3e_transformation_at_source(
            {checker.E3E_SOURCE_PATH: reconstructed},
            REPOSITORY_ROOT,
            transformations,
        )
        with self.assertRaisesRegex(
            checker.ImportCheckError, "does not reconstruct its pinned source"
        ):
            checker._verify_e3e_transformation_at_source(
                {checker.E3E_SOURCE_PATH: b"X" + reconstructed[1:]},
                REPOSITORY_ROOT,
                transformations,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / checker.E3E_SOURCE_PATH
            destination.parent.mkdir(parents=True)
            installed = (REPOSITORY_ROOT / checker.E3E_SOURCE_PATH).read_bytes()
            destination.write_bytes(
                installed.replace(b"_safe_write_text", b"_safe_write_taxt", 1)
            )
            with self.assertRaisesRegex(
                checker.ImportCheckError, "installed runner output metadata differs"
            ):
                checker._reconstruct_e3e_source(root, transformations)


class LiveGateTests(unittest.TestCase):
    def test_repository_gate_passes(self) -> None:
        result = checker.verify_repository(REPOSITORY_ROOT)
        self.assertEqual(result["copyFiles"], 78)
        self.assertEqual(result["e3aFiles"], 2)
        self.assertEqual(result["e3bFiles"], 1)
        self.assertEqual(result["e3cFiles"], 2)
        self.assertEqual(result["e3dFiles"], 1)
        self.assertEqual(result["e3eFiles"], 1)
        self.assertEqual(result["e4aFiles"], 17)
        self.assertEqual(result["e4aCopyFiles"], 15)
        self.assertEqual(result["e4aPortFiles"], 2)
        self.assertEqual(
            result["e4aInstalledTreeSha256"],
            checker.E4A_INSTALLED_TREE_SHA256,
        )
        self.assertEqual(result["e4bFiles"], 10)
        self.assertEqual(result["e4bCopyFiles"], 10)
        self.assertEqual(
            result["e4bInstalledTreeSha256"],
            checker.E4B_INSTALLED_TREE_SHA256,
        )
        self.assertEqual(
            result["e3cPackageTreeSha256"],
            checker.E3C_PACKAGE_TREE_SHA256,
        )
        self.assertFalse(result["sourceVerified"])


if __name__ == "__main__":
    unittest.main()
