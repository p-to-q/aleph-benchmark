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
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/check-v0-2-scoring-authority.py"


def _load_checker() -> Any:
    spec = importlib.util.spec_from_file_location("v0_2_authority_checker", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load v0.2 authority checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _copy_authority(temporary_root: Path) -> None:
    paths = [CHECKER.LOCK_PATH] + [entry[1] for entry in CHECKER.EXPECTED_FILES]
    for relative_path in paths:
        source = ROOT / relative_path
        destination = temporary_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


class V02ScoringAuthorityLockTests(unittest.TestCase):
    def test_checked_in_authority_is_exact(self) -> None:
        report = CHECKER.verify(ROOT)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["fileCount"], 41)
        self.assertEqual(
            report["sha256"],
            "be5aaa575fcc6314aa4f5518d2314b2fecfa5efe320ab84ec595d3fc586a59f2",
        )
        roles = [entry[0] for entry in CHECKER.EXPECTED_FILES]
        self.assertEqual(roles.count("public-s2-item"), 30)
        self.assertEqual(roles.count("public-schema"), 4)

    def test_manifest_cannot_self_authorize_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _copy_authority(root)
            scoring_path = root / "bench/engine/scoring_core.py"
            scoring_path.write_bytes(scoring_path.read_bytes() + b"# drift\n")

            manifest_path = root / CHECKER.LOCK_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest["files"]:
                if entry["path"] == "bench/engine/scoring_core.py":
                    data = scoring_path.read_bytes()
                    entry["bytes"] = len(data)
                    entry["sha256"] = hashlib.sha256(data).hexdigest()
                    break
            manifest_path.write_bytes(CHECKER._canonical_json(manifest))

            with self.assertRaisesRegex(
                CHECKER.AuthorityError,
                "lock differs from the reviewed explicit closure",
            ):
                CHECKER.verify(root)

    def test_duplicate_and_nonfinite_manifest_values_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _copy_authority(root)
            manifest_path = root / CHECKER.LOCK_PATH
            original = manifest_path.read_bytes()

            manifest_path.write_bytes(
                original.replace(b"{\n", b'{\n  "formatVersion": 1,\n', 1)
            )
            with self.assertRaisesRegex(CHECKER.AuthorityError, "duplicate key"):
                CHECKER.verify(root)

            manifest_path.write_bytes(
                original.replace(b'"fileCount": 41', b'"fileCount": NaN', 1)
            )
            with self.assertRaisesRegex(CHECKER.AuthorityError, "non-finite"):
                CHECKER.verify(root)

    def test_symlinked_ancestor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "root"
            root.mkdir()
            _copy_authority(root)
            engine = root / "bench/engine"
            relocated = Path(temporary_directory) / "relocated-engine"
            shutil.move(engine, relocated)
            engine.symlink_to(relocated, target_is_directory=True)
            with self.assertRaisesRegex(CHECKER.AuthorityError, "cannot safely read"):
                CHECKER.verify(root)

    def test_hardlinked_authority_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "root"
            root.mkdir()
            _copy_authority(root)
            scoring_path = root / "bench/engine/scoring_core.py"
            outside = Path(temporary_directory) / "outside-scoring.py"
            shutil.move(scoring_path, outside)
            os.link(outside, scoring_path)
            with self.assertRaisesRegex(
                CHECKER.AuthorityError,
                "singly linked regular file",
            ):
                CHECKER.verify(root)

    def test_cli_has_no_self_authorizing_write_mode(self) -> None:
        help_result = subprocess.run(
            ["python3", str(SCRIPT_PATH), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, msg=help_result.stderr)
        self.assertNotIn("--write", help_result.stdout)
        write_result = subprocess.run(
            ["python3", str(SCRIPT_PATH), "--write"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(write_result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
