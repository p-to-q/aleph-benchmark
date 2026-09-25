from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/check-portable-unicode-inputs.py"


def _load_checker() -> Any:
    spec = importlib.util.spec_from_file_location("portable_unicode_checker", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load portable Unicode input checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()
CP311 = next(
    artifact
    for artifact in CHECKER.EXPECTED_ARTIFACTS
    if artifact.get("pythonTag") == "cp311"
)


def _copy_inputs(temporary_root: Path) -> None:
    paths = [CHECKER.MANIFEST_PATH, CHECKER.SCHEMA_PATH]
    paths.extend(record["path"] for record in (CHECKER.WHEEL_LICENSE, CHECKER.UNICODE_LICENSE))
    paths.extend(artifact["repositoryPath"] for artifact in CHECKER.EXPECTED_ARTIFACTS)
    for relative_path in paths:
        source = ROOT / relative_path
        destination = temporary_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _rewrite_cp311(
    mutate: Callable[[str, bytes], tuple[str, bytes]],
    *,
    extra: tuple[str, bytes] | None = None,
    mode_override: tuple[str, int] | None = None,
) -> bytes:
    source = (ROOT / CP311["repositoryPath"]).read_bytes()
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(source)) as original, zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as rewritten:
        for info in original.infolist():
            data = b"" if info.is_dir() else original.read(info)
            name, data = mutate(info.filename, data)
            if name == info.filename:
                output_info = info
            else:
                output_info = zipfile.ZipInfo(name, date_time=info.date_time)
                output_info.compress_type = info.compress_type
                output_info.external_attr = info.external_attr
                output_info.flag_bits = info.flag_bits
            if mode_override is not None and info.filename == mode_override[0]:
                output_info.external_attr = (
                    mode_override[1] << 16
                ) | (output_info.external_attr & 0xFFFF)
            rewritten.writestr(output_info, data)
        if extra is not None:
            extra_info = zipfile.ZipInfo(extra[0])
            extra_info.create_system = 3
            extra_info.external_attr = (stat.S_IFREG | 0o644) << 16
            rewritten.writestr(extra_info, extra[1])
    return output.getvalue()


class PortableUnicodeInputTests(unittest.TestCase):
    def test_checked_in_inputs_and_wheel_internals_are_exact(self) -> None:
        report = CHECKER.verify(ROOT)
        self.assertEqual(
            report,
            {
                "artifactCount": 5,
                "licenseCount": 2,
                "status": "ok",
                "wheelCount": 2,
            },
        )

    def test_manifest_is_strict_and_cannot_self_authorize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _copy_inputs(root)
            manifest_path = root / CHECKER.MANIFEST_PATH
            original = manifest_path.read_bytes()
            manifest_path.write_bytes(
                original.replace(b"{\n", b'{\n  "artifactCount": 5,\n', 1)
            )
            with self.assertRaisesRegex(CHECKER.PortableInputError, "duplicate key"):
                CHECKER.verify(root)

            manifest_path.write_bytes(
                original.replace(b'"artifactCount": 5', b'"artifactCount": NaN', 1)
            )
            with self.assertRaisesRegex(CHECKER.PortableInputError, "non-finite"):
                CHECKER.verify(root)

    def test_schema_rejects_fields_outside_the_closed_contract(self) -> None:
        manifest = CHECKER._expected_manifest()
        manifest["unexpected"] = True
        schema = json.loads((ROOT / CHECKER.SCHEMA_PATH).read_text(encoding="utf-8"))
        with self.assertRaisesRegex(CHECKER.PortableInputError, "schema validation failed"):
            CHECKER._validate_schema(manifest, schema)

    def test_symlinked_ancestor_and_hardlinked_input_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "root"
            root.mkdir()
            _copy_inputs(root)
            ucd = root / "bench/vendor/unicode/15.1.0/ucd"
            relocated = Path(temporary_directory) / "relocated-ucd"
            shutil.move(ucd, relocated)
            ucd.symlink_to(relocated, target_is_directory=True)
            with self.assertRaisesRegex(CHECKER.PortableInputError, "cannot safely read"):
                CHECKER.verify(root)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "root"
            root.mkdir()
            _copy_inputs(root)
            casefold = root / "bench/vendor/unicode/15.1.0/ucd/CaseFolding.txt"
            outside = Path(temporary_directory) / "outside-casefold.txt"
            shutil.move(casefold, outside)
            os.link(outside, casefold)
            with self.assertRaisesRegex(
                CHECKER.PortableInputError,
                "singly linked regular file",
            ):
                CHECKER.verify(root)

    def test_wheel_rejects_duplicate_members(self) -> None:
        metadata_path = f"{CHECKER.DIST_INFO}/METADATA"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            wheel = _rewrite_cp311(lambda name, data: (name, data), extra=(metadata_path, b"x"))
        with self.assertRaisesRegex(CHECKER.PortableInputError, "duplicate members"):
            CHECKER.inspect_wheel_bytes(
                wheel,
                artifact=CP311,
                retained_license=(ROOT / CHECKER.WHEEL_LICENSE_PATH).read_bytes(),
            )

    def test_wheel_rejects_traversal_and_unexpected_members(self) -> None:
        for member, message in (("../escape", "unsafe"), ("extra.txt", "closure differs")):
            with self.subTest(member=member):
                wheel = _rewrite_cp311(
                    lambda name, data: (name, data),
                    extra=(member, b"unexpected"),
                )
                with self.assertRaisesRegex(CHECKER.PortableInputError, message):
                    CHECKER.inspect_wheel_bytes(
                        wheel,
                        artifact=CP311,
                        retained_license=(ROOT / CHECKER.WHEEL_LICENSE_PATH).read_bytes(),
                    )

    def test_wheel_rejects_non_regular_files_and_false_directories(self) -> None:
        cases = (
            (
                CP311["extensionBasename"],
                stat.S_IFIFO | 0o644,
                "not a regular file",
            ),
            (
                "unicodedata2.libs/",
                stat.S_IFREG | 0o755,
                "directory member type is invalid",
            ),
        )
        for member, mode, message in cases:
            with self.subTest(member=member):
                wheel = _rewrite_cp311(
                    lambda name, data: (name, data),
                    mode_override=(member, mode),
                )
                with self.assertRaisesRegex(CHECKER.PortableInputError, message):
                    CHECKER.inspect_wheel_bytes(
                        wheel,
                        artifact=CP311,
                        retained_license=(ROOT / CHECKER.WHEEL_LICENSE_PATH).read_bytes(),
                    )

    def test_wheel_rejects_metadata_tag_extension_and_license_drift(self) -> None:
        metadata_path = f"{CHECKER.DIST_INFO}/METADATA"
        wheel_path = f"{CHECKER.DIST_INFO}/WHEEL"
        license_path = f"{CHECKER.DIST_INFO}/LICENSE"
        mutations = (
            (
                "metadata",
                lambda name, data: (
                    name,
                    data.replace(b"License: Apache License 2.0", b"License: unknown")
                    if name == metadata_path
                    else data,
                ),
                "License metadata",
            ),
            (
                "tag",
                lambda name, data: (
                    name,
                    data.replace(b"cp311-cp311", b"cp310-cp310")
                    if name == wheel_path
                    else data,
                ),
                "tags differ",
            ),
            (
                "extension",
                lambda name, data: (
                    "unicodedata2.cpython-310-x86_64-linux-gnu.so"
                    if name == CP311["extensionBasename"]
                    else name,
                    data,
                ),
                "closure differs",
            ),
            (
                "license",
                lambda name, data: (
                    name,
                    data + b"drift" if name == license_path else data,
                ),
                "embedded Apache-2.0 license differs",
            ),
        )
        for label, mutation, message in mutations:
            with self.subTest(mutation=label):
                wheel = _rewrite_cp311(mutation)
                with self.assertRaisesRegex(CHECKER.PortableInputError, message):
                    CHECKER.inspect_wheel_bytes(
                        wheel,
                        artifact=CP311,
                        retained_license=(ROOT / CHECKER.WHEEL_LICENSE_PATH).read_bytes(),
                    )

    def test_wheel_rejects_record_drift(self) -> None:
        record_path = f"{CHECKER.DIST_INFO}/RECORD"
        wheel = _rewrite_cp311(
            lambda name, data: (
                name,
                data.replace(b"sha256=", b"sha256=x", 1) if name == record_path else data,
            )
        )
        with self.assertRaisesRegex(CHECKER.PortableInputError, "RECORD mismatch"):
            CHECKER.inspect_wheel_bytes(
                wheel,
                artifact=CP311,
                retained_license=(ROOT / CHECKER.WHEEL_LICENSE_PATH).read_bytes(),
            )

    def test_cli_has_no_write_install_or_import_path(self) -> None:
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
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import unicodedata2", source)
        self.assertNotIn("pip install", source)
        self.assertNotIn("extractall", source)


if __name__ == "__main__":
    unittest.main()
