from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/generate-portable-string-semantics.py"


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "portable_string_semantics_generator", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load portable string-semantics generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


def _manifest() -> dict[str, Any]:
    return json.loads((ROOT / GENERATOR.MANIFEST_PATH).read_text(encoding="utf-8"))


def _copy_verification_closure(temporary_root: Path) -> None:
    manifest = _manifest()
    paths = [
        GENERATOR.MANIFEST_PATH,
        GENERATOR.SCHEMA_PATH,
        GENERATOR.GENERATOR_PATH,
        GENERATOR.CASEFOLD_INPUT["path"],
        GENERATOR.PROPLIST_INPUT["path"],
    ]
    paths.extend(record["path"] for record in manifest["artifacts"])
    for relative_path in paths:
        source = ROOT / relative_path
        destination = temporary_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_authority_inputs(temporary_root: Path) -> None:
    for relative_path in (
        "bench/__init__.py",
        "bench/engine/__init__.py",
        "bench/engine/schema_validation.py",
        "bench/portable/__init__.py",
        "bench/portable/string_semantics.py",
        GENERATOR.SCHEMA_PATH,
        GENERATOR.GENERATOR_PATH,
        GENERATOR.CASEFOLD_INPUT["path"],
        GENERATOR.PROPLIST_INPUT["path"],
    ):
        source = ROOT / relative_path
        destination = temporary_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _checked_in_outputs() -> dict[str, bytes]:
    manifest = _manifest()
    paths = [GENERATOR.MANIFEST_PATH]
    paths.extend(record["path"] for record in manifest["artifacts"])
    return {path: (ROOT / path).read_bytes() for path in paths}


def _artifact_path(kind: str) -> str:
    return next(
        record["path"] for record in _manifest()["artifacts"] if record["kind"] == kind
    )


class PortableStringSemanticsTests(unittest.TestCase):
    def test_checked_in_tables_exhaust_the_frozen_domain(self) -> None:
        report = GENERATOR.verify(ROOT)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["codePointPositionCount"], 1_114_112)
        self.assertEqual(report["surrogatePositionCount"], 2_048)
        self.assertEqual(report["casefoldChangedMappingCount"], 1_530)
        self.assertEqual(report["pythonWhitespaceCount"], 29)
        self.assertEqual(report["splitCaseCount"], 8)
        self.assertEqual(
            report["casefoldChangedSha256"],
            "b4127130645c3d4a153f99d25f7fa6ffcf28c32c4f6cd5e553790e490170a9cd",
        )
        self.assertEqual(
            report["pythonWhitespaceSha256"],
            "9096d8eea85f36f62cb291c93fbae4baa00cb701d2145ba9f0ca646fe2e619bb",
        )
        self.assertEqual(
            report["combinedFullDomainSha256"],
            "0e1ff78255295b92d0c7f5e13aa5f9dfa9df72f55609d90373ca2f199f2a7e52",
        )
        self.assertEqual(
            report["authorityRegenerationChecked"],
            GENERATOR._runtime_identity() == GENERATOR.REFERENCE_RUNTIME,
        )

    def test_casefold_consumer_uses_full_c_and_f_mappings(self) -> None:
        semantics = GENERATOR.load_string_semantics(ROOT)
        self.assertEqual(semantics.casefold("A Stra\u00dfe \ufb03 \u03c2"), "a strasse ffi \u03c3")
        self.assertEqual(semantics.casefold("I\u0130\u0131"), "ii\u0307\u0131")
        self.assertEqual(semantics.casefold("\ud800x\udfff"), "\ud800x\udfff")
        self.assertEqual(semantics.mapping_for_code_point(0x10FFFF), (0x10FFFF,))
        with self.assertRaises(ValueError):
            semantics.mapping_for_code_point(0x110000)
        with self.assertRaises(TypeError):
            semantics.casefold(b"text")  # type: ignore[arg-type]

        casefold_input = GENERATOR._read_pinned_input(ROOT, GENERATOR.CASEFOLD_INPUT)
        mappings = GENERATOR._parse_casefold_input(casefold_input)
        self.assertEqual(mappings[0x0049], (0x0069,))
        self.assertEqual(mappings[0x0130], (0x0069, 0x0307))
        self.assertNotEqual(mappings[0x0049], (0x0131,))
        self.assertEqual(len(mappings), 1_530)

    def test_whitespace_predicate_and_explicit_split_edges(self) -> None:
        semantics = GENERATOR.load_string_semantics(ROOT)
        self.assertFalse(semantics.is_whitespace(""))
        self.assertTrue(semantics.is_whitespace("\u001c\u00a0\u3000"))
        self.assertFalse(semantics.is_whitespace("\u200b"))
        self.assertFalse(semantics.is_whitespace(" a"))
        with self.assertRaises(TypeError):
            semantics.is_whitespace(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            semantics.split_whitespace(None)  # type: ignore[arg-type]

        cases = {
            "": [],
            "alpha": ["alpha"],
            " \talpha": ["alpha"],
            "alpha\r\n": ["alpha"],
            "alpha  \t\nbeta": ["alpha", "beta"],
            "\u00a0alpha\u001cbeta\u3000gamma\u2029": ["alpha", "beta", "gamma"],
            "\u001c \u00a0\u3000": [],
            "alpha\u200bbeta": ["alpha\u200bbeta"],
        }
        observed_transitions: set[str] = set()
        for source, expected in cases.items():
            with self.subTest(source=source.encode("unicode_escape")):
                tokens, transitions = semantics._split_whitespace_with_trace(source)
                self.assertEqual(tokens, expected)
                self.assertEqual(semantics.split_whitespace(source), expected)
                observed_transitions.update(transitions)
        self.assertEqual(
            sorted(observed_transitions), GENERATOR.REQUIRED_SPLIT_TRANSITIONS
        )

    def test_python_whitespace_differs_from_unicode_property_only_at_c0_controls(self) -> None:
        semantics = GENERATOR.load_string_semantics(ROOT)
        proplist = GENERATOR._read_pinned_input(ROOT, GENERATOR.PROPLIST_INPUT)
        unicode_whitespace = GENERATOR._parse_unicode_whitespace(proplist)
        self.assertEqual(
            semantics.whitespace_code_points - unicode_whitespace,
            {0x1C, 0x1D, 0x1E, 0x1F},
        )
        self.assertEqual(unicode_whitespace - semantics.whitespace_code_points, set())
        self.assertEqual(len(unicode_whitespace), 25)

    def test_non_reference_runtimes_consume_identical_tables(self) -> None:
        program = """
import json
from bench.portable import load_string_semantics
s = load_string_semantics()
print(json.dumps({
    "fold": s.casefold("A Stra\\u00dfe \\ufb03 \\u03c2"),
    "split": s.split_whitespace("\\u001c alpha\\u00a0beta\\u3000"),
    "surrogate": [hex(x) for x in s.mapping_for_code_point(0xD800)],
    "white": sorted(s.whitespace_code_points),
}, ensure_ascii=True, sort_keys=True))
"""
        reports: dict[str, str] = {}
        seen_executables: set[str] = set()
        for command in ("python3.11", "python3.12", "python3.13", "python3.14", "python3"):
            executable = shutil.which(command)
            if executable is None:
                continue
            resolved = str(Path(executable).resolve())
            if resolved in seen_executables:
                continue
            seen_executables.add(resolved)
            result = subprocess.run(
                [executable, "-S", "-c", program],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f"{command}: {result.stderr}")
            reports[command] = result.stdout.strip()
        self.assertTrue(reports)
        self.assertEqual(len(set(reports.values())), 1, msg=reports)

    def test_non_reference_runtime_cannot_enter_write_path(self) -> None:
        fake_identity = dict(GENERATOR.REFERENCE_RUNTIME)
        fake_identity["pythonVersion"] = "3.12.10"
        with self.assertRaisesRegex(
            GENERATOR.GenerationError, "requires exactly CPython 3.13.2"
        ):
            GENERATOR._require_reference_runtime(fake_identity)

        manifest_before = (ROOT / GENERATOR.MANIFEST_PATH).read_bytes()
        candidate = None
        for command in ("python3.11", "python3.12", "python3.14", "python3"):
            executable = shutil.which(command)
            if executable is None:
                continue
            identity = subprocess.run(
                [
                    executable,
                    "-c",
                    (
                        "import platform,sys,unicodedata;"
                        "print(platform.python_implementation(),"
                        "f'{sys.version_info.major}.{sys.version_info.minor}."
                        "{sys.version_info.micro}',unicodedata.unidata_version)"
                    ),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if identity.returncode == 0 and identity.stdout.strip() != "CPython 3.13.2 15.1.0":
                candidate = executable
                break
        if candidate is not None:
            result = subprocess.run(
                [candidate, str(SCRIPT_PATH), "--write"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("requires exactly CPython 3.13.2", result.stderr)
        self.assertEqual((ROOT / GENERATOR.MANIFEST_PATH).read_bytes(), manifest_before)

    def test_portable_package_ast_forbids_ambient_string_semantics(self) -> None:
        violations: list[str] = []
        for path in sorted((ROOT / "bench/portable").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                if node.attr in {"casefold", "isspace"}:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.attr}")
                if node.attr != "split" or not isinstance(node.ctx, ast.Load):
                    continue
                parent_call = next(
                    (
                        candidate
                        for candidate in ast.walk(tree)
                        if isinstance(candidate, ast.Call) and candidate.func is node
                    ),
                    None,
                )
                if parent_call is None:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:split-reference")
                    continue
                explicit_separator = bool(parent_call.args) and not (
                    isinstance(parent_call.args[0], ast.Constant)
                    and parent_call.args[0].value is None
                )
                explicit_separator = explicit_separator or any(
                    keyword.arg == "sep"
                    and not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is None
                    )
                    for keyword in parent_call.keywords
                )
                if not explicit_separator:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}:ambient-split"
                    )
        self.assertEqual(violations, [])

    def test_manifest_schema_and_receipt_are_strict_and_content_addressed(self) -> None:
        manifest_path = ROOT / GENERATOR.MANIFEST_PATH
        schema_path = ROOT / GENERATOR.SCHEMA_PATH
        manifest_bytes = manifest_path.read_bytes()
        schema_bytes = schema_path.read_bytes()
        manifest = GENERATOR._strict_json(manifest_bytes, label=GENERATOR.MANIFEST_PATH)
        schema = GENERATOR._strict_json(schema_bytes, label=GENERATOR.SCHEMA_PATH)
        self.assertEqual(manifest_bytes, GENERATOR._canonical_json(manifest))
        self.assertEqual(schema_bytes, GENERATOR._canonical_json(schema))
        self.assertEqual(len(manifest_bytes), GENERATOR.MANIFEST_BYTE_COUNT)
        self.assertEqual(
            hashlib.sha256(manifest_bytes).hexdigest(), GENERATOR.MANIFEST_SHA256
        )
        self.assertEqual(len(schema_bytes), GENERATOR.SCHEMA_BYTE_COUNT)
        self.assertEqual(hashlib.sha256(schema_bytes).hexdigest(), GENERATOR.SCHEMA_SHA256)
        GENERATOR._validate_manifest_schema(manifest, schema)

        mutated = dict(manifest)
        mutated["unexpected"] = True
        with self.assertRaisesRegex(GENERATOR.GenerationError, "schema validation failed"):
            GENERATOR._validate_manifest_schema(mutated, schema)
        for record in manifest["artifacts"]:
            data = (ROOT / record["path"]).read_bytes()
            self.assertEqual(record["byteCount"], len(data))
            self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
            if record["kind"] in {"casefold-table", "whitespace-table"}:
                self.assertIn(record["sha256"], Path(record["path"]).name)

    def test_strict_json_wraps_resource_limit_parser_errors(self) -> None:
        for label, parser_error in (
            ("deep nesting", RecursionError("maximum recursion depth exceeded")),
            ("giant integer", ValueError("integer string conversion limit")),
        ):
            with (
                self.subTest(label=label),
                mock.patch.object(GENERATOR.json, "loads", side_effect=parser_error),
                self.assertRaisesRegex(
                    GENERATOR.StringSemanticsError, "not strict UTF-8 JSON"
                ),
            ):
                GENERATOR._strict_json(b"[]", label=label)
        with (
            mock.patch.object(
                GENERATOR.json,
                "dumps",
                side_effect=RecursionError("maximum recursion depth exceeded"),
            ),
            self.assertRaisesRegex(
                GENERATOR.StringSemanticsError, "cannot encode canonical strict JSON"
            ),
        ):
            GENERATOR._canonical_json([])

    def test_manifest_cannot_self_authorize_drift_or_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _copy_verification_closure(root)
            manifest_path = root / GENERATOR.MANIFEST_PATH
            original = manifest_path.read_bytes()
            manifest_path.write_bytes(
                original.replace(b"{\n", b'{\n  "artifactCount": 3,\n', 1)
            )
            with self.assertRaisesRegex(
                GENERATOR.GenerationError, "manifest|artifactCount"
            ):
                GENERATOR.verify(root)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "root"
            root.mkdir()
            _copy_verification_closure(root)
            manifest = json.loads(
                (root / GENERATOR.MANIFEST_PATH).read_text(encoding="utf-8")
            )
            casefold_record = next(
                record for record in manifest["artifacts"] if record["kind"] == "casefold-table"
            )
            table_path = root / casefold_record["path"]
            outside = Path(temporary_directory) / "outside-casefold.json"
            shutil.move(table_path, outside)
            os.link(outside, table_path)
            with self.assertRaisesRegex(GENERATOR.GenerationError, "single-link"):
                GENERATOR.verify(root)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "root"
            root.mkdir()
            _copy_verification_closure(root)
            data_directory = root / "bench/portable/data"
            relocated = Path(temporary_directory) / "relocated-data"
            shutil.move(data_directory, relocated)
            data_directory.symlink_to(relocated, target_is_directory=True)
            with self.assertRaisesRegex(
                GENERATOR.GenerationError, "cannot safely (?:read|open)"
            ):
                GENERATOR.verify(root)

    def test_consumer_requires_schema_and_receipt_closure(self) -> None:
        mutations = (
            ("missing-schema", GENERATOR.SCHEMA_PATH, "delete"),
            ("corrupt-schema", GENERATOR.SCHEMA_PATH, "corrupt"),
            ("missing-receipt", _artifact_path("generation-receipt"), "delete"),
            ("corrupt-receipt", _artifact_path("generation-receipt"), "corrupt"),
        )
        for label, relative_path, action in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _copy_verification_closure(root)
                target = root / relative_path
                if action == "delete":
                    target.unlink()
                else:
                    target.write_bytes(target.read_bytes() + b" ")
                with self.assertRaises(GENERATOR.StringSemanticsError):
                    GENERATOR.load_string_semantics(root)

    def test_data_directory_is_an_exact_five_regular_json_file_closure(self) -> None:
        outputs = _checked_in_outputs()
        additions = ("extra.json", "extra.txt", "directory.json", "alias.json", "fifo.json")
        for name in additions:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "root"
                root.mkdir()
                _copy_verification_closure(root)
                data_directory = root / GENERATOR.DATA_DIRECTORY_PATH
                extra = data_directory / name
                if name == "directory.json":
                    extra.mkdir()
                elif name == "alias.json":
                    outside = Path(directory) / "outside.json"
                    outside.write_text("{}\n", encoding="ascii")
                    extra.symlink_to(outside)
                elif name == "fifo.json":
                    os.mkfifo(extra)
                else:
                    extra.write_text("{}\n", encoding="ascii")
                with self.assertRaisesRegex(
                    GENERATOR.StringSemanticsError, "closed-world"
                ):
                    GENERATOR.load_string_semantics(root)
                with (
                    mock.patch.object(GENERATOR, "_require_reference_runtime"),
                    mock.patch.object(
                        GENERATOR, "_build_authority_outputs", return_value=outputs
                    ),
                    self.assertRaises(GENERATOR.GenerationError),
                ):
                    GENERATOR.write_authority(root)

    def test_regular_reader_uses_nofollow_cloexec_and_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_directory = root / "payloads"
            payload_directory.mkdir()
            (payload_directory / "payload.json").write_bytes(b"{}\n")
            original_open = os.open
            observed_flags: list[int] = []

            def observing_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
                if path == "payload.json":
                    observed_flags.append(flags)
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(GENERATOR.os, "open", side_effect=observing_open):
                data = GENERATOR._read_regular_file(
                    root, "payloads/payload.json", max_bytes=16
                )
            self.assertEqual(data, b"{}\n")
            self.assertTrue(observed_flags)
            for flags in observed_flags:
                self.assertTrue(flags & os.O_NOFOLLOW)
                self.assertTrue(flags & os.O_CLOEXEC)
                self.assertTrue(flags & os.O_NONBLOCK)

    def test_fifo_swap_between_stat_and_open_fails_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_directory = root / "payloads"
            payload_directory.mkdir()
            target = payload_directory / "payload.json"
            target.write_bytes(b"{}\n")
            original_open = os.open
            swapped = False

            def swapping_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
                nonlocal swapped
                if path == "payload.json" and not swapped:
                    swapped = True
                    target.unlink()
                    os.mkfifo(target)
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(GENERATOR.os, "open", side_effect=swapping_open):
                with self.assertRaisesRegex(
                    GENERATOR.StringSemanticsError, "changed while it was opened"
                ):
                    GENERATOR._read_regular_file(
                        root, "payloads/payload.json", max_bytes=16
                    )
            self.assertTrue(swapped)

    def test_hardlink_added_while_reading_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_directory = root / "payloads"
            payload_directory.mkdir()
            target = payload_directory / "payload.json"
            target.write_bytes(b"x" * (96 * 1024))
            alias = payload_directory / "alias.json"
            original_read = os.read
            linked = False

            def linking_read(descriptor: int, count: int) -> bytes:
                nonlocal linked
                data = original_read(descriptor, count)
                if data and not linked:
                    os.link(target, alias)
                    linked = True
                return data

            with mock.patch.object(GENERATOR.os, "read", side_effect=linking_read):
                with self.assertRaisesRegex(
                    GENERATOR.StringSemanticsError, "changed while it was read"
                ):
                    GENERATOR._read_regular_file(
                        root, "payloads/payload.json", max_bytes=128 * 1024
                    )
            self.assertTrue(linked)

    def test_authority_writer_rejects_symlink_hardlink_and_fifo_targets(self) -> None:
        outputs = _checked_in_outputs()
        table_relative = _artifact_path("casefold-table")
        for replacement in ("symlink", "hardlink", "fifo"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "root"
                root.mkdir()
                _copy_verification_closure(root)
                target = root / table_relative
                original = target.read_bytes()
                outside = Path(directory) / "outside-table.json"
                outside.write_bytes(original)
                target.unlink()
                if replacement == "symlink":
                    target.symlink_to(outside)
                elif replacement == "hardlink":
                    os.link(outside, target)
                else:
                    os.mkfifo(target)
                with (
                    mock.patch.object(GENERATOR, "_require_reference_runtime"),
                    mock.patch.object(
                        GENERATOR, "_build_authority_outputs", return_value=outputs
                    ),
                    self.assertRaises(GENERATOR.GenerationError),
                ):
                    GENERATOR.write_authority(root)

    def test_authority_writer_rejects_aliased_data_parent(self) -> None:
        outputs = _checked_in_outputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            _copy_verification_closure(root)
            data_directory = root / GENERATOR.DATA_DIRECTORY_PATH
            relocated = Path(directory) / "relocated-data"
            shutil.move(data_directory, relocated)
            data_directory.symlink_to(relocated, target_is_directory=True)
            with (
                mock.patch.object(GENERATOR, "_require_reference_runtime"),
                mock.patch.object(
                    GENERATOR, "_build_authority_outputs", return_value=outputs
                ),
                self.assertRaises(GENERATOR.GenerationError),
            ):
                GENERATOR.write_authority(root)

    def test_reference_authority_generates_from_schema_only_staging(self) -> None:
        if GENERATOR._runtime_identity() != GENERATOR.REFERENCE_RUNTIME:
            self.skipTest("exact CPython 3.13.2 / UCD 15.1.0 is required")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _copy_authority_inputs(root)
            staged_script = root / GENERATOR.GENERATOR_PATH
            first_process = subprocess.run(
                [sys.executable, str(staged_script), "--write"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(first_process.returncode, 0, msg=first_process.stderr)
            first = json.loads(first_process.stdout)
            self.assertEqual(first["status"], "written")
            self.assertTrue(first["consumerLockMatches"])
            check_process = subprocess.run(
                [sys.executable, str(staged_script), "--check"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(check_process.returncode, 0, msg=check_process.stderr)
            self.assertEqual(json.loads(check_process.stdout)["status"], "ok")
            second_process = subprocess.run(
                [sys.executable, str(staged_script), "--write"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(second_process.returncode, 0, msg=second_process.stderr)
            second = json.loads(second_process.stdout)
            self.assertEqual(second["status"], "unchanged")
            self.assertTrue(second["consumerLockMatches"])

    def test_authority_writer_emits_reviewable_candidate_before_lock_update(self) -> None:
        outputs = _checked_in_outputs()
        receipt_path = _artifact_path("generation-receipt")
        receipt = GENERATOR._strict_json(outputs[receipt_path], label=receipt_path)
        receipt["generator"]["sha256"] = "0" * 64
        candidate_receipt = GENERATOR._canonical_json(receipt)
        manifest = GENERATOR._strict_json(
            outputs[GENERATOR.MANIFEST_PATH], label=GENERATOR.MANIFEST_PATH
        )
        receipt_record = next(
            record
            for record in manifest["artifacts"]
            if record["kind"] == "generation-receipt"
        )
        receipt_record["byteCount"] = len(candidate_receipt)
        receipt_record["sha256"] = hashlib.sha256(candidate_receipt).hexdigest()
        candidate_outputs = dict(outputs)
        candidate_outputs[receipt_path] = candidate_receipt
        candidate_outputs[GENERATOR.MANIFEST_PATH] = GENERATOR._canonical_json(manifest)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _copy_authority_inputs(root)
            with (
                mock.patch.object(GENERATOR, "_require_reference_runtime"),
                mock.patch.object(
                    GENERATOR,
                    "_build_authority_outputs",
                    return_value=candidate_outputs,
                ),
            ):
                report = GENERATOR.write_authority(root)
            self.assertEqual(report["status"], "written")
            self.assertFalse(report["consumerLockMatches"])
            self.assertEqual((root / receipt_path).read_bytes(), candidate_receipt)

    def test_generator_is_offline_and_has_explicit_modes(self) -> None:
        help_result = subprocess.run(
            ["python3", str(SCRIPT_PATH), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, msg=help_result.stderr)
        self.assertIn("--check", help_result.stdout)
        self.assertIn("--write", help_result.stdout)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "import requests",
            "import socket",
            "import urllib",
            "pip install",
            "subprocess",
            "urlopen",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
