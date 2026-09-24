from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import run
from bench.engine.protocol import validate_scoring_runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPOSITORY_ROOT / "bench/run.py"
LAUNCHER_PATH = REPOSITORY_ROOT / "aleph-bench"
IMPORT_CHECKER_PATH = REPOSITORY_ROOT / "scripts/check-source-v0.2-import.py"
EXPECTED_COMMANDS = {
    "doctor",
    "manifest",
    "run",
    "verify",
    "report",
    "package-v0.2",
    "validate-croissant",
}
REMOVED_COMMANDS = {"audit", "bundle", "package", "kaggle-replay"}
EXPECTED_ENGINE_IMPORTS = {
    "bench.engine.frozen_ladder",
    "bench.engine.manifest",
    "bench.engine.platform_package_v0_2",
    "bench.engine.preflight",
    "bench.engine.report",
    "bench.engine.schema_validation",
    "bench.engine.verify",
}


def command_choices(parser: argparse.ArgumentParser) -> set[str]:
    subparser_actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    if len(subparser_actions) != 1:
        raise AssertionError("runner must expose exactly one subparser action")
    return set(subparser_actions[0].choices)


def cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-B", str(RUNNER_PATH), *arguments],
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def canonical_runtime_available() -> bool:
    try:
        validate_scoring_runtime()
    except RuntimeError:
        return False
    return True


class RunnerParserTests(unittest.TestCase):
    def test_parser_exposes_exact_v0_2_command_set(self) -> None:
        parser = run.build_parser()
        self.assertEqual(command_choices(parser), EXPECTED_COMMANDS)

    def test_removed_legacy_commands_are_argparse_errors(self) -> None:
        for command in sorted(REMOVED_COMMANDS):
            with self.subTest(command=command):
                completed = cli(command)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("invalid choice", completed.stderr)

    def test_isolated_runner_and_repository_launcher_help(self) -> None:
        direct = cli("--help")
        self.assertEqual(direct.returncode, 0, direct.stderr)
        self.assertIn("package-v0.2", direct.stdout)
        launched = subprocess.run(
            [str(LAUNCHER_PATH), "--help"],
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(launched.returncode, 0, launched.stderr)
        self.assertIn("validate-croissant", launched.stdout)

    def test_ordinary_help_keeps_closed_source_tree_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copied_root = Path(tmp) / "standalone"
            shutil.copytree(
                REPOSITORY_ROOT,
                copied_root,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
            )
            environment = os.environ.copy()
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            environment.pop("PYTHONPYCACHEPREFIX", None)

            def execute(*arguments: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    list(arguments),
                    cwd=copied_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )

            before = execute(
                sys.executable,
                "-I",
                "-B",
                str(copied_root / IMPORT_CHECKER_PATH.relative_to(REPOSITORY_ROOT)),
            )
            self.assertEqual(before.returncode, 0, before.stderr)

            direct = execute(sys.executable, "bench/run.py", "--help")
            self.assertEqual(direct.returncode, 0, direct.stderr)
            launched = execute("./aleph-bench", "--help")
            self.assertEqual(launched.returncode, 0, launched.stderr)

            after = execute(
                sys.executable,
                "-I",
                "-B",
                str(copied_root / IMPORT_CHECKER_PATH.relative_to(REPOSITORY_ROOT)),
            )
            self.assertEqual(after.returncode, 0, after.stderr)

    def test_only_reviewed_standalone_engine_modules_are_imported(self) -> None:
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("bench.engine")
        }
        self.assertEqual(imports, EXPECTED_ENGINE_IMPORTS)

    def test_required_arguments_and_v0_2_defaults(self) -> None:
        parser = run.build_parser()
        with self.assertRaises(SystemExit) as verify_exit:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["verify"])
        self.assertEqual(verify_exit.exception.code, 2)
        with self.assertRaises(SystemExit) as report_exit:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["report"])
        self.assertEqual(report_exit.exception.code, 2)

        args = parser.parse_args(
            ["verify", "--result", "result.json", "--manifest", "manifest.json"]
        )
        self.assertEqual(Path(args.data_dir), run.V2_DATA_DIR)
        for command in ("run", "manifest"):
            parsed = parser.parse_args(
                [command, "--model", "mock-frontier", "--out", "out.json"]
            )
            self.assertEqual(Path(parsed.data_dir), run.V2_DATA_DIR)

        source = RUNNER_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "bench/data/public/s2",
            "bench/results/m0-",
            "schemas/aleph-bench-",
        ):
            self.assertNotIn(forbidden, source)

    def test_optional_croissant_dependency_is_lazy_and_concise(self) -> None:
        original_import = __import__

        def reject_mlcroissant(name, *args, **kwargs):
            if name == "mlcroissant":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        stderr = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=reject_mlcroissant):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    run.main(["validate-croissant", "croissant.json"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("pip install mlcroissant", stderr.getvalue())


class StandaloneOutputBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def temporary_entries(self, target: Path) -> list[Path]:
        if not target.parent.exists() or target.parent.is_symlink():
            return []
        return list(target.parent.glob(f".{target.name}.*.tmp"))

    def assert_rejected_target(self, target: Path) -> None:
        with self.assertRaises((OSError, ValueError)):
            run._safe_write_text(target, "replacement\n")
        self.assertEqual(self.temporary_entries(target), [])

    def test_creates_missing_parent_and_mode_0600_file(self) -> None:
        target = self.root / "missing/parents/result.json"
        run._safe_write_text(target, "payload\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "payload\n")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        self.assertEqual(self.temporary_entries(target), [])

    def test_replacement_preserves_existing_mode(self) -> None:
        target = self.root / "result.json"
        target.write_text("old\n", encoding="utf-8")
        target.chmod(0o640)
        old_inode = target.stat().st_ino
        run._safe_write_text(target, "new\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assertNotEqual(target.stat().st_ino, old_inode)

    def test_symlink_hardlink_directory_and_fifo_targets_are_rejected(self) -> None:
        regular = self.root / "regular"
        regular.write_text("original\n", encoding="utf-8")

        symlink = self.root / "symlink"
        symlink.symlink_to(regular.name)
        self.assert_rejected_target(symlink)
        self.assertEqual(regular.read_text(encoding="utf-8"), "original\n")

        hardlink = self.root / "hardlink"
        os.link(regular, hardlink)
        self.assert_rejected_target(hardlink)
        self.assertEqual(regular.read_text(encoding="utf-8"), "original\n")

        directory = self.root / "directory"
        directory.mkdir()
        self.assert_rejected_target(directory)

        if hasattr(os, "mkfifo"):
            fifo = self.root / "fifo"
            os.mkfifo(fifo)
            self.assert_rejected_target(fifo)

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "requires Unix-domain sockets")
    def test_socket_target_is_rejected(self) -> None:
        socket_path = self.root / "socket"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            try:
                listener.bind(str(socket_path))
            except PermissionError:
                self.skipTest("sandbox forbids creating a Unix-domain socket")
            self.assert_rejected_target(socket_path)
        finally:
            listener.close()

    def test_final_parent_symlink_is_rejected(self) -> None:
        real_parent = self.root / "real"
        real_parent.mkdir()
        linked_parent = self.root / "linked"
        linked_parent.symlink_to(real_parent.name, target_is_directory=True)
        target = linked_parent / "result.json"
        self.assert_rejected_target(target)
        self.assertFalse((real_parent / "result.json").exists())

    def test_intermediate_parent_symlink_cannot_create_or_escape(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        destination = self.root / "destination"
        destination.mkdir()
        intermediate = destination / "intermediate"
        intermediate.symlink_to(outside, target_is_directory=True)
        target = intermediate / "missing" / "result.json"

        self.assert_rejected_target(target)
        self.assertFalse((outside / "missing").exists())

    def test_failed_publication_removes_only_owned_temporary_entry(self) -> None:
        target = self.root / "result.json"

        def reject_publication(src, dst, *, src_dir_fd, dst_dir_fd):
            metadata = os.stat(src, dir_fd=src_dir_fd, follow_symlinks=False)
            self.assertTrue(stat.S_ISREG(metadata.st_mode))
            self.assertEqual(metadata.st_nlink, 1)
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            raise OSError("injected")

        with mock.patch.object(run.os, "replace", side_effect=reject_publication):
            with self.assertRaisesRegex(OSError, "injected"):
                run._safe_write_text(target, "payload\n")
        self.assertFalse(target.exists())
        self.assertEqual(self.temporary_entries(target), [])

    def test_cleanup_failure_preserves_error_and_attempts_both_closes(self) -> None:
        target = self.root / "result.json"
        real_close = os.close
        publication_failed = False
        final_close_calls: list[int] = []

        def reject_publication(src, dst, *, src_dir_fd, dst_dir_fd):
            nonlocal publication_failed
            publication_failed = True
            raise OSError("primary publication failure")

        def reject_cleanup(name, *, dir_fd):
            raise OSError("temporary cleanup failure")

        def close_and_fail_first(descriptor: int) -> None:
            real_close(descriptor)
            if publication_failed:
                final_close_calls.append(descriptor)
                if len(final_close_calls) == 1:
                    raise OSError("temporary descriptor close failure")

        with mock.patch.object(
            run.os, "replace", side_effect=reject_publication
        ), mock.patch.object(
            run.os, "unlink", side_effect=reject_cleanup
        ), mock.patch.object(
            run.os, "close", side_effect=close_and_fail_first
        ):
            with self.assertRaisesRegex(OSError, "primary publication failure"):
                run._safe_write_text(target, "payload\n")

        self.assertEqual(len(final_close_calls), 2)
        self.assertNotEqual(final_close_calls[0], final_close_calls[1])
        self.assertFalse(target.exists())

    def test_success_fsyncs_content_and_parent_directory(self) -> None:
        target = self.root / "result.json"
        real_fsync = os.fsync
        fsync_types: list[str] = []

        def record_fsync(descriptor: int) -> None:
            metadata = os.fstat(descriptor)
            fsync_types.append(
                "directory" if stat.S_ISDIR(metadata.st_mode) else "file"
            )
            real_fsync(descriptor)

        with mock.patch.object(run.os, "fsync", side_effect=record_fsync):
            run._safe_write_text(target, "payload\n")
        self.assertEqual(fsync_types, ["file", "directory"])

    def test_failed_publication_does_not_unlink_replaced_temporary_entry(self) -> None:
        target = self.root / "result.json"

        def replace_temporary(src, dst, *, src_dir_fd, dst_dir_fd):
            moved = f"{src}.moved"
            os.rename(src, moved, src_dir_fd=src_dir_fd, dst_dir_fd=src_dir_fd)
            descriptor = os.open(
                src,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                os.write(descriptor, b"foreign\n")
            finally:
                os.close(descriptor)
            raise OSError("injected replacement race")

        with mock.patch.object(run.os, "replace", side_effect=replace_temporary):
            with self.assertRaisesRegex(OSError, "replacement race"):
                run._safe_write_text(target, "owned\n")
        visible = self.temporary_entries(target)
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0].read_bytes(), b"foreign\n")
        moved = list(self.root.glob(f".{target.name}.*.tmp.moved"))
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0].read_bytes(), b"owned\n")

    def test_target_change_before_publication_is_not_overwritten(self) -> None:
        target = self.root / "result.json"
        real_fsync = os.fsync
        calls = 0

        def inject_target(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            real_fsync(descriptor)
            if calls == 1:
                target.write_text("racer\n", encoding="utf-8")

        with mock.patch.object(run.os, "fsync", side_effect=inject_target):
            with self.assertRaisesRegex(ValueError, "changed before publication"):
                run._safe_write_text(target, "owned\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "racer\n")
        self.assertEqual(self.temporary_entries(target), [])

    def test_in_place_target_change_before_publication_is_not_overwritten(self) -> None:
        target = self.root / "result.json"
        target.write_text("existing\n", encoding="utf-8")
        real_fsync = os.fsync
        calls = 0

        def mutate_target(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            real_fsync(descriptor)
            if calls == 1:
                target.write_text("racer changed the existing file\n", encoding="utf-8")

        with mock.patch.object(run.os, "fsync", side_effect=mutate_target):
            with self.assertRaisesRegex(ValueError, "changed before publication"):
                run._safe_write_text(target, "owned\n")
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "racer changed the existing file\n",
        )
        self.assertEqual(self.temporary_entries(target), [])


class RunnerFailureBoundaryTests(unittest.TestCase):
    def test_package_argument_failures_return_exit_2(self) -> None:
        for arguments in (
            ("package-v0.2",),
            ("package-v0.2", "--out-dir", "a", "--check", "b"),
        ):
            with self.subTest(arguments=arguments):
                completed = cli(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("requires exactly one", completed.stderr)

    def test_invalid_report_does_not_create_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "invalid.json"
            destination = root / "report.md"
            source.write_text("{}\n", encoding="utf-8")
            completed = cli(
                "report",
                "--result",
                str(source),
                "--out",
                str(destination),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(destination.exists())

    @unittest.skipIf(canonical_runtime_available(), "requires a non-canonical runtime")
    def test_noncanonical_manifest_and_run_fail_before_output(self) -> None:
        for command in ("manifest", "run"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp) / f"{command}.json"
                completed = cli(
                    command,
                    "--model",
                    "mock-frontier",
                    "--out",
                    str(destination),
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("Python 3.13", completed.stderr)
                self.assertFalse(destination.exists())

    @unittest.skipIf(canonical_runtime_available(), "requires a non-canonical runtime")
    def test_noncanonical_doctor_is_structurally_blocked(self) -> None:
        completed = cli("doctor", "--model", "mock-frontier", "--json")
        self.assertEqual(completed.returncode, 2)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("Python 3.13" in error for error in report["errors"]))


class RunnerRuntimeNeutralTests(unittest.TestCase):
    def test_package_command_build_and_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_path = Path(tmp) / "package"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    run.main(["package-v0.2", "--out-dir", str(package_path)]),
                    0,
                )
                self.assertEqual(
                    run.main(
                        [
                            "package-v0.2",
                            "--check",
                            str(package_path / "package-manifest.json"),
                        ]
                    ),
                    0,
                )


@unittest.skipUnless(canonical_runtime_available(), "requires Python 3.13 / UCD 15.1")
class CanonicalRuntimeLifecycleTests(unittest.TestCase):
    def test_no_network_mock_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            result_path = root / "result.json"
            report_path = root / "report.md"
            package_path = root / "package"

            command_stdout = io.StringIO()
            with contextlib.redirect_stdout(command_stdout):
                with mock.patch(
                    "urllib.request.OpenerDirector.open",
                    side_effect=AssertionError(
                        "network is forbidden in mock lifecycle"
                    ),
                ), mock.patch(
                    "socket.create_connection",
                    side_effect=AssertionError(
                        "network is forbidden in mock lifecycle"
                    ),
                ):
                    self.assertEqual(
                        run.main(["doctor", "--model", "mock-frontier", "--json"]),
                        0,
                    )
                    self.assertEqual(
                        run.main(
                            [
                                "manifest",
                                "--model",
                                "mock-frontier",
                                "--out",
                                str(manifest_path),
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(
                        run.main(
                            [
                                "run",
                                "--model",
                                "mock-frontier",
                                "--seed",
                                "0",
                                "--out",
                                str(result_path),
                            ]
                        ),
                        0,
                    )
                    verification_stdout = io.StringIO()
                    with contextlib.redirect_stdout(verification_stdout):
                        self.assertEqual(
                            run.main(
                                [
                                    "verify",
                                    "--result",
                                    str(result_path),
                                    "--manifest",
                                    str(manifest_path),
                                    "--json",
                                ]
                            ),
                            0,
                        )
                    self.assertEqual(
                        run.main(
                            [
                                "report",
                                "--result",
                                str(result_path),
                                "--out",
                                str(report_path),
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(
                        run.main(
                            ["package-v0.2", "--out-dir", str(package_path)]
                        ),
                        0,
                    )
                    self.assertEqual(
                        run.main(
                            [
                                "package-v0.2",
                                "--check",
                                str(package_path / "package-manifest.json"),
                            ]
                        ),
                        0,
                    )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            verification = json.loads(verification_stdout.getvalue())
            self.assertEqual(manifest["protocolVersion"], "0.2.0")
            self.assertEqual(result["protocolVersion"], "0.2.0")
            self.assertEqual(verification["status"], "ok")
            for field in ("datasetSha256", "manifestSha256", "resultSha256"):
                self.assertRegex(verification[field], r"\A[0-9a-f]{64}\Z")
            rendered = report_path.read_text(encoding="utf-8")
            self.assertIn("mock", rendered.lower())
            self.assertIn("not a model leaderboard", rendered)


if __name__ == "__main__":
    unittest.main()
