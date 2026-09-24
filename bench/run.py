#!/usr/bin/env python3
from __future__ import annotations

import sys

# The migration gate treats the repository source tree as closed.  Set this
# before importing any project module so ordinary repository-root CLI use does
# not create ``bench/**/__pycache__`` entries that invalidate that gate.
sys.dont_write_bytecode = True

import argparse
import json
import os
import secrets
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_DATA_DIR = REPO_ROOT / "bench/data/v0.2/public/s2"
V2_SCHEMA_DIR = REPO_ROOT / "schemas/v0.2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.engine.frozen_ladder import run_benchmark  # noqa: E402
from bench.engine.manifest import build_manifest  # noqa: E402
from bench.engine.platform_package_v0_2 import (  # noqa: E402
    check_v0_2_package,
    format_v0_2_package_check,
    write_v0_2_package,
)
from bench.engine.preflight import format_preflight, preflight  # noqa: E402
from bench.engine.report import render_report_file  # noqa: E402
from bench.engine.schema_validation import (  # noqa: E402
    SchemaValidationError,
    load_schema,
    validate,
)
from bench.engine.verify import format_verify_report, verify_artifacts  # noqa: E402


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_CREATE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_MAX_TRUSTED_ALIAS_EXPANSIONS = 16


def parse_models(values: list[str]) -> list[str]:
    models: list[str] = []
    for value in values:
        models.extend(part.strip() for part in value.split(",") if part.strip())
    return models


def _target_snapshot(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _require_safe_output_directory(metadata: os.stat_result, *, label: str) -> None:
    """Reject a parent namespace writable by a different principal."""

    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"output ancestor must be a directory: {label}")
    if metadata.st_uid not in {0, os.geteuid()}:
        raise ValueError(f"output ancestor has an untrusted owner: {label}")
    writable_by_others = stat.S_IMODE(metadata.st_mode) & 0o022
    if writable_by_others and not metadata.st_mode & stat.S_ISVTX:
        raise ValueError(
            "output ancestor is group/world writable without sticky protection: "
            f"{label}"
        )


def _absolute_components(path: str) -> list[str]:
    if not os.path.isabs(path):
        raise ValueError(f"output path must be absolute: {path}")
    return [component for component in path.split(os.sep) if component]


def _trusted_top_level_alias(
    parent: os.stat_result,
    link: os.stat_result,
    resolved_components: list[str],
) -> bool:
    """Allow only root-managed top-level aliases such as macOS ``/tmp``."""

    return (
        not resolved_components
        and parent.st_uid == 0
        and stat.S_IMODE(parent.st_mode) & 0o022 == 0
        and link.st_uid == 0
    )


def _open_output_parent(path: Path, *, create: bool) -> tuple[int, str]:
    """Open an output parent by descriptor without following untrusted links."""

    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError("output must be a text path")
    if not raw or "\x00" in raw:
        raise ValueError("output path must have a valid file name")
    absolute = os.path.abspath(raw)
    components = _absolute_components(absolute)
    if not components:
        raise ValueError("output path must name a file")
    leaf = components[-1]
    if leaf in {"", ".", ".."}:
        raise ValueError("output file name must not be empty, '.' or '..'")

    current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS)
    resolved_components: list[str] = []
    pending = components[:-1]
    alias_expansions = 0
    try:
        _require_safe_output_directory(os.fstat(current_fd), label=os.sep)
        while pending:
            component = pending.pop(0)
            created = False
            try:
                before = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create:
                    raise ValueError(
                        f"output ancestor is unavailable: {component!r}"
                    ) from None
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    created = True
                    os.fsync(current_fd)
                except FileExistsError:
                    # A concurrent creator won the race.  Inspect and open the
                    # visible entry under the same rules as every other level.
                    pass
                before = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(
                    f"output ancestor is unavailable: {component!r}: {exc}"
                ) from exc

            if stat.S_ISLNK(before.st_mode):
                parent_metadata = os.fstat(current_fd)
                if not _trusted_top_level_alias(
                    parent_metadata,
                    before,
                    resolved_components,
                ):
                    raise ValueError(
                        f"output ancestor must not be a symlink: {component!r}"
                    )
                alias_expansions += 1
                if alias_expansions > _MAX_TRUSTED_ALIAS_EXPANSIONS:
                    raise ValueError("too many trusted system-alias expansions")
                alias_target = os.readlink(component, dir_fd=current_fd)
                if os.path.isabs(alias_target):
                    expanded = os.path.normpath(alias_target)
                else:
                    expanded = os.path.normpath(
                        os.path.join(
                            os.sep,
                            *resolved_components,
                            alias_target,
                        )
                    )
                pending = _absolute_components(expanded) + pending
                os.close(current_fd)
                current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS)
                resolved_components = []
                continue

            _require_safe_output_directory(before, label=component)
            try:
                next_fd = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=current_fd,
                )
            except OSError as exc:
                raise ValueError(
                    f"could not safely open output ancestor {component!r}: {exc}"
                ) from exc
            try:
                opened = os.fstat(next_fd)
                if _identity(before) != _identity(opened):
                    raise ValueError(
                        f"output ancestor changed while opening: {component!r}"
                    )
                if created:
                    if opened.st_uid != os.geteuid():
                        raise ValueError(
                            "created output ancestor has an unexpected owner: "
                            f"{component!r}"
                        )
                    os.fchmod(next_fd, 0o700)
                    os.fsync(next_fd)
            except Exception:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
            resolved_components.append(component)

        try:
            encoded_leaf = os.fsencode(leaf)
        except UnicodeError as exc:
            raise ValueError(
                "output file name is not valid in the filesystem encoding"
            ) from exc
        try:
            name_max = os.fpathconf(current_fd, "PC_NAME_MAX")
        except (AttributeError, OSError, ValueError):
            name_max = 255
        if name_max >= 0 and len(encoded_leaf) > name_max:
            raise ValueError("output file name exceeds the parent name limit")
        return current_fd, leaf
    except Exception:
        os.close(current_fd)
        raise


def _revalidate_output_parent(path: Path, parent_fd: int, leaf: str) -> None:
    """Confirm the pathname still identifies the retained parent descriptor."""

    check_fd, check_leaf = _open_output_parent(path, create=False)
    try:
        if check_leaf != leaf or _identity(os.fstat(check_fd)) != _identity(
            os.fstat(parent_fd)
        ):
            raise ValueError(f"output parent changed before publication: {path.parent}")
    finally:
        os.close(check_fd)


def _inspect_output_target(parent_fd: int, name: str, target: Path) -> os.stat_result | None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"output must be a singly linked regular file: {target}")
    return metadata


def _safe_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Publish text atomically without following or deleting untrusted entries.

    The output boundary is intentionally local to this repository runner.  The
    response cache has its own stricter lifecycle in ``ResponseCacheAdapter``.
    """

    target = Path(path).absolute()
    parent_fd, target_name = _open_output_parent(target, create=True)

    temporary_name = f".{target.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"
    temporary_fd: int | None = None
    published = False
    try:
        existing = _inspect_output_target(parent_fd, target_name, target)
        existing_snapshot = (
            _target_snapshot(existing) if existing is not None else None
        )
        temporary_fd = os.open(
            temporary_name,
            _FILE_CREATE_FLAGS,
            0o600,
            dir_fd=parent_fd,
        )
        temporary_metadata = os.fstat(temporary_fd)
        if (
            not stat.S_ISREG(temporary_metadata.st_mode)
            or temporary_metadata.st_nlink != 1
        ):
            raise ValueError(f"temporary output is not a singly linked regular file: {target}")
        os.fchmod(temporary_fd, 0o600)

        payload = content.encode(encoding)
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise OSError("output write made no progress")
            view = view[written:]
        os.fsync(temporary_fd)

        ready = os.fstat(temporary_fd)
        if (
            not stat.S_ISREG(ready.st_mode)
            or ready.st_nlink != 1
            or (ready.st_dev, ready.st_ino)
            != (temporary_metadata.st_dev, temporary_metadata.st_ino)
            or stat.S_IMODE(ready.st_mode) != 0o600
            or ready.st_size != len(payload)
        ):
            raise ValueError(f"temporary output changed before publication: {target}")

        _revalidate_output_parent(target, parent_fd, target_name)
        current = _inspect_output_target(parent_fd, target_name, target)
        current_snapshot = _target_snapshot(current) if current is not None else None
        if current_snapshot != existing_snapshot:
            raise ValueError(f"output changed before publication: {target}")

        os.replace(
            temporary_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        published = True
        if existing is not None:
            # Restore the exact prior permission bits only after the mode-0600
            # temporary entry has atomically replaced the destination.
            os.fchmod(temporary_fd, stat.S_IMODE(existing.st_mode))
            os.fsync(temporary_fd)
        os.fsync(parent_fd)
    finally:
        active_error = sys.exc_info()[1] is not None
        cleanup_error: OSError | None = None
        if not published and temporary_fd is not None:
            try:
                created = os.fstat(temporary_fd)
                try:
                    visible = os.stat(
                        temporary_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    visible = None
                if visible is not None and (
                    visible.st_dev,
                    visible.st_ino,
                ) == (created.st_dev, created.st_ino):
                    os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError as error:
                cleanup_error = error
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except OSError as error:
                if cleanup_error is None:
                    cleanup_error = error
        try:
            os.close(parent_fd)
        except OSError as error:
            if cleanup_error is None:
                cleanup_error = error
        if not active_error and cleanup_error is not None:
            raise cleanup_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aleph-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run a frozen-ladder benchmark")
    run.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model id; repeat or comma-separate",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", required=True)
    run.add_argument("--data-dir", default=str(V2_DATA_DIR))
    run.add_argument(
        "--cache-dir",
        default=None,
        help="Optional response cache directory for resumable adapter calls",
    )
    doctor = subparsers.add_parser(
        "doctor", help="Check dataset, leakage gate, model env, and call budget"
    )
    doctor.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model id; repeat or comma-separate",
    )
    doctor.add_argument("--data-dir", default=str(V2_DATA_DIR))
    doctor.add_argument(
        "--json", action="store_true", help="Emit machine-readable preflight JSON"
    )
    manifest = subparsers.add_parser(
        "manifest", help="Write a no-call manifest of non-leaking benchmark prompts"
    )
    manifest.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model id; repeat or comma-separate",
    )
    manifest.add_argument("--seed", type=int, default=0)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--data-dir", default=str(V2_DATA_DIR))
    verify = subparsers.add_parser(
        "verify", help="Validate v0.2 dataset, result, and manifest artifacts"
    )
    verify.add_argument("--data-dir", default=str(V2_DATA_DIR))
    verify.add_argument("--result", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument(
        "--json", action="store_true", help="Emit machine-readable verification JSON"
    )
    report = subparsers.add_parser(
        "report", help="Render markdown tables from a v0.2 BenchResult"
    )
    report.add_argument("--result", required=True)
    report.add_argument("--out", default=None, help="Optional markdown output path")
    package_v0_2 = subparsers.add_parser(
        "package-v0.2",
        help="Build or check a v0.2 scorer conformance package",
    )
    package_v0_2.add_argument("--out-dir", default=None)
    package_v0_2.add_argument("--check", default=None)
    package_v0_2.add_argument("--json", action="store_true")
    validate_croissant = subparsers.add_parser(
        "validate-croissant",
        help="Validate a Croissant JSON-LD file with mlcroissant (must be installed)",
    )
    validate_croissant.add_argument("path", help="Path to croissant.json")
    validate_croissant.add_argument(
        "--records",
        action="append",
        default=[],
        help="Optional record set ids to stream end-to-end (repeat or comma-separate)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        models = parse_models(args.model)
        report = preflight(data_dir=Path(args.data_dir), models=models)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(format_preflight(report))
        return 0 if report["status"] == "ready" else 2
    if args.command == "manifest":
        out = Path(args.out)
        models = parse_models(args.model)
        manifest = build_manifest(
            data_dir=Path(args.data_dir),
            models=models,
            seed=args.seed,
        )
        schema = load_schema(V2_SCHEMA_DIR / "aleph-bench-manifest.schema.json")
        validate(manifest, schema)
        _safe_write_text(
            out,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        print(f"wrote {out}")
        return 0
    if args.command == "verify":
        report = verify_artifacts(
            data_dir=Path(args.data_dir),
            result_path=Path(args.result),
            manifest_path=Path(args.manifest),
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(format_verify_report(report))
        return 0 if report["status"] == "ok" else 2
    if args.command == "report":
        markdown = render_report_file(Path(args.result))
        if args.out:
            out = Path(args.out)
            _safe_write_text(out, markdown)
            print(f"wrote {out}")
        else:
            print(markdown)
        return 0
    if args.command == "validate-croissant":
        try:
            import mlcroissant as mlc  # type: ignore[import-untyped]
        except ImportError as exc:
            print(
                "error: mlcroissant is not installed. "
                "Install with `pip install mlcroissant` to run this check.",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        path = Path(args.path)
        ds = mlc.Dataset(jsonld=str(path))
        record_ids = (
            parse_models(args.records)
            if args.records
            else [rs.id for rs in ds.metadata.record_sets]
        )
        summary: dict[str, int] = {}
        for record_id in record_ids:
            rows = list(ds.records(record_id))
            summary[record_id] = len(rows)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "path": str(path),
                    "name": ds.metadata.name,
                    "version": getattr(ds.metadata, "version", None),
                    "recordSets": summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "package-v0.2":
        if bool(args.out_dir) == bool(args.check):
            raise ValueError("package-v0.2 requires exactly one of --out-dir or --check")
        if args.check:
            report = check_v0_2_package(Path(args.check))
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(format_v0_2_package_check(report))
            return 0 if report["status"] == "ok" else 2
        out_dir = Path(args.out_dir)
        manifest = write_v0_2_package(out_dir)
        if args.json:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            print(f"wrote {out_dir / 'package-manifest.json'}")
        return 0
    if args.command != "run":
        raise AssertionError(args.command)
    out = Path(args.out)
    models = parse_models(args.model)
    result = run_benchmark(
        data_dir=Path(args.data_dir),
        models=models,
        seed=args.seed,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    schema = load_schema(V2_SCHEMA_DIR / "aleph-bench-result.schema.json")
    validate(result, schema)
    _safe_write_text(out, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        SchemaValidationError,
        UnicodeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
