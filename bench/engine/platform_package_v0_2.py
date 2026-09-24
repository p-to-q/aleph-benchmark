from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any

from .protocol import (
    FROZEN_DATASET_HASH_ALGORITHM,
    FROZEN_DATASET_ID,
    FROZEN_DATASET_ITEM_COUNT,
    FROZEN_DATASET_SHA256,
)
from .scoring_core import PROTOCOL_VERSION, scoring_profile
from .schema_validation import SchemaValidationError, load_schema, validate


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ID = "aleph-bench-v0.2-scorer-conformance"
PACKAGE_VERSION = "0.2.0"
FIXED_CREATED_AT = "2026-09-17T00:00:00Z"

SCORING_SOURCE_PATH = REPO_ROOT / "bench/engine/scoring_core.py"
CONFORMANCE_SOURCE_PATH = REPO_ROOT / "bench/conformance/scorer-v0.2.json"
ITEMS_SOURCE_DIR = REPO_ROOT / "bench/data/v0.2/public/s2"
ITEM_SCHEMA_SOURCE_PATH = REPO_ROOT / "schemas/v0.2/aleph-bench-item.schema.json"
RESULT_SCHEMA_SOURCE_PATH = REPO_ROOT / "schemas/v0.2/aleph-bench-result.schema.json"
PACKAGE_SCHEMA_SOURCE_PATH = (
    REPO_ROOT / "schemas/v0.2/aleph-bench-platform-package.schema.json"
)

MANIFEST_NAME = "package-manifest.json"
CHECKSUMS_NAME = "checksums.sha256"
RUN_CONFORMANCE_PATH = "kaggle/run_conformance.py"
PACKAGE_TREE_HASH_ALGORITHM = (
    "sha256-length-framed-path-type-mode-and-content-v1"
)

_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_FILE_CREATE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_MAX_TRUSTED_ALIAS_EXPANSIONS = 16


class UnsupportedPlatformError(RuntimeError):
    """The host cannot meet the no-follow/no-replace publication contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def _dataset_sha256(snapshots: list[tuple[str, bytes]]) -> str:
    if FROZEN_DATASET_HASH_ALGORITHM != (
        "sha256-length-framed-filename-and-content-v1"
    ):
        raise ValueError(
            "unsupported frozen dataset hash algorithm: "
            f"{FROZEN_DATASET_HASH_ALGORITHM}"
        )
    digest = hashlib.sha256()
    for filename, content in snapshots:
        name = filename.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _load_v0_2_items() -> list[dict[str, Any]]:
    """Load frozen package data without executing any scoring operation."""

    directory_fd: int | None = None
    try:
        directory_fd = os.open(ITEMS_SOURCE_DIR, _DIRECTORY_OPEN_FLAGS)
        with os.scandir(directory_fd) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as exc:
        if directory_fd is not None:
            os.close(directory_fd)
        raise ValueError(f"could not inspect frozen dataset {ITEMS_SOURCE_DIR}: {exc}") from exc
    snapshots: list[tuple[str, bytes]] = []
    try:
        for entry in entries:
            try:
                before = os.stat(
                    entry.name, dir_fd=directory_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise ValueError(
                    f"could not inspect frozen dataset entry {entry.name}: {exc}"
                ) from exc
            if (
                not entry.name.endswith(".json")
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
            ):
                raise ValueError(
                    "frozen dataset directory must contain only single-link regular JSON "
                    f"files; found {entry.name!r}"
                )
            file_fd = os.open(entry.name, _FILE_READ_FLAGS, dir_fd=directory_fd)
            try:
                opened = os.fstat(file_fd)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                ):
                    raise ValueError(
                        f"frozen dataset entry changed while opening: {entry.name}"
                    )
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                after = os.fstat(file_fd)
                opened_snapshot = (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_mode,
                    opened.st_nlink,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                after_snapshot = (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_nlink,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                if opened_snapshot != after_snapshot:
                    raise ValueError(
                        f"frozen dataset entry changed while reading: {entry.name}"
                    )
                content = b"".join(chunks)
                if len(content) != opened.st_size:
                    raise ValueError(
                        f"frozen dataset entry size changed while reading: {entry.name}"
                    )
                snapshots.append((entry.name, content))
            finally:
                os.close(file_fd)
    finally:
        os.close(directory_fd)

    observed_digest = _dataset_sha256(snapshots)
    if (
        len(snapshots) != FROZEN_DATASET_ITEM_COUNT
        or observed_digest != FROZEN_DATASET_SHA256
    ):
        raise ValueError(
            f"dataset does not match frozen {FROZEN_DATASET_ID}: expected "
            f"{FROZEN_DATASET_ITEM_COUNT} items/{FROZEN_DATASET_SHA256}, found "
            f"{len(snapshots)} items/{observed_digest}"
        )

    item_schema = load_schema(ITEM_SCHEMA_SOURCE_PATH)
    items: list[dict[str, Any]] = []
    for filename, content in snapshots:
        try:
            item = json.loads(content.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not load BenchItem {filename}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"BenchItem {filename} must be a JSON object")
        try:
            validate(item, item_schema)
        except SchemaValidationError as exc:
            raise ValueError(
                f"BenchItem {filename} failed v0.2 schema validation: {exc}"
            ) from exc
        items.append(item)
    return items


def _run_conformance_source() -> str:
    return '''#!/usr/bin/env python3
"""Execute Aleph-Bench v0.2 scorer conformance inside this package."""

from __future__ import annotations

import json
import math
import sys
import unicodedata
from pathlib import Path


sys.dont_write_bytecode = True
KAGGLE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = KAGGLE_DIR.parent
sys.path.insert(0, str(KAGGLE_DIR))

import _scoring  # noqa: E402


FIXTURE_PATH = PACKAGE_ROOT / "conformance/scorer-v0.2.json"


def _runtime_fields() -> dict[str, str]:
    return {
        "requiredPythonVersion": _scoring.REQUIRED_PYTHON_VERSION,
        "observedPythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
        "requiredUnicodeDatabaseVersion": _scoring.REQUIRED_UNICODE_DATABASE_VERSION,
        "observedUnicodeDatabaseVersion": unicodedata.unidata_version,
    }


def _same_number(observed: object, expected: object) -> bool:
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return type(observed) is type(expected) and observed == expected
    if isinstance(expected, int):
        return isinstance(observed, int) and not isinstance(observed, bool) and observed == expected
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        return False
    if not math.isfinite(expected) or (
        isinstance(observed, float) and not math.isfinite(observed)
    ):
        return False
    return abs(float(observed) - float(expected)) <= 1e-9


def main() -> int:
    try:
        _scoring.validate_scoring_runtime()
    except Exception as exc:
        report = {
            "status": "runtime_incompatible",
            "protocolVersion": _scoring.PROTOCOL_VERSION,
            "scoringProfile": _scoring.scoring_profile(),
            "fidelityChecks": 0,
            "leakageChecks": 0,
            "unsupportedMetricChecks": 0,
            "errors": [f"runtime: {exc}"],
            **_runtime_fields(),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_fixture_fields = {
        "fixtureVersion",
        "protocolVersion",
        "scoringProfile",
        "leakageThresholds",
        "supportedMetricClasses",
        "unsupportedMetricClasses",
        "fidelityVectors",
        "leakageVectors",
    }
    if set(fixture) != expected_fixture_fields:
        errors.append("fixture fields do not match the scorer conformance contract")

    if fixture.get("protocolVersion") != _scoring.PROTOCOL_VERSION:
        errors.append(
            "fixture protocolVersion does not match vendored scorer: "
            f"{fixture.get('protocolVersion')!r} != {_scoring.PROTOCOL_VERSION!r}"
        )
    if fixture.get("scoringProfile") != _scoring.scoring_profile():
        errors.append("fixture scoringProfile does not match vendored scorer")
    if tuple(fixture.get("supportedMetricClasses", [])) != _scoring.SUPPORTED_METRIC_CLASSES:
        errors.append("fixture supportedMetricClasses do not match vendored scorer")

    fidelity_checks = 0
    for vector in fixture.get("fidelityVectors", []):
        expected_by_metric = vector.get("expected", {})
        if not isinstance(expected_by_metric, dict) or set(expected_by_metric) != set(
            _scoring.SUPPORTED_METRIC_CLASSES
        ):
            errors.append(
                f"fidelity {vector.get('id')}: expected fields do not match supported metrics"
            )
            continue
        for metric_class, expected in expected_by_metric.items():
            fidelity_checks += 1
            try:
                observed = _scoring.fidelity(
                    vector["target"], vector["output"], metric_class
                )
            except Exception as exc:
                errors.append(f"fidelity {vector.get('id')}/{metric_class}: {exc}")
                continue
            if not _same_number(observed, expected):
                errors.append(
                    f"fidelity {vector.get('id')}/{metric_class}: "
                    f"expected {expected!r}, found {observed!r}"
                )

    leakage_checks = 0
    thresholds = fixture.get("leakageThresholds", {})
    expected_leakage_fields = {
        "disqualified",
        "unit",
        "failClosedReason",
        "lcsRatio",
        "targetTrigramRecall",
        "verbatimSpanUnits",
        "skeletonLcsRatio",
        "skeletonTargetTrigramRecall",
        "skeletonVerbatimSpanUnits",
    }
    for vector in fixture.get("leakageVectors", []):
        leakage_checks += 1
        try:
            observed = _scoring.evaluate_leakage(
                vector["prompt"], vector["target"], thresholds
            ).as_dict()
        except Exception as exc:
            errors.append(f"leakage {vector.get('id')}: {exc}")
            continue
        expected = vector.get("expected", {})
        if not isinstance(expected, dict) or set(expected) != expected_leakage_fields:
            errors.append(
                f"leakage {vector.get('id')}: expected fields do not match scorer output"
            )
            continue
        observed_without_thresholds = {
            field: value for field, value in observed.items() if field != "thresholds"
        }
        if set(observed_without_thresholds) != expected_leakage_fields:
            errors.append(
                f"leakage {vector.get('id')}: scorer fields do not match conformance contract"
            )
            continue
        for field, expected_value in expected.items():
            observed_value = observed.get(field)
            if not _same_number(observed_value, expected_value):
                errors.append(
                    f"leakage {vector.get('id')}/{field}: "
                    f"expected {expected_value!r}, found {observed_value!r}"
                )

    unsupported_checks = 0
    for metric_class in fixture.get("unsupportedMetricClasses", []):
        unsupported_checks += 1
        try:
            _scoring.fidelity("target", "output", metric_class)
        except ValueError:
            pass
        except Exception as exc:
            errors.append(
                f"unsupported metric {metric_class!r} raised {type(exc).__name__}, "
                "expected ValueError"
            )
        else:
            errors.append(f"unsupported metric {metric_class!r} did not fail closed")

    report = {
        "status": "failed" if errors else "ok",
        "protocolVersion": _scoring.PROTOCOL_VERSION,
        "scoringProfile": _scoring.scoring_profile(),
        "fidelityChecks": fidelity_checks,
        "leakageChecks": leakage_checks,
        "unsupportedMetricChecks": unsupported_checks,
        "errors": errors,
        **_runtime_fields(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _readme_source() -> str:
    return """# Aleph-Bench v0.2 scorer conformance package

This is a deterministic, self-contained contract package for the Aleph-Bench
v0.2 Unicode scorer core. It is integration and conformance staging for a
future Kaggle evaluator; it is **not** itself a complete Kaggle evaluator,
hosted model evidence, or a leaderboard. It does not provide submission I/O,
model execution, AURC/ECL aggregation, or leaderboard hosting.

The file `kaggle/_scoring.py` is copied byte-for-byte from the repository's
canonical `bench/engine/scoring_core.py`. Do not edit the vendored copy. The
shared conformance fixture, 30 public S2 items, and v0.2 item/result schemas are
included so a platform integration can test the exact released contract.

The scorer fails closed unless it runs on Python 3.13 with Unicode database
15.1.0. Run the complete fidelity, leakage, unsupported-metric, and runtime
checks with:

```bash
python3 kaggle/run_conformance.py
```

Verify every declared digest and the package's closed-world file set from the
Aleph repository with `check_v0_2_package(...)`. `package-manifest.json` and
`checksums.sha256` describe all other files in this directory.
"""


def _encoding_for(path: str) -> str:
    if path.endswith(".jsonl"):
        return "application/x-ndjson"
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".py"):
        return "text/x-python"
    if path.endswith(".md"):
        return "text/markdown"
    return "application/octet-stream"


def _role_for(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "docs"


def _artifact_rows(artifact_bytes: dict[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "role": _role_for(path),
            "path": path,
            "sha256": _sha256(artifact_bytes[path]),
            "bytes": len(artifact_bytes[path]),
            "encodingFormat": _encoding_for(path),
        }
        for path in sorted(artifact_bytes)
    ]


def _checksums_bytes(artifacts: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(f"{artifact['sha256']}  {artifact['path']}" for artifact in artifacts)
        + "\n"
    ).encode("utf-8")


def _expected_package() -> tuple[dict[str, Any], dict[str, bytes], bytes]:
    artifact_bytes = {
        "README.md": _readme_source().encode("utf-8"),
        "conformance/scorer-v0.2.json": CONFORMANCE_SOURCE_PATH.read_bytes(),
        "data/public_s2_items.jsonl": _jsonl_bytes(_load_v0_2_items()),
        "kaggle/_scoring.py": SCORING_SOURCE_PATH.read_bytes(),
        RUN_CONFORMANCE_PATH: _run_conformance_source().encode("utf-8"),
        "schemas/aleph-bench-item.schema.json": ITEM_SCHEMA_SOURCE_PATH.read_bytes(),
        "schemas/aleph-bench-platform-package.schema.json": (
            PACKAGE_SCHEMA_SOURCE_PATH.read_bytes()
        ),
        "schemas/aleph-bench-result.schema.json": RESULT_SCHEMA_SOURCE_PATH.read_bytes(),
    }
    artifacts = _artifact_rows(artifact_bytes)
    manifest = {
        "protocolVersion": PROTOCOL_VERSION,
        "packageVersion": PACKAGE_VERSION,
        "id": PACKAGE_ID,
        "createdAt": FIXED_CREATED_AT,
        "datasetId": FROZEN_DATASET_ID,
        "datasetItemCount": FROZEN_DATASET_ITEM_COUNT,
        "datasetSha256": FROZEN_DATASET_SHA256,
        "datasetHashAlgorithm": FROZEN_DATASET_HASH_ALGORITHM,
        "packageKind": "scorer_conformance",
        "evidenceMode": "none",
        "targetPlatforms": ["kaggle_community_benchmark"],
        "scoringProfile": scoring_profile(),
        "artifacts": artifacts,
        "validationCommands": ["python3 kaggle/run_conformance.py"],
        "notes": [
            "The vendored scorer is byte-identical to bench/engine/scoring_core.py.",
            "This scorer-core package is integration and conformance staging, not a complete Kaggle evaluator.",
            "Submission I/O, model execution, AURC/ECL aggregation, and leaderboard hosting are out of scope.",
            "This package is not model evidence or a leaderboard.",
        ],
    }
    return manifest, artifact_bytes, _checksums_bytes(artifacts)


def _safe_artifact_path(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if "\\" in value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        return None
    return value


def _expected_directories(allowed_files: set[str]) -> set[str]:
    directories: set[str] = set()
    for relative in allowed_files:
        parent = Path(relative).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return _identity(first) == _identity(second)


def _trusted_alias(parent_stat: os.stat_result, link_stat: os.stat_result) -> bool:
    return (
        parent_stat.st_uid == 0
        and stat.S_IMODE(parent_stat.st_mode) & 0o022 == 0
        and link_stat.st_uid == 0
    )


def _require_safe_namespace_directory(value: os.stat_result, *, label: str) -> None:
    """Reject namespaces writable by a different effective-UID principal."""

    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"destination ancestor must be a directory: {label}")
    if value.st_uid not in {0, os.geteuid()}:
        raise ValueError(
            f"destination ancestor has an untrusted owner: {label}"
        )
    writable_by_others = stat.S_IMODE(value.st_mode) & 0o022
    if writable_by_others and not value.st_mode & stat.S_ISVTX:
        raise ValueError(
            f"destination ancestor is group/world writable without sticky protection: {label}"
        )


def _absolute_components(path: str) -> list[str]:
    if not os.path.isabs(path):
        raise ValueError(f"path must be absolute: {path}")
    return [part for part in path.split(os.sep) if part]


def _normalize_destination(out_dir: Path) -> tuple[str, str]:
    raw = os.fspath(out_dir)
    if not isinstance(raw, str):
        raise TypeError("v0.2 package destination must be a text path")
    if not raw or "\x00" in raw:
        raise ValueError("v0.2 package destination must have a valid leaf name")
    if raw != os.sep and raw.endswith(os.sep):
        raise ValueError("v0.2 package destination must not have an empty leaf")
    raw_leaf = raw.rsplit(os.sep, 1)[-1]
    if raw_leaf in {"", ".", ".."}:
        raise ValueError("v0.2 package destination leaf must not be empty, '.' or '..'")
    absolute = os.path.abspath(raw)
    if absolute == os.sep:
        raise ValueError("v0.2 package destination must not be the filesystem root")
    components = _absolute_components(absolute)
    if not components:
        raise ValueError("v0.2 package destination must have a leaf name")
    return absolute, components[-1]


def _open_directory_components(
    components: list[str], *, require_safe_namespace: bool = False
) -> int:
    """Open an absolute directory path without following untrusted links.

    Writers additionally enforce a POSIX owner/mode namespace policy. Read-only
    verification keeps the descriptor and inode checks, but does not impose an
    ownership policy on mounted or shared package trees.
    """

    current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS)
    resolved_components: list[str] = []
    pending = list(components)
    alias_expansions = 0
    try:
        if require_safe_namespace:
            _require_safe_namespace_directory(os.fstat(current_fd), label=os.sep)
        while pending:
            component = pending.pop(0)
            try:
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(
                    f"destination ancestor is unavailable: {component!r}: {exc}"
                ) from exc
            if stat.S_ISLNK(before.st_mode):
                parent_stat = os.fstat(current_fd)
                if not _trusted_alias(parent_stat, before):
                    raise ValueError(
                        f"destination ancestor must not be a symlink: {component!r}"
                    )
                alias_expansions += 1
                if alias_expansions > _MAX_TRUSTED_ALIAS_EXPANSIONS:
                    raise ValueError("too many trusted system-alias expansions")
                target = os.readlink(component, dir_fd=current_fd)
                if os.path.isabs(target):
                    expanded = os.path.normpath(target)
                else:
                    expanded = os.path.normpath(
                        os.path.join(os.sep, *resolved_components, target)
                    )
                target_components = _absolute_components(expanded)
                pending = target_components + pending
                os.close(current_fd)
                current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS)
                resolved_components = []
                continue
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    f"destination ancestor must be a directory: {component!r}"
                )
            if require_safe_namespace:
                _require_safe_namespace_directory(before, label=component)
            try:
                next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            except OSError as exc:
                raise ValueError(
                    f"could not open destination ancestor {component!r}: {exc}"
                ) from exc
            after = os.fstat(next_fd)
            if not _same_identity(before, after):
                os.close(next_fd)
                raise ValueError(
                    f"destination ancestor changed while opening: {component!r}"
                )
            os.close(current_fd)
            current_fd = next_fd
            resolved_components.append(component)
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_parent_descriptor(absolute_destination: str) -> tuple[int, str]:
    components = _absolute_components(absolute_destination)
    if not components:
        raise ValueError("v0.2 package destination must not be the filesystem root")
    parent_fd = _open_directory_components(
        components[:-1], require_safe_namespace=True
    )
    leaf = components[-1]
    try:
        try:
            encoded_leaf = os.fsencode(leaf)
        except UnicodeError as exc:
            raise ValueError(
                "v0.2 package destination leaf is not valid in the filesystem encoding"
            ) from exc
        try:
            name_max = os.fpathconf(parent_fd, "PC_NAME_MAX")
        except (AttributeError, OSError, ValueError):
            name_max = 255
        if name_max >= 0 and len(encoded_leaf) > name_max:
            raise ValueError(
                "v0.2 package destination leaf exceeds the parent name limit"
            )
        return parent_fd, leaf
    except Exception:
        os.close(parent_fd)
        raise


def _ensure_destination_absent(parent_fd: int, leaf: str, display_path: str) -> None:
    try:
        os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"could not inspect v0.2 package destination: {exc}") from exc
    raise ValueError(f"v0.2 package destination must not already exist: {display_path}")


def _revalidate_destination_parent(
    absolute_destination: str,
    parent_fd: int,
    destination_name: str,
    *,
    phase: str,
) -> None:
    """Require the lexical destination parent to retain its opened identity."""

    reopened_fd: int | None = None
    try:
        reopened_fd, reopened_name = _open_parent_descriptor(absolute_destination)
        if reopened_name != destination_name or not _same_identity(
            os.fstat(parent_fd), os.fstat(reopened_fd)
        ):
            raise RuntimeError(
                f"destination parent identity changed {phase}; publication refused"
            )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"destination parent could not be revalidated {phase}: {exc}"
        ) from exc
    finally:
        if reopened_fd is not None:
            os.close(reopened_fd)


def _require_staging_identity(
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
    expected_identity: tuple[int, int],
) -> None:
    """Reject a staging-name swap before invoking the native rename primitive."""

    try:
        named = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"staging entry is unavailable before publication: {exc}") from exc
    if (
        not stat.S_ISDIR(named.st_mode)
        or _identity(named) != expected_identity
        or _identity(os.fstat(staging_fd)) != expected_identity
    ):
        raise RuntimeError("staging entry identity changed before publication")


def _atomic_rename_implementation() -> tuple[Any, int]:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        try:
            function = libc.renameat2
        except AttributeError as exc:
            raise UnsupportedPlatformError(
                "atomic directory no-replace publication requires renameat2"
            ) from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        return function, 1  # RENAME_NOREPLACE
    if sys.platform == "darwin":
        try:
            function = libc.renameatx_np
        except AttributeError as exc:
            raise UnsupportedPlatformError(
                "atomic directory no-replace publication requires renameatx_np"
            ) from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        return function, 0x00000004  # RENAME_EXCL
    raise UnsupportedPlatformError(
        "v0.2 package publication supports only Linux renameat2 and Darwin renameatx_np"
    )


def _probe_atomic_rename_implementation() -> None:
    """Prove the no-replace primitive is callable without mutating a namespace."""

    function, flag = _atomic_rename_implementation()
    ctypes.set_errno(0)
    result = function(
        -1,
        b"aleph-capability-probe-source",
        -1,
        b"aleph-capability-probe-destination",
        flag,
    )
    error_number = ctypes.get_errno()
    if result != -1 or error_number != errno.EBADF:
        raise UnsupportedPlatformError(
            "atomic directory no-replace publication failed its non-mutating "
            f"capability probe: result={result}, errno={error_number}"
        )


def _require_platform_capabilities() -> None:
    required_dir_fd = (os.open, os.mkdir, os.stat, os.unlink, os.rmdir, os.readlink)
    if (
        os.name != "posix"
        or not getattr(os, "O_DIRECTORY", 0)
        or not getattr(os, "O_NOFOLLOW", 0)
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
        or os.scandir not in os.supports_fd
    ):
        raise UnsupportedPlatformError(
            "v0.2 package publication requires descriptor-relative no-follow operations"
        )
    _probe_atomic_rename_implementation()


def _publish_directory_noreplace(
    parent_fd: int,
    source_name: str,
    destination_name: str,
    *,
    source_fd: int | None = None,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    if (source_fd is None) != (expected_identity is None):
        raise ValueError(
            "publication identity requires both source_fd and expected_identity"
        )
    if source_fd is not None and expected_identity is not None:
        _require_staging_identity(
            parent_fd, source_fd, source_name, expected_identity
        )
    function, flag = _atomic_rename_implementation()
    ctypes.set_errno(0)
    result = function(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        flag,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "v0.2 package destination was created by a competing writer",
            destination_name,
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _remove_directory_contents(directory_fd: int) -> None:
    """Best-effort recursive removal anchored to retained descriptors.

    The publication namespace is restricted by POSIX owner/mode checks. Within
    that boundary, every name is rechecked immediately before removal and
    directories are never followed through links.
    """

    with os.scandir(directory_fd) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        before = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(before.st_mode):
            child_fd = os.open(
                entry.name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd
            )
            try:
                if not _same_identity(before, os.fstat(child_fd)):
                    raise RuntimeError(
                        f"cleanup entry changed while opening: {entry.name!r}"
                    )
                _remove_directory_contents(child_fd)
                current = os.stat(
                    entry.name, dir_fd=directory_fd, follow_symlinks=False
                )
                if not stat.S_ISDIR(current.st_mode) or not _same_identity(
                    before, current
                ):
                    raise RuntimeError(
                        f"cleanup directory identity changed: {entry.name!r}"
                    )
                os.rmdir(entry.name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
            continue

        current = os.stat(
            entry.name, dir_fd=directory_fd, follow_symlinks=False
        )
        if not _same_identity(before, current):
            raise RuntimeError(f"cleanup entry identity changed: {entry.name!r}")
        os.unlink(entry.name, dir_fd=directory_fd)


def _cleanup_staging_directory(
    parent_fd: int,
    staging_name: str,
    staging_fd: int,
    expected_identity: tuple[int, int],
) -> None:
    """Remove only a still-name-bound staging tree.

    There is no portable unlink-if-inode syscall. The containing namespace is
    therefore restricted by a POSIX owner/mode policy. Processes sharing the
    writer's effective UID, and principals granted mutation through platform
    ACL policy, remain explicit trust boundaries. A missing, renamed, replaced,
    or linked staging name fails closed and is left untouched.
    """

    opened = os.fstat(staging_fd)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != expected_identity:
        raise RuntimeError("retained staging descriptor identity changed")
    named = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(named.st_mode) or _identity(named) != expected_identity:
        raise RuntimeError("staging name no longer identifies the recorded directory")

    os.fchmod(staging_fd, 0o700)
    _remove_directory_contents(staging_fd)

    named = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(named.st_mode) or _identity(named) != expected_identity:
        raise RuntimeError(
            "staging name changed after cleanup; refusing parent-relative removal"
        )
    with os.scandir(staging_fd) as iterator:
        if next(iterator, None) is not None:
            raise RuntimeError("staging directory is not empty after cleanup")
    os.rmdir(staging_name, dir_fd=parent_fd)


def _create_staging_directory(parent_fd: int) -> tuple[str, int, tuple[int, int]]:
    effective_uid = os.geteuid()
    for _ in range(128):
        staging_name = f".aleph-v0.2-{secrets.token_hex(16)}.tmp"
        try:
            os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        expected_identity: tuple[int, int] | None = None
        staging_fd: int | None = None
        try:
            entry_stat = os.stat(
                staging_name, dir_fd=parent_fd, follow_symlinks=False
            )
            expected_identity = _identity(entry_stat)
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise RuntimeError("new staging entry is not a directory")
            if entry_stat.st_uid != effective_uid:
                raise RuntimeError("staging directory is not owned by the effective user")
            staging_fd = os.open(
                staging_name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd
            )
            opened_stat = os.fstat(staging_fd)
            if _identity(opened_stat) != expected_identity:
                raise RuntimeError("staging directory changed while opening")
            if opened_stat.st_uid != effective_uid:
                raise RuntimeError("staging directory is not owned by the effective user")
            os.fchmod(staging_fd, 0o700)
            opened_stat = os.fstat(staging_fd)
            if stat.S_IMODE(opened_stat.st_mode) != 0o700:
                raise RuntimeError("staging directory must have mode 0700")
            return staging_name, staging_fd, expected_identity
        except Exception as exc:
            cleanup_error: Exception | None = None
            if staging_fd is not None and expected_identity is not None:
                try:
                    _cleanup_staging_directory(
                        parent_fd,
                        staging_name,
                        staging_fd,
                        expected_identity,
                    )
                except Exception as cleanup_exc:
                    cleanup_error = cleanup_exc
            else:
                cleanup_error = RuntimeError(
                    "staging identity could not be retained for safe cleanup"
                )
            if staging_fd is not None:
                os.close(staging_fd)
            if cleanup_error is not None:
                raise RuntimeError(
                    f"{exc}; staging cleanup failed: {cleanup_error}"
                ) from exc
            raise
    raise FileExistsError("could not allocate a unique staging directory")


def _open_directory_at(root_fd: int, components: tuple[str, ...], *, create: bool) -> int:
    current_fd = os.dup(root_fd)
    try:
        for component in components:
            if component in {"", ".", ".."} or os.sep in component:
                raise ValueError(f"unsafe package directory component: {component!r}")
            if create:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                except FileExistsError:
                    pass
            before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"package path component is not a directory: {component}")
            next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            after = os.fstat(next_fd)
            if not _same_identity(before, after):
                os.close(next_fd)
                raise ValueError(f"package directory changed while opening: {component}")
            if create:
                os.fchmod(next_fd, 0o755)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _write_file_at(root_fd: int, relative: str, content: bytes, mode: int) -> None:
    safe = _safe_artifact_path(relative)
    if safe is None:
        raise ValueError(f"unsafe package file path: {relative!r}")
    path = Path(relative)
    parent_fd = _open_directory_at(root_fd, tuple(path.parts[:-1]), create=True)
    try:
        file_fd = os.open(path.name, _FILE_CREATE_FLAGS, mode=0o600, dir_fd=parent_fd)
        try:
            view = memoryview(content)
            while view:
                written = os.write(file_fd, view)
                if written <= 0:
                    raise OSError("short write while creating package artifact")
                view = view[written:]
            os.fchmod(file_fd, mode)
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise RuntimeError(f"created package artifact is not private: {relative}")
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _read_file_at(
    root_fd: int, relative: str, *, max_bytes: int | None = None
) -> bytes:
    safe = _safe_artifact_path(relative)
    if safe is None:
        raise ValueError(f"unsafe package file path: {relative!r}")
    path = Path(relative)
    parent_fd = _open_directory_at(root_fd, tuple(path.parts[:-1]), create=False)
    try:
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"package entry must be a single-link regular file: {relative}"
            )
        file_fd = os.open(path.name, _FILE_READ_FLAGS, dir_fd=parent_fd)
        try:
            after = os.fstat(file_fd)
            if (
                not stat.S_ISREG(after.st_mode)
                or after.st_nlink != 1
                or not _same_identity(before, after)
            ):
                raise ValueError(f"package file changed while opening: {relative}")
            if max_bytes is not None and after.st_size > max_bytes:
                raise ValueError(
                    f"package file exceeds its byte limit: {relative} "
                    f"{after.st_size} > {max_bytes}"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                read_size = 1024 * 1024
                if max_bytes is not None:
                    read_size = min(read_size, max_bytes - total + 1)
                chunk = os.read(file_fd, read_size)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise ValueError(
                        f"package file grew beyond its byte limit: {relative}"
                    )
                chunks.append(chunk)
            finished = os.fstat(file_fd)
            opened_snapshot = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            finished_snapshot = (
                finished.st_dev,
                finished.st_ino,
                finished.st_mode,
                finished.st_nlink,
                finished.st_size,
                finished.st_mtime_ns,
                finished.st_ctime_ns,
            )
            if opened_snapshot != finished_snapshot or finished.st_nlink != 1:
                raise ValueError(f"package file changed while reading: {relative}")
            content = b"".join(chunks)
            if len(content) != after.st_size:
                raise ValueError(f"package file size changed while reading: {relative}")
            return content
        finally:
            os.close(file_fd)
    finally:
        os.close(parent_fd)


def _inspect_package_layout_fd(
    root_fd: int, allowed_files: set[str]
) -> tuple[list[str], set[str]]:
    """Inspect through retained descriptors and return trusted regular files."""

    errors: list[str] = []
    regular_files: set[str] = set()
    expected_directories = _expected_directories(allowed_files)
    observed_directories: set[str] = set()
    maximum_entries = len(allowed_files) + len(expected_directories)
    observed_entry_count = 0

    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode):
        return ["package root must be a directory"], regular_files
    if stat.S_IMODE(root_stat.st_mode) != 0o755:
        errors.append(
            "package directory mode mismatch: . expected 0755, "
            f"found {stat.S_IMODE(root_stat.st_mode):04o}"
        )

    def inspect(directory_fd: int, relative_directory: str) -> None:
        nonlocal observed_entry_count
        try:
            entries = []
            with os.scandir(directory_fd) as iterator:
                for entry in iterator:
                    observed_entry_count += 1
                    if observed_entry_count > maximum_entries:
                        errors.append(
                            "package entry count exceeds the closed-world limit"
                        )
                        return
                    entries.append(entry)
            entries.sort(key=lambda entry: entry.name)
        except OSError as exc:
            errors.append(
                f"could not inspect package directory {relative_directory or '.'}: {exc}"
            )
            return
        for entry in entries:
            relative = (
                f"{relative_directory}/{entry.name}"
                if relative_directory
                else entry.name
            )
            try:
                entry_stat = os.stat(
                    entry.name, dir_fd=directory_fd, follow_symlinks=False
                )
            except OSError as exc:
                errors.append(f"could not inspect package entry {relative}: {exc}")
                continue
            mode = entry_stat.st_mode
            if stat.S_ISLNK(mode):
                errors.append(f"unexpected package symlink: {relative}")
            elif stat.S_ISDIR(mode):
                if relative not in expected_directories:
                    errors.append(f"unexpected package directory: {relative}")
                    continue
                observed_directories.add(relative)
                if stat.S_IMODE(mode) != 0o755:
                    errors.append(
                        f"package directory mode mismatch: {relative} expected 0755, "
                        f"found {stat.S_IMODE(mode):04o}"
                    )
                try:
                    child_fd = os.open(
                        entry.name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd
                    )
                except OSError as exc:
                    errors.append(f"could not open package directory {relative}: {exc}")
                    continue
                try:
                    if not _same_identity(entry_stat, os.fstat(child_fd)):
                        errors.append(f"package directory changed while opening: {relative}")
                        continue
                    inspect(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(mode):
                if relative not in allowed_files:
                    errors.append(f"unexpected package file: {relative}")
                    continue
                regular_files.add(relative)
                if entry_stat.st_nlink != 1:
                    errors.append(
                        f"package file link count mismatch: {relative} expected 1, "
                        f"found {entry_stat.st_nlink}"
                    )
                expected_mode = 0o755 if relative == RUN_CONFORMANCE_PATH else 0o644
                if stat.S_IMODE(mode) != expected_mode:
                    errors.append(
                        f"package file mode mismatch: {relative} expected "
                        f"{expected_mode:04o}, found {stat.S_IMODE(mode):04o}"
                    )
            else:
                errors.append(f"unexpected package non-regular entry: {relative}")

    inspect(root_fd, "")
    for relative in sorted(expected_directories - observed_directories):
        errors.append(f"missing package directory: {relative}")
    return errors, regular_files


def _tree_digest(entries: list[tuple[str, str, int, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, entry_type, mode, content in sorted(entries):
        path_bytes = relative.encode("utf-8")
        type_bytes = entry_type.encode("ascii")
        digest.update(len(path_bytes).to_bytes(4, "big"))
        digest.update(path_bytes)
        digest.update(len(type_bytes).to_bytes(1, "big"))
        digest.update(type_bytes)
        digest.update(mode.to_bytes(4, "big"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _collect_tree_entries_fd(
    root_fd: int, file_size_limits: dict[str, int]
) -> list[tuple[str, str, int, bytes]]:
    entries: list[tuple[str, str, int, bytes]] = [
        (".", "directory", stat.S_IMODE(os.fstat(root_fd).st_mode), b"")
    ]
    allowed_directories = _expected_directories(set(file_size_limits))
    maximum_entries = len(file_size_limits) + len(allowed_directories)
    observed_entry_count = 0

    def collect(directory_fd: int, relative_directory: str) -> None:
        nonlocal observed_entry_count
        scanned_entries = []
        with os.scandir(directory_fd) as iterator:
            for entry in iterator:
                observed_entry_count += 1
                if observed_entry_count > maximum_entries:
                    raise ValueError(
                        "package entry count exceeds the closed-world limit while hashing"
                    )
                scanned_entries.append(entry)
        for entry in sorted(scanned_entries, key=lambda value: value.name):
            relative = (
                f"{relative_directory}/{entry.name}"
                if relative_directory
                else entry.name
            )
            entry_stat = os.stat(
                entry.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if stat.S_ISDIR(entry_stat.st_mode):
                if relative not in allowed_directories:
                    raise ValueError(
                        f"unexpected package directory while hashing: {relative}"
                    )
                child_fd = os.open(
                    entry.name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd
                )
                try:
                    if not _same_identity(entry_stat, os.fstat(child_fd)):
                        raise ValueError(
                            f"package directory changed while hashing: {relative}"
                        )
                    entries.append(
                        (
                            relative,
                            "directory",
                            stat.S_IMODE(entry_stat.st_mode),
                            b"",
                        )
                    )
                    collect(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(entry_stat.st_mode):
                if relative not in file_size_limits:
                    raise ValueError(
                        f"unexpected package file while hashing: {relative}"
                    )
                content = _read_file_at(
                    root_fd,
                    relative,
                    max_bytes=file_size_limits[relative],
                )
                entries.append(
                    (relative, "file", stat.S_IMODE(entry_stat.st_mode), content)
                )
            elif stat.S_ISLNK(entry_stat.st_mode):
                raise ValueError(f"cannot hash package symlink: {relative}")
            else:
                raise ValueError(f"cannot hash non-regular package entry: {relative}")

    collect(root_fd, "")
    return entries


def _expected_package_tree_sha256(
    manifest: dict[str, Any], artifact_bytes: dict[str, bytes], checksums: bytes
) -> str:
    all_files = dict(artifact_bytes)
    all_files[CHECKSUMS_NAME] = checksums
    all_files[MANIFEST_NAME] = _json_bytes(manifest)
    directories = _expected_directories(set(all_files))
    entries = [(".", "directory", 0o755, b"")]
    entries.extend((directory, "directory", 0o755, b"") for directory in directories)
    entries.extend(
        (
            relative,
            "file",
            0o755 if relative == RUN_CONFORMANCE_PATH else 0o644,
            content,
        )
        for relative, content in all_files.items()
    )
    return _tree_digest(entries)


def _check_v0_2_package_fd(root_fd: int, manifest_display: str) -> dict[str, Any]:
    errors: list[str] = []
    expected_manifest, expected_bytes, expected_checksums = _expected_package()
    expected_manifest_bytes = _json_bytes(expected_manifest)
    expected_paths = set(expected_bytes)
    allowed_files = expected_paths | {MANIFEST_NAME, CHECKSUMS_NAME}
    expected_file_bytes = dict(expected_bytes)
    expected_file_bytes[MANIFEST_NAME] = expected_manifest_bytes
    expected_file_bytes[CHECKSUMS_NAME] = expected_checksums
    file_size_limits = {
        relative: len(content) for relative, content in expected_file_bytes.items()
    }
    expected_tree_digest = _expected_package_tree_sha256(
        expected_manifest, expected_bytes, expected_checksums
    )
    layout_errors, regular_files = _inspect_package_layout_fd(root_fd, allowed_files)
    errors.extend(layout_errors)

    manifest: dict[str, Any] | None = None
    if MANIFEST_NAME not in regular_files:
        errors.append(f"missing {MANIFEST_NAME}")
    else:
        try:
            manifest_bytes = _read_file_at(
                root_fd,
                MANIFEST_NAME,
                max_bytes=file_size_limits[MANIFEST_NAME],
            )
            if manifest_bytes != expected_manifest_bytes:
                errors.append(f"{MANIFEST_NAME} is not canonical deterministic JSON")
            loaded = json.loads(manifest_bytes.decode("utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
                try:
                    validate(manifest, load_schema(PACKAGE_SCHEMA_SOURCE_PATH))
                except Exception as exc:
                    errors.append(f"{MANIFEST_NAME} schema validation failed: {exc}")
            else:
                errors.append(f"{MANIFEST_NAME} must contain a JSON object")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid {MANIFEST_NAME}: {exc}")

    declared_artifacts: list[dict[str, Any]] = []
    if manifest is not None:
        for field in (
            "protocolVersion",
            "packageVersion",
            "datasetId",
            "datasetItemCount",
            "datasetSha256",
            "datasetHashAlgorithm",
            "scoringProfile",
            "artifacts",
        ):
            if field not in manifest:
                errors.append(f"manifest missing required field: {field}")
        if manifest.get("protocolVersion") != PROTOCOL_VERSION:
            errors.append(
                f"manifest protocolVersion must be {PROTOCOL_VERSION!r}; "
                f"found {manifest.get('protocolVersion')!r}"
            )
        if manifest.get("packageVersion") != PACKAGE_VERSION:
            errors.append(
                f"manifest packageVersion must be {PACKAGE_VERSION!r}; "
                f"found {manifest.get('packageVersion')!r}"
            )
        if manifest.get("scoringProfile") != scoring_profile():
            errors.append("manifest scoringProfile does not match the canonical scorer")
        raw_artifacts = manifest.get("artifacts")
        if isinstance(raw_artifacts, list):
            declared_artifacts = [row for row in raw_artifacts if isinstance(row, dict)]
            if len(declared_artifacts) != len(raw_artifacts):
                errors.append("manifest artifacts must all be objects")
        else:
            errors.append("manifest artifacts must be an array")
        if manifest != expected_manifest:
            errors.append("package manifest does not match current deterministic package contents")

    seen_paths: set[str] = set()
    for artifact in declared_artifacts:
        relative = _safe_artifact_path(artifact.get("path"))
        if relative is None:
            errors.append(f"unsafe or invalid artifact path: {artifact.get('path')!r}")
            continue
        if relative in seen_paths:
            errors.append(f"duplicate artifact path: {relative}")
            continue
        seen_paths.add(relative)
        if relative not in regular_files:
            errors.append(f"missing artifact: {relative}")
            continue
        try:
            data = _read_file_at(
                root_fd,
                relative,
                max_bytes=file_size_limits[relative],
            )
        except (OSError, ValueError) as exc:
            errors.append(f"could not read artifact {relative}: {exc}")
            continue
        if artifact.get("sha256") != _sha256(data):
            errors.append(f"digest mismatch: {relative}")
        if artifact.get("bytes") != len(data):
            errors.append(f"size mismatch: {relative}")

    for relative in sorted(expected_paths - seen_paths):
        errors.append(f"missing artifact declaration: {relative}")
    for relative in sorted(expected_paths):
        if relative not in regular_files:
            if relative not in seen_paths:
                errors.append(f"missing artifact: {relative}")
            continue
        try:
            data = _read_file_at(
                root_fd,
                relative,
                max_bytes=file_size_limits[relative],
            )
        except (OSError, ValueError) as exc:
            errors.append(f"could not read expected artifact {relative}: {exc}")
            continue
        if data != expected_bytes[relative]:
            errors.append(f"artifact content is stale: {relative}")

    if CHECKSUMS_NAME not in regular_files:
        errors.append(f"missing {CHECKSUMS_NAME}")
    else:
        try:
            checksums = _read_file_at(
                root_fd,
                CHECKSUMS_NAME,
                max_bytes=file_size_limits[CHECKSUMS_NAME],
            )
        except (OSError, ValueError) as exc:
            errors.append(f"could not read {CHECKSUMS_NAME}: {exc}")
        else:
            if checksums != expected_checksums:
                errors.append(
                    f"{CHECKSUMS_NAME} does not match the manifest artifact list"
                )

    observed_tree_digest: str | None = None
    if not layout_errors:
        try:
            observed_tree_digest = _tree_digest(
                _collect_tree_entries_fd(root_fd, file_size_limits)
            )
        except (OSError, ValueError) as exc:
            errors.append(f"could not compute package tree digest: {exc}")
    if observed_tree_digest is not None and observed_tree_digest != expected_tree_digest:
        errors.append("package tree digest does not match current deterministic package")

    return {
        "status": "ok" if not errors else "failed",
        "packageManifest": manifest_display,
        "protocolVersion": PROTOCOL_VERSION,
        "packageVersion": PACKAGE_VERSION,
        "artifactCount": len(expected_paths),
        "packageTreeHashAlgorithm": PACKAGE_TREE_HASH_ALGORITHM,
        "packageTreeSha256": observed_tree_digest,
        "expectedPackageTreeSha256": expected_tree_digest,
        "errors": errors,
    }


def package_tree_sha256(package_dir: Path) -> str:
    """Hash the bounded closed-world package tree, including types and modes."""

    manifest, artifact_bytes, checksums = _expected_package()
    expected_file_bytes = dict(artifact_bytes)
    expected_file_bytes[MANIFEST_NAME] = _json_bytes(manifest)
    expected_file_bytes[CHECKSUMS_NAME] = checksums
    file_size_limits = {
        relative: len(content) for relative, content in expected_file_bytes.items()
    }
    absolute = os.path.abspath(os.fspath(package_dir))
    root_fd = _open_directory_components(_absolute_components(absolute))
    try:
        return _tree_digest(_collect_tree_entries_fd(root_fd, file_size_limits))
    finally:
        os.close(root_fd)


def check_v0_2_package(manifest_path: Path) -> dict[str, Any]:
    """Verify deterministic bytes and a closed-world v0.2 package file set."""

    manifest_path = Path(manifest_path)
    manifest_display = str(manifest_path)
    package_dir = os.path.abspath(os.fspath(manifest_path.parent))
    expected_manifest, expected_bytes, expected_checksums = _expected_package()
    expected_tree_digest = _expected_package_tree_sha256(
        expected_manifest, expected_bytes, expected_checksums
    )
    initial_errors: list[str] = []
    if manifest_path.name != MANIFEST_NAME:
        initial_errors.append(
            f"package manifest path must end with {MANIFEST_NAME}, "
            f"found {manifest_path.name}"
        )
    try:
        package_lstat = os.lstat(package_dir)
        if stat.S_ISLNK(package_lstat.st_mode):
            initial_errors.append("package root must not be a symlink")
            raise ValueError("package root must not be a symlink")
        root_fd = _open_directory_components(_absolute_components(package_dir))
    except (OSError, ValueError) as exc:
        if not initial_errors:
            initial_errors.append(f"package root is unavailable: {exc}")
        return {
            "status": "failed",
            "packageManifest": manifest_display,
            "protocolVersion": PROTOCOL_VERSION,
            "packageVersion": PACKAGE_VERSION,
            "artifactCount": len(expected_bytes),
            "packageTreeHashAlgorithm": PACKAGE_TREE_HASH_ALGORITHM,
            "packageTreeSha256": None,
            "expectedPackageTreeSha256": expected_tree_digest,
            "errors": initial_errors,
        }
    try:
        report = _check_v0_2_package_fd(root_fd, manifest_display)
    finally:
        os.close(root_fd)
    if initial_errors:
        report["errors"] = initial_errors + list(report["errors"])
        report["status"] = "failed"
    return report


def write_v0_2_package(out_dir: Path) -> dict[str, Any]:
    """Write the deterministic Aleph-Bench v0.2 scorer conformance package."""

    _require_platform_capabilities()
    absolute_out_dir, _ = _normalize_destination(out_dir)
    parent_fd, destination_name = _open_parent_descriptor(absolute_out_dir)
    staging_name: str | None = None
    staging_fd: int | None = None
    staging_identity: tuple[int, int] | None = None
    rename_completed = False
    try:
        _ensure_destination_absent(parent_fd, destination_name, absolute_out_dir)
        manifest, artifact_bytes, checksums = _expected_package()
        validate(manifest, load_schema(PACKAGE_SCHEMA_SOURCE_PATH))

        staging_name, staging_fd, staging_identity = _create_staging_directory(parent_fd)
        try:
            for relative_path, content in artifact_bytes.items():
                mode = 0o755 if relative_path == RUN_CONFORMANCE_PATH else 0o644
                _write_file_at(staging_fd, relative_path, content, mode)
            _write_file_at(staging_fd, CHECKSUMS_NAME, checksums, 0o644)
            _write_file_at(staging_fd, MANIFEST_NAME, _json_bytes(manifest), 0o644)
            os.fchmod(staging_fd, 0o755)
            os.fsync(staging_fd)

            check = _check_v0_2_package_fd(
                staging_fd, os.path.join(absolute_out_dir, MANIFEST_NAME)
            )
            if check["status"] != "ok":
                raise RuntimeError(
                    "staged v0.2 package failed self-check: "
                    + "; ".join(check["errors"])
                )

            _revalidate_destination_parent(
                absolute_out_dir,
                parent_fd,
                destination_name,
                phase="before publication",
            )
            _publish_directory_noreplace(
                parent_fd,
                staging_name,
                destination_name,
                source_fd=staging_fd,
                expected_identity=staging_identity,
            )
            rename_completed = True
            published_stat = os.stat(
                destination_name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(published_stat.st_mode)
                or _identity(published_stat) != staging_identity
            ):
                raise RuntimeError(
                    "published v0.2 package does not identify the verified staging inode"
                )
            _revalidate_destination_parent(
                absolute_out_dir,
                parent_fd,
                destination_name,
                phase="after publication",
            )
        except Exception as exc:
            if not rename_completed:
                assert staging_name is not None
                assert staging_fd is not None
                assert staging_identity is not None
                try:
                    _cleanup_staging_directory(
                        parent_fd,
                        staging_name,
                        staging_fd,
                        staging_identity,
                    )
                except Exception as cleanup_exc:
                    try:
                        os.fchmod(staging_fd, 0o700)
                    except OSError:
                        pass
                    raise RuntimeError(
                        f"{exc}; staging cleanup failed: {cleanup_exc}"
                    ) from exc
            raise
        return manifest
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)


def format_v0_2_package_check(report: dict[str, Any]) -> str:
    lines = [
        f"status: {report['status']}",
        f"package: {report['packageManifest']}",
        f"protocol version: {report['protocolVersion']}",
        f"package version: {report['packageVersion']}",
        f"artifacts: {report['artifactCount']}",
        f"package tree sha256: {report.get('packageTreeSha256')}",
    ]
    lines.extend(f"error: {error}" for error in report.get("errors", []))
    return "\n".join(lines)
