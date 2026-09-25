#!/usr/bin/env python3
"""Generate and verify frozen Python 3.13 string-semantics tables.

``--write`` is an authority operation and is accepted only by the exact
CPython 3.13.2 / UCD 15.1.0 reference runtime. ``--check`` is read-only and
works on non-reference runtimes: it verifies and consumes the frozen tables
without consulting ambient string semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import stat
import sys
import unicodedata
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.portable.string_semantics import (  # noqa: E402
    CODE_POINT_COUNT,
    DATA_DIRECTORY_PATH,
    EXPECTED_CASEFOLD_CHANGED_SHA256,
    EXPECTED_CHANGED_MAPPING_COUNT,
    EXPECTED_COMBINED_SHA256,
    EXPECTED_PYTHON_WHITESPACE_COUNT,
    EXPECTED_UNICODE_WHITESPACE_COUNT,
    EXPECTED_WHITESPACE_SHA256,
    MANIFEST_BYTE_COUNT,
    MANIFEST_PATH,
    MANIFEST_SHA256,
    SCHEMA_BYTE_COUNT,
    SCHEMA_PATH,
    SCHEMA_SHA256,
    SURROGATE_END,
    SURROGATE_START,
    FrozenStringSemantics,
    StringSemanticsError,
    _DataDirectorySnapshot,
    _canonical_json,
    _open_directory_below_root,
    _read_regular_file,
    _read_regular_member,
    _read_snapshot_artifact,
    _snapshot_data_directory,
    _stat_state,
    _strict_json,
    load_string_semantics,
)


GENERATOR_PATH = "scripts/generate-portable-string-semantics.py"
CASEFOLD_INPUT = {
    "byteCount": 84_870,
    "path": "bench/vendor/unicode/15.1.0/ucd/CaseFolding.txt",
    "sha256": "4e55acfdc32825a22e87670e9056a3bf94ad7c5400065778e9e10f8314372bcf",
    "unicodeVersion": "15.1.0",
}
PROPLIST_INPUT = {
    "byteCount": 136_397,
    "path": "bench/vendor/unicode/15.1.0/ucd/PropList.txt",
    "sha256": "05672956317b6296bc2ec3d6cef1f6452b57ff4f2efc6dc55b0a19277d5fcfd1",
    "unicodeVersion": "15.1.0",
}
RECEIPT_PATH = (
    "bench/portable/data/"
    "python-3.13.2-ucd-15.1-string-semantics-generation-receipt-v1.json"
)
SEMANTICS_ID = "python-3.13.2-ucd-15.1-string-semantics-v1"
REFERENCE_RUNTIME = {
    "cacheTag": "cpython-313",
    "implementation": "CPython",
    "pythonHexVersion": "0x030d02f0",
    "pythonVersion": "3.13.2",
    "unicodeDatabaseVersion": "15.1.0",
}
REQUIRED_SPLIT_TRANSITIONS = [
    "eof:emit",
    "eof:idle",
    "token:continue",
    "token:end",
    "token:start",
    "whitespace:skip",
]
SPLIT_INPUTS = [
    {"id": "empty", "input": ""},
    {"id": "plain", "input": "alpha"},
    {"id": "leading", "input": " \talpha"},
    {"id": "trailing", "input": "alpha\r\n"},
    {"id": "repeated", "input": "alpha  \t\nbeta"},
    {"id": "mixed", "input": "\u00a0alpha\u001cbeta\u3000gamma\u2029"},
    {"id": "all-whitespace", "input": "\u001c \u00a0\u3000"},
    {"id": "non-whitespace-format", "input": "alpha\u200bbeta"},
]


class GenerationError(RuntimeError):
    """Raised when generation or independent verification fails closed."""


def _runtime_identity() -> dict[str, str]:
    return {
        "cacheTag": sys.implementation.cache_tag or "",
        "implementation": platform.python_implementation(),
        "pythonHexVersion": f"0x{sys.hexversion:08x}",
        "pythonVersion": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "unicodeDatabaseVersion": unicodedata.unidata_version,
    }


def _require_reference_runtime(identity: Mapping[str, str] | None = None) -> None:
    observed = dict(_runtime_identity() if identity is None else identity)
    if observed != REFERENCE_RUNTIME:
        raise GenerationError(
            "authority generation requires exactly CPython 3.13.2 / UCD 15.1.0; "
            f"observed {observed!r}"
        )


def _read_pinned_input(root: Path, record: dict[str, Any]) -> bytes:
    data = _read_regular_file(root, record["path"], max_bytes=record["byteCount"])
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if len(data) != record["byteCount"] or actual_sha256 != record["sha256"]:
        raise GenerationError(
            f"pinned Unicode input differs at {record['path']}: expected "
            f"bytes={record['byteCount']} sha256={record['sha256']}, found "
            f"bytes={len(data)} sha256={actual_sha256}"
        )
    return data


def _parse_casefold_input(data: bytes) -> dict[int, tuple[int, ...]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GenerationError(f"CaseFolding.txt is not UTF-8: {error}") from error
    mappings: dict[int, tuple[int, ...]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        body = raw_line.partition("#")[0].strip()
        if not body:
            continue
        fields = [field.strip() for field in body.split(";")]
        if len(fields) != 4 or fields[-1]:
            raise GenerationError(
                f"CaseFolding.txt line {line_number} has {len(fields)} fields"
            )
        raw_code_point, status, raw_mapping, _ = fields
        if status not in {"C", "F", "S", "T"}:
            raise GenerationError(
                f"CaseFolding.txt line {line_number} has unknown status {status!r}"
            )
        try:
            code_point = int(raw_code_point, 16)
            mapping = tuple(int(item, 16) for item in raw_mapping.split())
        except ValueError as error:
            raise GenerationError(
                f"CaseFolding.txt line {line_number} contains invalid hexadecimal data"
            ) from error
        if (
            not 0 <= code_point < CODE_POINT_COUNT
            or not mapping
            or any(not 0 <= item < CODE_POINT_COUNT for item in mapping)
        ):
            raise GenerationError(
                f"CaseFolding.txt line {line_number} contains an invalid code point"
            )
        if status in {"C", "F"}:
            if code_point in mappings:
                raise GenerationError(
                    f"CaseFolding.txt repeats a C/F mapping for U+{code_point:04X}"
                )
            mappings[code_point] = mapping
    if len(mappings) != EXPECTED_CHANGED_MAPPING_COUNT:
        raise GenerationError(
            f"CaseFolding C/F count differs: expected {EXPECTED_CHANGED_MAPPING_COUNT}, "
            f"found {len(mappings)}"
        )
    return mappings


def _parse_unicode_whitespace(data: bytes) -> frozenset[int]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GenerationError(f"PropList.txt is not UTF-8: {error}") from error
    code_points: set[int] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        body = raw_line.partition("#")[0].strip()
        if not body:
            continue
        fields = [field.strip() for field in body.split(";")]
        if len(fields) != 2:
            raise GenerationError(
                f"PropList.txt line {line_number} has {len(fields)} fields"
            )
        raw_range, property_name = fields
        if property_name != "White_Space":
            continue
        endpoints = raw_range.split("..")
        if len(endpoints) not in {1, 2}:
            raise GenerationError(
                f"PropList.txt line {line_number} has an invalid range"
            )
        try:
            start = int(endpoints[0], 16)
            end = int(endpoints[-1], 16)
        except ValueError as error:
            raise GenerationError(
                f"PropList.txt line {line_number} contains invalid hexadecimal data"
            ) from error
        if not 0 <= start <= end < CODE_POINT_COUNT:
            raise GenerationError(
                f"PropList.txt line {line_number} contains an invalid range"
            )
        for code_point in range(start, end + 1):
            if code_point in code_points:
                raise GenerationError(
                    f"PropList.txt repeats White_Space U+{code_point:04X}"
                )
            code_points.add(code_point)
    if len(code_points) != EXPECTED_UNICODE_WHITESPACE_COUNT:
        raise GenerationError(
            f"Unicode White_Space count differs: expected "
            f"{EXPECTED_UNICODE_WHITESPACE_COUNT}, found {len(code_points)}"
        )
    return frozenset(code_points)


def _framed_digests(
    mappings: Mapping[int, tuple[int, ...]],
    whitespace: frozenset[int],
) -> dict[str, str]:
    changed_digest = hashlib.sha256()
    whitespace_digest = hashlib.sha256()
    combined_digest = hashlib.sha256()
    for code_point in range(CODE_POINT_COUNT):
        mapped = mappings.get(code_point, (code_point,))
        mapping_frame = len(mapped).to_bytes(4, "big") + b"".join(
            item.to_bytes(4, "big") for item in mapped
        )
        code_point_frame = code_point.to_bytes(4, "big")
        if code_point in mappings:
            changed_digest.update(code_point_frame)
            changed_digest.update(mapping_frame)
        if code_point in whitespace:
            whitespace_digest.update(code_point_frame)
        combined_digest.update(code_point_frame)
        combined_digest.update(bytes((code_point in whitespace,)))
        combined_digest.update(mapping_frame)
    result = {
        "casefoldChangedSha256": changed_digest.hexdigest(),
        "combinedFullDomainSha256": combined_digest.hexdigest(),
        "pythonWhitespaceSha256": whitespace_digest.hexdigest(),
    }
    expected = {
        "casefoldChangedSha256": EXPECTED_CASEFOLD_CHANGED_SHA256,
        "combinedFullDomainSha256": EXPECTED_COMBINED_SHA256,
        "pythonWhitespaceSha256": EXPECTED_WHITESPACE_SHA256,
    }
    if result != expected:
        raise GenerationError(
            f"framed string-semantics digests differ: expected {expected!r}, "
            f"found {result!r}"
        )
    return result


def _reference_semantics(
    casefold_input: bytes,
    proplist_input: bytes,
) -> tuple[dict[int, tuple[int, ...]], frozenset[int], frozenset[int]]:
    _require_reference_runtime()
    official_casefold = _parse_casefold_input(casefold_input)
    unicode_whitespace = _parse_unicode_whitespace(proplist_input)
    ambient_casefold: dict[int, tuple[int, ...]] = {}
    ambient_whitespace: set[int] = set()
    for code_point in range(CODE_POINT_COUNT):
        character = chr(code_point)
        folded = tuple(ord(item) for item in character.casefold())
        if folded != (code_point,):
            ambient_casefold[code_point] = folded
        if character.isspace():
            ambient_whitespace.add(code_point)
    if ambient_casefold != official_casefold:
        differing = next(
            code_point
            for code_point in range(CODE_POINT_COUNT)
            if ambient_casefold.get(code_point, (code_point,))
            != official_casefold.get(code_point, (code_point,))
        )
        raise GenerationError(
            f"Python 3.13 casefold differs from Unicode C/F mapping at "
            f"U+{differing:04X}"
        )
    frozen_whitespace = frozenset(ambient_whitespace)
    if len(frozen_whitespace) != EXPECTED_PYTHON_WHITESPACE_COUNT:
        raise GenerationError(
            f"Python whitespace count differs: expected "
            f"{EXPECTED_PYTHON_WHITESPACE_COUNT}, found {len(frozen_whitespace)}"
        )
    if frozen_whitespace - unicode_whitespace != {0x1C, 0x1D, 0x1E, 0x1F}:
        raise GenerationError("Python-only whitespace positions differ")
    if unicode_whitespace - frozen_whitespace:
        raise GenerationError("Unicode White_Space contains a non-Python whitespace position")
    _framed_digests(ambient_casefold, frozen_whitespace)
    return ambient_casefold, frozen_whitespace, unicode_whitespace


def _artifact_record(kind: str, path: str, data: bytes) -> dict[str, Any]:
    return {
        "byteCount": len(data),
        "kind": kind,
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _split_conformance(semantics: FrozenStringSemantics) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    observed_transitions: set[str] = set()
    for source in SPLIT_INPUTS:
        tokens, transitions = semantics._split_whitespace_with_trace(source["input"])
        if _runtime_identity() == REFERENCE_RUNTIME and tokens != source["input"].split():
            raise GenerationError(
                f"frozen split differs from Python 3.13 for case {source['id']!r}"
            )
        observed_transitions.update(transitions)
        cases.append(
            {
                "id": source["id"],
                "input": source["input"],
                "tokens": tokens,
                "transitions": list(transitions),
            }
        )
    observed = sorted(observed_transitions)
    if observed != REQUIRED_SPLIT_TRANSITIONS:
        raise GenerationError(
            f"split transition coverage differs: expected "
            f"{REQUIRED_SPLIT_TRANSITIONS!r}, found {observed!r}"
        )
    case_bytes = _canonical_json(cases)
    return {
        "caseCount": len(cases),
        "corpusFraming": "canonical-json-ensure-ascii-sort-keys-indent-2-lf-v1",
        "corpusSha256": hashlib.sha256(case_bytes).hexdigest(),
        "observedTransitions": observed,
        "requiredTransitions": REQUIRED_SPLIT_TRANSITIONS,
    }


def _casefold_table(
    mappings: Mapping[int, tuple[int, ...]],
) -> dict[str, Any]:
    return {
        "artifactKind": "aleph-portable-python-casefold-table",
        "changedMappingsSha256": EXPECTED_CASEFOLD_CHANGED_SHA256,
        "defaultMapping": "identity",
        "formatVersion": 1,
        "framing": (
            "code-point-u32be+mapping-length-u32be+"
            "mapping-code-points-u32be-v1"
        ),
        "mappingCount": len(mappings),
        "mappings": [
            [code_point, list(mappings[code_point])]
            for code_point in sorted(mappings)
        ],
        "pythonVersion": "3.13.2",
        "selectedStatuses": ["C", "F"],
        "source": {
            "path": CASEFOLD_INPUT["path"],
            "sha256": CASEFOLD_INPUT["sha256"],
        },
        "unicodeVersion": "15.1.0",
    }


def _whitespace_table(
    whitespace: frozenset[int],
    unicode_whitespace: frozenset[int],
) -> dict[str, Any]:
    return {
        "artifactKind": "aleph-portable-python-whitespace-table",
        "codePoints": sorted(whitespace),
        "formatVersion": 1,
        "framing": "sorted-code-points-u32be-v1",
        "predicate": "python-str-isspace-v1",
        "pythonOnlyCodePoints": sorted(whitespace - unicode_whitespace),
        "pythonVersion": "3.13.2",
        "pythonWhitespaceCount": len(whitespace),
        "pythonWhitespaceSha256": EXPECTED_WHITESPACE_SHA256,
        "source": {
            "path": PROPLIST_INPUT["path"],
            "property": "White_Space",
            "sha256": PROPLIST_INPUT["sha256"],
        },
        "unicodeOnlyCodePoints": sorted(unicode_whitespace - whitespace),
        "unicodeVersion": "15.1.0",
        "unicodeWhiteSpaceCodePoints": sorted(unicode_whitespace),
        "unicodeWhiteSpaceCount": len(unicode_whitespace),
    }


def _expected_receipt(
    root: Path,
    *,
    casefold_record: dict[str, Any],
    whitespace_record: dict[str, Any],
    semantics: FrozenStringSemantics,
) -> dict[str, Any]:
    generator_bytes = _read_regular_file(root, GENERATOR_PATH, max_bytes=128 * 1024)
    return {
        "artifactKind": "aleph-portable-string-semantics-generation-receipt",
        "authorityRuntime": REFERENCE_RUNTIME,
        "casefold": {
            "changedMappingCount": EXPECTED_CHANGED_MAPPING_COUNT,
            "changedMappingsSha256": EXPECTED_CASEFOLD_CHANGED_SHA256,
            "defaultMapping": "identity",
            "excludedStatuses": ["S", "T"],
            "selectedStatuses": ["C", "F"],
        },
        "domain": {
            "codePointPositionCount": CODE_POINT_COUNT,
            "combinedFullDomainSha256": EXPECTED_COMBINED_SHA256,
            "endCodePointInclusive": CODE_POINT_COUNT - 1,
            "framing": (
                "code-point-u32be+whitespace-u8+mapping-length-u32be+"
                "mapping-code-points-u32be-v1"
            ),
            "scalarValuePositionCount": CODE_POINT_COUNT
            - (SURROGATE_END - SURROGATE_START),
            "startCodePoint": 0,
            "surrogatePositionCount": SURROGATE_END - SURROGATE_START,
        },
        "formatVersion": 1,
        "generator": {
            "byteCount": len(generator_bytes),
            "path": GENERATOR_PATH,
            "sha256": hashlib.sha256(generator_bytes).hexdigest(),
        },
        "inputs": [CASEFOLD_INPUT, PROPLIST_INPUT],
        "profileVersion": "0.3.0-provisional",
        "semanticsId": SEMANTICS_ID,
        "splitConformance": _split_conformance(semantics),
        "tables": [casefold_record, whitespace_record],
        "unicodeVersion": "15.1.0",
        "whitespace": {
            "predicate": "python-str-isspace-v1",
            "pythonOnlyCodePoints": [0x1C, 0x1D, 0x1E, 0x1F],
            "pythonWhitespaceCount": EXPECTED_PYTHON_WHITESPACE_COUNT,
            "pythonWhitespaceSha256": EXPECTED_WHITESPACE_SHA256,
            "unicodeOnlyCodePoints": [],
            "unicodeWhiteSpaceCount": EXPECTED_UNICODE_WHITESPACE_COUNT,
        },
    }


def _build_authority_outputs(root: Path) -> dict[str, bytes]:
    casefold_input = _read_pinned_input(root, CASEFOLD_INPUT)
    proplist_input = _read_pinned_input(root, PROPLIST_INPUT)
    mappings, whitespace, unicode_whitespace = _reference_semantics(
        casefold_input, proplist_input
    )
    semantics = FrozenStringSemantics(
        MappingProxyType(dict(mappings)), frozenset(whitespace)
    )
    casefold_bytes = _canonical_json(_casefold_table(mappings))
    whitespace_bytes = _canonical_json(
        _whitespace_table(whitespace, unicode_whitespace)
    )
    casefold_path = (
        "bench/portable/data/casefold-cf-"
        f"{hashlib.sha256(casefold_bytes).hexdigest()}.json"
    )
    whitespace_path = (
        "bench/portable/data/python-whitespace-"
        f"{hashlib.sha256(whitespace_bytes).hexdigest()}.json"
    )
    casefold_record = _artifact_record(
        "casefold-table", casefold_path, casefold_bytes
    )
    whitespace_record = _artifact_record(
        "whitespace-table", whitespace_path, whitespace_bytes
    )
    receipt = _expected_receipt(
        root,
        casefold_record=casefold_record,
        whitespace_record=whitespace_record,
        semantics=semantics,
    )
    receipt_bytes = _canonical_json(receipt)
    receipt_record = _artifact_record(
        "generation-receipt", RECEIPT_PATH, receipt_bytes
    )
    schema_bytes = _read_regular_file(root, SCHEMA_PATH, max_bytes=64 * 1024)
    if (
        len(schema_bytes) != SCHEMA_BYTE_COUNT
        or hashlib.sha256(schema_bytes).hexdigest() != SCHEMA_SHA256
    ):
        raise GenerationError(
            "string-semantics schema differs from the reviewed lock"
        )
    schema_record = {
        "byteCount": len(schema_bytes),
        "path": SCHEMA_PATH,
        "sha256": hashlib.sha256(schema_bytes).hexdigest(),
    }
    manifest = {
        "artifactCount": 3,
        "artifactKind": "aleph-portable-string-semantics-manifest",
        "artifacts": [casefold_record, receipt_record, whitespace_record],
        "formatVersion": 1,
        "hashAlgorithm": "sha256",
        "profileVersion": "0.3.0-provisional",
        "schema": schema_record,
        "semanticsId": SEMANTICS_ID,
    }
    return {
        casefold_path: casefold_bytes,
        MANIFEST_PATH: _canonical_json(manifest),
        RECEIPT_PATH: receipt_bytes,
        whitespace_path: whitespace_bytes,
    }


def _snapshot_schema_only_data_directory(root: Path) -> _DataDirectorySnapshot:
    """Accept the one-file staging state used to create authority candidates."""

    relative = PurePosixPath(DATA_DIRECTORY_PATH)
    schema_name = PurePosixPath(SCHEMA_PATH).name
    directory_fd = _open_directory_below_root(root, relative)
    recheck_fd: int | None = None
    try:
        opened = os.fstat(directory_fd)
        names_before = sorted(os.listdir(directory_fd))
        if names_before != [schema_name]:
            raise GenerationError(
                "authority output directory must be either schema-only or the exact "
                "five-file string-semantics closure"
            )
        schema_bytes = _read_regular_member(
            directory_fd,
            schema_name,
            max_bytes=64 * 1024,
            label=SCHEMA_PATH,
        )
        if (
            len(schema_bytes) != SCHEMA_BYTE_COUNT
            or hashlib.sha256(schema_bytes).hexdigest() != SCHEMA_SHA256
        ):
            raise GenerationError("string-semantics schema differs from the reviewed lock")
        schema_value = _strict_json(schema_bytes, label=SCHEMA_PATH)
        if schema_bytes != _canonical_json(schema_value):
            raise GenerationError("string-semantics schema is not canonical JSON")
        names_after = sorted(os.listdir(directory_fd))
        after = os.fstat(directory_fd)
        recheck_fd = _open_directory_below_root(root, relative)
        final = os.fstat(recheck_fd)
        if (
            names_after != names_before
            or not stat.S_ISDIR(after.st_mode)
            or not stat.S_ISDIR(final.st_mode)
            or _stat_state(after) != _stat_state(opened)
            or _stat_state(final) != _stat_state(opened)
        ):
            raise GenerationError(
                "string-semantics schema-only directory changed while it was read"
            )
        return _DataDirectorySnapshot(
            directory_state=_stat_state(opened),
            files=MappingProxyType({SCHEMA_PATH: schema_bytes}),
        )
    except (GenerationError, StringSemanticsError):
        raise
    except OSError as error:
        raise GenerationError(
            f"cannot safely enumerate schema-only authority directory: {error}"
        ) from error
    finally:
        if recheck_fd is not None:
            os.close(recheck_fd)
        os.close(directory_fd)


def _snapshot_authority_prestate(root: Path) -> _DataDirectorySnapshot:
    """Read either a complete closure or the explicit schema-only staging state."""

    try:
        return _snapshot_data_directory(root)
    except StringSemanticsError as closure_error:
        try:
            return _snapshot_schema_only_data_directory(root)
        except (GenerationError, StringSemanticsError) as staging_error:
            raise GenerationError(
                "authority output directory is neither a safe complete closure nor a "
                f"safe schema-only staging state: {closure_error}; {staging_error}"
            ) from staging_error


def _publish_data_member(
    directory_fd: int,
    name: str,
    data: bytes,
    *,
    existing: bytes | None,
) -> bool:
    """Create or replace one member relative to a verified directory descriptor."""

    required_flags = ("O_NOFOLLOW", "O_CLOEXEC")
    required_dir_fd = (os.open, os.rename, os.unlink, os.link, os.stat)
    if (
        any(not hasattr(os, flag) for flag in required_flags)
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
    ):
        raise GenerationError("authority writer requires fail-closed POSIX file operations")

    if existing is None:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise GenerationError(
                f"cannot safely inspect new generated artifact {name}: {error}"
            ) from error
        else:
            raise GenerationError(f"new generated artifact already exists: {name}")
    else:
        checked_existing = _read_regular_member(
            directory_fd,
            name,
            max_bytes=max(len(existing), 1),
            label=f"existing generated artifact {name}",
        )
        if checked_existing != existing:
            raise GenerationError(f"generated artifact changed before write: {name}")
        if existing == data:
            return False

    temporary_name = f".{name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor: int | None = None
    temporary_exists = False
    try:
        descriptor = os.open(temporary_name, flags, 0o644, dir_fd=directory_fd)
        temporary_exists = True
        offset = 0
        while offset < len(data):
            written_count = os.write(descriptor, data[offset:])
            if written_count <= 0:
                raise GenerationError(f"short write for generated artifact {name}")
            offset += written_count
        os.fsync(descriptor)
        written = os.fstat(descriptor)
        if (
            not stat.S_ISREG(written.st_mode)
            or written.st_nlink != 1
            or written.st_size != len(data)
        ):
            raise GenerationError("temporary authority artifact is not stable")
        os.close(descriptor)
        descriptor = None
        temporary_readback = _read_regular_member(
            directory_fd,
            temporary_name,
            max_bytes=max(len(data), 1),
            label=f"temporary generated artifact {name}",
        )
        if temporary_readback != data:
            raise GenerationError(f"temporary generated artifact differs: {name}")

        if existing is None:
            try:
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as error:
                raise GenerationError(
                    f"cannot safely recheck new generated artifact {name}: {error}"
                ) from error
            else:
                raise GenerationError(f"new generated artifact appeared during write: {name}")
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=directory_fd)
            temporary_exists = False
        else:
            checked_existing = _read_regular_member(
                directory_fd,
                name,
                max_bytes=max(len(existing), 1),
                label=f"existing generated artifact {name}",
            )
            if checked_existing != existing:
                raise GenerationError(f"generated artifact changed before replace: {name}")
            os.rename(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary_exists = False
        os.fsync(directory_fd)
        readback = _read_regular_member(
            directory_fd,
            name,
            max_bytes=max(len(data), 1),
            label=f"written generated artifact {name}",
        )
        if readback != data:
            raise GenerationError(f"generated artifact readback differs: {name}")
        return True
    except (GenerationError, StringSemanticsError):
        raise
    except OSError as error:
        raise GenerationError(
            f"cannot safely publish generated artifact {name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except OSError:
                pass


def write_authority(root: Path = ROOT) -> dict[str, Any]:
    _require_reference_runtime()
    try:
        before = _snapshot_authority_prestate(root)
    except (GenerationError, StringSemanticsError) as error:
        raise GenerationError(str(error)) from error
    outputs = _build_authority_outputs(root)
    manifest_bytes = outputs[MANIFEST_PATH]
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    expected_paths = set(outputs) | {SCHEMA_PATH}
    schema_only = set(before.files) == {SCHEMA_PATH}
    if not schema_only and set(before.files) != expected_paths:
        raise GenerationError(
            "existing authority closure uses different generated paths; generate the "
            "candidate in a safe schema-only staging root"
        )
    data_directory = PurePosixPath(DATA_DIRECTORY_PATH)
    directory_fd = _open_directory_below_root(root, data_directory)
    changed = False
    try:
        opened = os.fstat(directory_fd)
        if _stat_state(opened) != before.directory_state:
            raise GenerationError(
                "string-semantics data directory changed before authority write"
            )
        current_names = sorted(os.listdir(directory_fd))
        if current_names != sorted(
            PurePosixPath(path).name for path in before.files
        ):
            raise GenerationError(
                "string-semantics data directory changed before authority write"
            )
        publish_order = sorted(path for path in outputs if path != MANIFEST_PATH)
        publish_order.append(MANIFEST_PATH)
        for relative_path in publish_order:
            data = outputs[relative_path]
            existing = before.files.get(relative_path)
            changed = (
                _publish_data_member(
                    directory_fd,
                    PurePosixPath(relative_path).name,
                    data,
                    existing=existing,
                )
                or changed
            )
    except (GenerationError, StringSemanticsError) as error:
        raise GenerationError(str(error)) from error
    finally:
        os.close(directory_fd)
    try:
        after = _snapshot_data_directory(root)
    except StringSemanticsError as error:
        raise GenerationError(str(error)) from error
    expected_files = dict(outputs)
    expected_files[SCHEMA_PATH] = before.files[SCHEMA_PATH]
    if dict(after.files) != expected_files:
        raise GenerationError("authority write readback differs from generated closure")
    consumer_lock_matches = (
        len(manifest_bytes) == MANIFEST_BYTE_COUNT
        and manifest_sha256 == MANIFEST_SHA256
    )
    if consumer_lock_matches:
        try:
            load_string_semantics(root)
        except StringSemanticsError as error:
            raise GenerationError(
                f"authority write consumer readback failed: {error}"
            ) from error
    return {
        "artifactCount": 3,
        "consumerLockMatches": consumer_lock_matches,
        "manifestBytes": len(manifest_bytes),
        "manifestSha256": manifest_sha256,
        "status": "written" if changed else "unchanged",
    }


def _validate_manifest_schema(manifest: Any, schema: Any) -> None:
    try:
        from bench.engine.schema_validation import SchemaValidationError, validate

        validate(manifest, schema)
    except SchemaValidationError as error:
        raise GenerationError(
            f"string-semantics manifest schema validation failed: {error}"
        ) from error


def _compare_official_inputs(
    root: Path, semantics: FrozenStringSemantics
) -> tuple[dict[int, tuple[int, ...]], frozenset[int]]:
    casefold_input = _read_pinned_input(root, CASEFOLD_INPUT)
    proplist_input = _read_pinned_input(root, PROPLIST_INPUT)
    official_casefold = _parse_casefold_input(casefold_input)
    unicode_whitespace = _parse_unicode_whitespace(proplist_input)
    for code_point in range(CODE_POINT_COUNT):
        expected_mapping = official_casefold.get(code_point, (code_point,))
        if semantics.mapping_for_code_point(code_point) != expected_mapping:
            raise GenerationError(
                f"frozen casefold differs from Unicode C/F mapping at "
                f"U+{code_point:04X}"
            )
    whitespace = semantics.whitespace_code_points
    if whitespace - unicode_whitespace != {0x1C, 0x1D, 0x1E, 0x1F}:
        raise GenerationError("frozen Python-only whitespace positions differ")
    if unicode_whitespace - whitespace:
        raise GenerationError("frozen table omits a Unicode White_Space position")
    return official_casefold, unicode_whitespace


def verify(root: Path = ROOT) -> dict[str, Any]:
    try:
        snapshot = _snapshot_data_directory(root)
        semantics = load_string_semantics(root)
    except StringSemanticsError as error:
        raise GenerationError(str(error)) from error
    manifest_bytes = snapshot.files[MANIFEST_PATH]
    manifest = _strict_json(manifest_bytes, label=MANIFEST_PATH)
    schema_bytes = snapshot.files[SCHEMA_PATH]
    if (
        len(schema_bytes) != SCHEMA_BYTE_COUNT
        or hashlib.sha256(schema_bytes).hexdigest() != SCHEMA_SHA256
    ):
        raise GenerationError("string-semantics schema bytes differ from the frozen lock")
    schema = _strict_json(schema_bytes, label=SCHEMA_PATH)
    if schema_bytes != _canonical_json(schema):
        raise GenerationError("string-semantics schema is not canonical JSON")
    _validate_manifest_schema(manifest, schema)

    records = {
        record["kind"]: record
        for record in manifest["artifacts"]
        if isinstance(record, dict) and isinstance(record.get("kind"), str)
    }
    if set(records) != {"casefold-table", "generation-receipt", "whitespace-table"}:
        raise GenerationError("string-semantics manifest artifact kinds differ")
    receipt_bytes = _read_snapshot_artifact(
        snapshot, records["generation-receipt"], max_bytes=128 * 1024
    )
    receipt = _strict_json(receipt_bytes, label=records["generation-receipt"]["path"])
    if receipt_bytes != _canonical_json(receipt):
        raise GenerationError("string-semantics generation receipt is not canonical JSON")
    _compare_official_inputs(root, semantics)
    digests = _framed_digests(
        semantics._casefold_mappings, semantics.whitespace_code_points
    )
    expected_receipt = _expected_receipt(
        root,
        casefold_record=records["casefold-table"],
        whitespace_record=records["whitespace-table"],
        semantics=semantics,
    )
    if receipt != expected_receipt:
        raise GenerationError(
            "string-semantics generation receipt differs from independently "
            "recomputed authority metadata"
        )

    regeneration_checked = False
    if _runtime_identity() == REFERENCE_RUNTIME:
        regenerated = _build_authority_outputs(root)
        expected_paths = {
            records["casefold-table"]["path"],
            records["generation-receipt"]["path"],
            records["whitespace-table"]["path"],
            MANIFEST_PATH,
        }
        if set(regenerated) != expected_paths:
            raise GenerationError("regenerated authority path closure differs")
        for relative_path, expected_bytes in regenerated.items():
            checked_bytes = snapshot.files.get(relative_path)
            if checked_bytes != expected_bytes:
                raise GenerationError(
                    f"regenerated authority bytes differ at {relative_path}"
                )
        regeneration_checked = True

    return {
        "authorityRegenerationChecked": regeneration_checked,
        "casefoldChangedMappingCount": semantics.changed_mapping_count,
        "casefoldChangedSha256": digests["casefoldChangedSha256"],
        "codePointPositionCount": CODE_POINT_COUNT,
        "combinedFullDomainSha256": digests["combinedFullDomainSha256"],
        "manifestBytes": MANIFEST_BYTE_COUNT,
        "manifestSha256": MANIFEST_SHA256,
        "pythonWhitespaceCount": len(semantics.whitespace_code_points),
        "pythonWhitespaceSha256": digests["pythonWhitespaceSha256"],
        "splitCaseCount": receipt["splitConformance"]["caseCount"],
        "status": "ok",
        "surrogatePositionCount": SURROGATE_END - SURROGATE_START,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify the frozen Python 3.13 casefold, whitespace, "
            "and split semantics."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify checked-in bytes offline")
    mode.add_argument(
        "--write",
        action="store_true",
        help="regenerate authority bytes on exact CPython 3.13.2 only",
    )
    arguments = parser.parse_args(argv)
    try:
        report = write_authority() if arguments.write else verify()
    except (GenerationError, StringSemanticsError) as error:
        print(f"portable string semantics: FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
