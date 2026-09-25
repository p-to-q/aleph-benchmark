"""Consume the frozen Python 3.13 casefold and whitespace semantics.

The implementation is intentionally independent of the host interpreter's
``str.casefold``, ``str.isspace``, and no-argument ``str.split`` behavior.
Authority generation lives in ``scripts/generate-portable-string-semantics.py``;
this module is the read-only, pure consumer boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY_PATH = "bench/portable/data"
MANIFEST_PATH = "bench/portable/data/portable-string-semantics-v1.manifest.json"
MANIFEST_BYTE_COUNT = 1_276
MANIFEST_SHA256 = "ca28e6cfc636493e9a18156d313ddea4cb77a57212a12530163564a4d3e0f19f"
SCHEMA_PATH = "bench/portable/data/portable-string-semantics-v1.schema.json"
SCHEMA_BYTE_COUNT = 2_618
SCHEMA_SHA256 = "046b6b714fdb0e4358b352840fa2b4763ce396082e86dd1e1cbe492976c76b61"

CODE_POINT_COUNT = 0x110000
SURROGATE_START = 0xD800
SURROGATE_END = 0xE000
EXPECTED_CHANGED_MAPPING_COUNT = 1_530
EXPECTED_PYTHON_WHITESPACE_COUNT = 29
EXPECTED_UNICODE_WHITESPACE_COUNT = 25
EXPECTED_CASEFOLD_CHANGED_SHA256 = (
    "b4127130645c3d4a153f99d25f7fa6ffcf28c32c4f6cd5e553790e490170a9cd"
)
EXPECTED_WHITESPACE_SHA256 = (
    "9096d8eea85f36f62cb291c93fbae4baa00cb701d2145ba9f0ca646fe2e619bb"
)
EXPECTED_COMBINED_SHA256 = (
    "0e1ff78255295b92d0c7f5e13aa5f9dfa9df72f55609d90373ca2f199f2a7e52"
)
MAX_DATA_ARTIFACT_BYTES = 512 * 1024


class StringSemanticsError(RuntimeError):
    """Raised when the frozen string-semantics closure fails closed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (RecursionError, TypeError, ValueError) as error:
        raise StringSemanticsError(
            f"cannot encode canonical strict JSON: {error}"
        ) from error


def _strict_json(data: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise StringSemanticsError(
            f"{label} contains non-finite JSON value {value!r}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StringSemanticsError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise StringSemanticsError(
            f"{label} is not strict UTF-8 JSON: {error}"
        ) from error


def _validated_relative_path(raw_path: str) -> PurePosixPath:
    path = PurePosixPath(raw_path)
    if (
        not raw_path
        or path.is_absolute()
        or path.as_posix() != raw_path
        or ".." in path.parts
        or "\\" in raw_path
        or any(ord(character) < 0x20 for character in raw_path)
    ):
        raise StringSemanticsError(
            f"unsafe or non-canonical repository path: {raw_path!r}"
        )
    return path


def _required_filesystem_flags() -> tuple[int, int]:
    required = ("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK", "O_DIRECTORY")
    if any(not hasattr(os, name) for name in required):
        raise StringSemanticsError(
            "string-semantics verification requires fail-closed filesystem flags"
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    return directory_flags, file_flags


def _stat_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_directory_below_root(root: Path, relative: PurePosixPath) -> int:
    directory_flags, _ = _required_filesystem_flags()
    try:
        current_fd = os.open(root, directory_flags)
    except OSError as error:
        raise StringSemanticsError(f"cannot safely open repository root: {error}") from error
    try:
        for part in relative.parts:
            next_fd: int | None = None
            try:
                before = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
                opened = os.fstat(next_fd)
                after = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            except OSError as error:
                if next_fd is not None:
                    os.close(next_fd)
                raise StringSemanticsError(
                    f"cannot safely open directory {relative.as_posix()}: {error}"
                ) from error
            if (
                not stat.S_ISDIR(before.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(after.st_mode)
                or _stat_state(before) != _stat_state(opened)
                or _stat_state(after) != _stat_state(opened)
            ):
                os.close(next_fd)
                raise StringSemanticsError(
                    f"directory {relative.as_posix()} changed while it was opened"
                )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _safe_basename(name: str, *, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or PurePosixPath(name).name != name
        or "/" in name
        or "\\" in name
        or any(ord(character) < 0x20 for character in name)
    ):
        raise StringSemanticsError(f"unsafe {label}: {name!r}")
    return name


def _read_regular_member(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Snapshot one member with stable pre/open/post metadata checks."""

    name = _safe_basename(name, label=label)
    _, file_flags = _required_filesystem_flags()
    file_fd: int | None = None
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > max_bytes
        ):
            raise StringSemanticsError(
                f"{label} must be one bounded single-link regular file"
            )
        file_fd = os.open(name, file_flags, dir_fd=directory_fd)
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stat_state(opened) != _stat_state(before)
        ):
            raise StringSemanticsError(f"{label} changed while it was opened")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(file_fd)
        final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            len(data) > max_bytes
            or len(data) != opened.st_size
            or after.st_nlink != 1
            or final.st_nlink != 1
            or _stat_state(after) != _stat_state(opened)
            or _stat_state(final) != _stat_state(opened)
        ):
            raise StringSemanticsError(f"{label} changed while it was read")
        return data
    except StringSemanticsError:
        raise
    except OSError as error:
        raise StringSemanticsError(f"cannot safely read {label}: {error}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)


def _read_regular_file(root: Path, raw_path: str, *, max_bytes: int) -> bytes:
    """Read below ``root`` without following aliases in any path component."""

    relative_path = _validated_relative_path(raw_path)
    parent = PurePosixPath(*relative_path.parts[:-1])
    parent_fd = _open_directory_below_root(root, parent)
    try:
        return _read_regular_member(
            parent_fd,
            relative_path.name,
            max_bytes=max_bytes,
            label=raw_path,
        )
    finally:
        os.close(parent_fd)


@dataclass(frozen=True, slots=True)
class _DataDirectorySnapshot:
    directory_state: tuple[int, ...]
    files: Mapping[str, bytes]


def _manifest_artifact_paths(manifest: Any) -> set[str]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), list):
        raise StringSemanticsError("string-semantics manifest artifact closure is invalid")
    if len(manifest["artifacts"]) != 3:
        raise StringSemanticsError("string-semantics manifest must name three artifacts")
    paths = {MANIFEST_PATH, SCHEMA_PATH}
    for record in manifest["artifacts"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"byteCount", "kind", "path", "sha256"}
            or not isinstance(record.get("path"), str)
        ):
            raise StringSemanticsError("string-semantics manifest artifact path is invalid")
        path = _validated_relative_path(record["path"])
        if PurePosixPath(*path.parts[:-1]).as_posix() != DATA_DIRECTORY_PATH:
            raise StringSemanticsError("string-semantics artifact escapes the data directory")
        paths.add(path.as_posix())
    if len(paths) != 5:
        raise StringSemanticsError("string-semantics manifest path closure differs")
    return paths


def _snapshot_data_directory(root: Path) -> _DataDirectorySnapshot:
    """Read the exact five-file data closure through one stable directory fd."""

    relative = _validated_relative_path(DATA_DIRECTORY_PATH)
    directory_fd = _open_directory_below_root(root, relative)
    recheck_fd: int | None = None
    try:
        opened = os.fstat(directory_fd)
        names_before = sorted(os.listdir(directory_fd))
        if (
            len(names_before) != 5
            or PurePosixPath(MANIFEST_PATH).name not in names_before
            or PurePosixPath(SCHEMA_PATH).name not in names_before
            or any(
                _safe_basename(name, label="data entry") != name
                or not name.endswith(".json")
                for name in names_before
            )
        ):
            raise StringSemanticsError(
                "string-semantics data directory is not the exact closed-world file set"
            )
        files_by_name = {
            name: _read_regular_member(
                directory_fd,
                name,
                max_bytes=MAX_DATA_ARTIFACT_BYTES,
                label=f"string-semantics data entry {name}",
            )
            for name in names_before
        }
        manifest_name = PurePosixPath(MANIFEST_PATH).name
        manifest_bytes = files_by_name[manifest_name]
        manifest = _strict_json(manifest_bytes, label=MANIFEST_PATH)
        if manifest_bytes != _canonical_json(manifest):
            raise StringSemanticsError(
                "string-semantics manifest is not canonical JSON"
            )
        expected_paths = _manifest_artifact_paths(manifest)
        expected_names = sorted(PurePosixPath(path).name for path in expected_paths)
        if names_before != expected_names:
            raise StringSemanticsError(
                "string-semantics data directory is not the manifest's closed-world file set"
            )
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
            raise StringSemanticsError(
                "string-semantics data directory changed while it was read"
            )
        return _DataDirectorySnapshot(
            directory_state=_stat_state(opened),
            files=MappingProxyType(
                {
                    f"{DATA_DIRECTORY_PATH}/{name}": data
                    for name, data in files_by_name.items()
                }
            ),
        )
    except StringSemanticsError:
        raise
    except OSError as error:
        raise StringSemanticsError(
            f"cannot safely enumerate string-semantics data: {error}"
        ) from error
    finally:
        if recheck_fd is not None:
            os.close(recheck_fd)
        os.close(directory_fd)


def _require_exact_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StringSemanticsError(f"{label} must be an object")
    if set(value) != expected:
        raise StringSemanticsError(
            f"{label} keys differ: expected {sorted(expected)!r}, "
            f"found {sorted(value)!r}"
        )
    return value


def _artifact_by_kind(manifest: dict[str, Any], kind: str) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise StringSemanticsError("string-semantics manifest artifacts must be an array")
    matches = [record for record in artifacts if isinstance(record, dict) and record.get("kind") == kind]
    if len(matches) != 1:
        raise StringSemanticsError(
            f"string-semantics manifest must contain exactly one {kind!r} artifact"
        )
    return matches[0]


def _read_artifact(root: Path, record: dict[str, Any], *, max_bytes: int) -> bytes:
    record = _require_exact_keys(
        record,
        {"byteCount", "kind", "path", "sha256"},
        label="string-semantics artifact record",
    )
    byte_count = record["byteCount"]
    expected_sha256 = record["sha256"]
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count < 1
        or byte_count > max_bytes
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise StringSemanticsError("string-semantics artifact metadata is invalid")
    data = _read_regular_file(root, record["path"], max_bytes=byte_count)
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if len(data) != byte_count or actual_sha256 != expected_sha256:
        raise StringSemanticsError(
            f"string-semantics artifact differs at {record['path']}: expected "
            f"bytes={byte_count} sha256={expected_sha256}, found "
            f"bytes={len(data)} sha256={actual_sha256}"
        )
    return data


def _read_snapshot_artifact(
    snapshot: _DataDirectorySnapshot,
    record: dict[str, Any],
    *,
    max_bytes: int,
) -> bytes:
    record = _require_exact_keys(
        record,
        {"byteCount", "kind", "path", "sha256"},
        label="string-semantics artifact record",
    )
    byte_count = record["byteCount"]
    expected_sha256 = record["sha256"]
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count < 1
        or byte_count > max_bytes
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or record["path"] not in snapshot.files
    ):
        raise StringSemanticsError("string-semantics artifact metadata is invalid")
    data = snapshot.files[record["path"]]
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if len(data) != byte_count or actual_sha256 != expected_sha256:
        raise StringSemanticsError(
            f"string-semantics artifact differs at {record['path']}: expected "
            f"bytes={byte_count} sha256={expected_sha256}, found "
            f"bytes={len(data)} sha256={actual_sha256}"
        )
    return data


def _changed_stream_digest(mappings: Mapping[int, tuple[int, ...]]) -> str:
    digest = hashlib.sha256()
    for code_point in sorted(mappings):
        mapped = mappings[code_point]
        digest.update(code_point.to_bytes(4, "big"))
        digest.update(len(mapped).to_bytes(4, "big"))
        for mapped_code_point in mapped:
            digest.update(mapped_code_point.to_bytes(4, "big"))
    return digest.hexdigest()


def _whitespace_stream_digest(code_points: frozenset[int]) -> str:
    digest = hashlib.sha256()
    for code_point in sorted(code_points):
        digest.update(code_point.to_bytes(4, "big"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenStringSemantics:
    """Frozen casefold mapping and whitespace predicate for portable consumers."""

    _casefold_mappings: Mapping[int, tuple[int, ...]]
    _whitespace_code_points: frozenset[int]

    @property
    def changed_mapping_count(self) -> int:
        return len(self._casefold_mappings)

    @property
    def whitespace_code_points(self) -> frozenset[int]:
        return self._whitespace_code_points

    def mapping_for_code_point(self, code_point: int) -> tuple[int, ...]:
        if (
            not isinstance(code_point, int)
            or isinstance(code_point, bool)
            or not 0 <= code_point < CODE_POINT_COUNT
        ):
            raise ValueError("code_point must be an integer from 0 through 0x10FFFF")
        return self._casefold_mappings.get(code_point, (code_point,))

    def casefold(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return "".join(
            chr(mapped_code_point)
            for character in text
            for mapped_code_point in self.mapping_for_code_point(ord(character))
        )

    def is_whitespace(self, text: str) -> bool:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return bool(text) and all(
            ord(character) in self._whitespace_code_points for character in text
        )

    def _split_whitespace(
        self, text: str, *, collect_trace: bool
    ) -> tuple[list[str], tuple[str, ...]]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[str] = []
        transitions: list[str] | None = [] if collect_trace else None
        token_start: int | None = None
        for index, character in enumerate(text):
            if ord(character) in self._whitespace_code_points:
                if token_start is None:
                    if transitions is not None:
                        transitions.append("whitespace:skip")
                else:
                    tokens.append(text[token_start:index])
                    token_start = None
                    if transitions is not None:
                        transitions.append("token:end")
            elif token_start is None:
                token_start = index
                if transitions is not None:
                    transitions.append("token:start")
            elif transitions is not None:
                transitions.append("token:continue")
        if token_start is None:
            if transitions is not None:
                transitions.append("eof:idle")
        else:
            tokens.append(text[token_start:])
            if transitions is not None:
                transitions.append("eof:emit")
        return tokens, tuple(transitions or ())

    def _split_whitespace_with_trace(
        self, text: str
    ) -> tuple[list[str], tuple[str, ...]]:
        return self._split_whitespace(text, collect_trace=True)

    def split_whitespace(self, text: str) -> list[str]:
        """Implement no-argument ``str.split`` from the frozen predicate."""

        tokens, _ = self._split_whitespace(text, collect_trace=False)
        return tokens


def _parse_casefold_table(value: Any) -> Mapping[int, tuple[int, ...]]:
    table = _require_exact_keys(
        value,
        {
            "artifactKind",
            "changedMappingsSha256",
            "defaultMapping",
            "formatVersion",
            "framing",
            "mappingCount",
            "mappings",
            "pythonVersion",
            "selectedStatuses",
            "source",
            "unicodeVersion",
        },
        label="casefold table",
    )
    expected_fields = {
        "artifactKind": "aleph-portable-python-casefold-table",
        "changedMappingsSha256": EXPECTED_CASEFOLD_CHANGED_SHA256,
        "defaultMapping": "identity",
        "formatVersion": 1,
        "framing": "code-point-u32be+mapping-length-u32be+mapping-code-points-u32be-v1",
        "mappingCount": EXPECTED_CHANGED_MAPPING_COUNT,
        "pythonVersion": "3.13.2",
        "selectedStatuses": ["C", "F"],
        "unicodeVersion": "15.1.0",
    }
    for key, expected in expected_fields.items():
        if table[key] != expected:
            raise StringSemanticsError(
                f"casefold table {key} differs: expected {expected!r}, found {table[key]!r}"
            )
    source = _require_exact_keys(
        table["source"],
        {"path", "sha256"},
        label="casefold source",
    )
    if source != {
        "path": "bench/vendor/unicode/15.1.0/ucd/CaseFolding.txt",
        "sha256": "4e55acfdc32825a22e87670e9056a3bf94ad7c5400065778e9e10f8314372bcf",
    }:
        raise StringSemanticsError("casefold table source identity differs")
    raw_mappings = table["mappings"]
    if not isinstance(raw_mappings, list) or len(raw_mappings) != EXPECTED_CHANGED_MAPPING_COUNT:
        raise StringSemanticsError("casefold table mapping count differs")
    mappings: dict[int, tuple[int, ...]] = {}
    previous = -1
    for index, entry in enumerate(raw_mappings):
        if not isinstance(entry, list) or len(entry) != 2:
            raise StringSemanticsError(f"casefold mapping {index} must be a two-item array")
        code_point, raw_mapping = entry
        if (
            not isinstance(code_point, int)
            or isinstance(code_point, bool)
            or not 0 <= code_point < CODE_POINT_COUNT
            or code_point <= previous
            or not isinstance(raw_mapping, list)
            or not raw_mapping
        ):
            raise StringSemanticsError(f"casefold mapping {index} is invalid or unsorted")
        mapped: list[int] = []
        for mapped_code_point in raw_mapping:
            if (
                not isinstance(mapped_code_point, int)
                or isinstance(mapped_code_point, bool)
                or not 0 <= mapped_code_point < CODE_POINT_COUNT
            ):
                raise StringSemanticsError(
                    f"casefold mapping {index} contains an invalid code point"
                )
            mapped.append(mapped_code_point)
        mapped_tuple = tuple(mapped)
        if mapped_tuple == (code_point,):
            raise StringSemanticsError(
                f"casefold mapping {index} redundantly records the identity mapping"
            )
        mappings[code_point] = mapped_tuple
        previous = code_point
    if _changed_stream_digest(mappings) != EXPECTED_CASEFOLD_CHANGED_SHA256:
        raise StringSemanticsError("casefold changed-mapping stream digest differs")
    return MappingProxyType(mappings)


def _parse_code_point_array(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise StringSemanticsError(f"{label} must be an array")
    result: list[int] = []
    previous = -1
    for index, code_point in enumerate(value):
        if (
            not isinstance(code_point, int)
            or isinstance(code_point, bool)
            or not 0 <= code_point < CODE_POINT_COUNT
            or code_point <= previous
        ):
            raise StringSemanticsError(
                f"{label} item {index} is invalid, duplicated, or unsorted"
            )
        result.append(code_point)
        previous = code_point
    return tuple(result)


def _parse_whitespace_table(value: Any) -> frozenset[int]:
    table = _require_exact_keys(
        value,
        {
            "artifactKind",
            "codePoints",
            "formatVersion",
            "framing",
            "predicate",
            "pythonOnlyCodePoints",
            "pythonVersion",
            "pythonWhitespaceCount",
            "pythonWhitespaceSha256",
            "source",
            "unicodeOnlyCodePoints",
            "unicodeVersion",
            "unicodeWhiteSpaceCodePoints",
            "unicodeWhiteSpaceCount",
        },
        label="whitespace table",
    )
    expected_fields = {
        "artifactKind": "aleph-portable-python-whitespace-table",
        "formatVersion": 1,
        "framing": "sorted-code-points-u32be-v1",
        "predicate": "python-str-isspace-v1",
        "pythonVersion": "3.13.2",
        "pythonWhitespaceCount": EXPECTED_PYTHON_WHITESPACE_COUNT,
        "pythonWhitespaceSha256": EXPECTED_WHITESPACE_SHA256,
        "unicodeVersion": "15.1.0",
        "unicodeWhiteSpaceCount": EXPECTED_UNICODE_WHITESPACE_COUNT,
    }
    for key, expected in expected_fields.items():
        if table[key] != expected:
            raise StringSemanticsError(
                f"whitespace table {key} differs: expected {expected!r}, found {table[key]!r}"
            )
    source = _require_exact_keys(
        table["source"],
        {"path", "property", "sha256"},
        label="whitespace source",
    )
    if source != {
        "path": "bench/vendor/unicode/15.1.0/ucd/PropList.txt",
        "property": "White_Space",
        "sha256": "05672956317b6296bc2ec3d6cef1f6452b57ff4f2efc6dc55b0a19277d5fcfd1",
    }:
        raise StringSemanticsError("whitespace table source identity differs")
    python_code_points = _parse_code_point_array(table["codePoints"], label="Python whitespace")
    unicode_code_points = _parse_code_point_array(
        table["unicodeWhiteSpaceCodePoints"], label="Unicode White_Space"
    )
    python_only = _parse_code_point_array(
        table["pythonOnlyCodePoints"], label="Python-only whitespace"
    )
    unicode_only = _parse_code_point_array(
        table["unicodeOnlyCodePoints"], label="Unicode-only whitespace"
    )
    if (
        len(python_code_points) != EXPECTED_PYTHON_WHITESPACE_COUNT
        or len(unicode_code_points) != EXPECTED_UNICODE_WHITESPACE_COUNT
        or python_only != (0x1C, 0x1D, 0x1E, 0x1F)
        or unicode_only
        or set(python_code_points) - set(unicode_code_points) != set(python_only)
        or set(unicode_code_points) - set(python_code_points) != set(unicode_only)
    ):
        raise StringSemanticsError(
            "Python whitespace and Unicode White_Space difference is invalid"
        )
    frozen = frozenset(python_code_points)
    if _whitespace_stream_digest(frozen) != EXPECTED_WHITESPACE_SHA256:
        raise StringSemanticsError("Python whitespace stream digest differs")
    return frozen


def _validate_generation_receipt(
    receipt: Any,
    *,
    casefold_record: dict[str, Any],
    whitespace_record: dict[str, Any],
) -> None:
    receipt = _require_exact_keys(
        receipt,
        {
            "artifactKind",
            "authorityRuntime",
            "casefold",
            "domain",
            "formatVersion",
            "generator",
            "inputs",
            "profileVersion",
            "semanticsId",
            "splitConformance",
            "tables",
            "unicodeVersion",
            "whitespace",
        },
        label="string-semantics generation receipt",
    )
    fixed_fields = {
        "artifactKind": "aleph-portable-string-semantics-generation-receipt",
        "formatVersion": 1,
        "profileVersion": "0.3.0-provisional",
        "semanticsId": "python-3.13.2-ucd-15.1-string-semantics-v1",
        "unicodeVersion": "15.1.0",
    }
    for key, expected in fixed_fields.items():
        if receipt[key] != expected:
            raise StringSemanticsError(
                f"string-semantics generation receipt {key} differs"
            )
    if receipt["authorityRuntime"] != {
        "cacheTag": "cpython-313",
        "implementation": "CPython",
        "pythonHexVersion": "0x030d02f0",
        "pythonVersion": "3.13.2",
        "unicodeDatabaseVersion": "15.1.0",
    }:
        raise StringSemanticsError("string-semantics authority runtime differs")
    if receipt["casefold"] != {
        "changedMappingCount": EXPECTED_CHANGED_MAPPING_COUNT,
        "changedMappingsSha256": EXPECTED_CASEFOLD_CHANGED_SHA256,
        "defaultMapping": "identity",
        "excludedStatuses": ["S", "T"],
        "selectedStatuses": ["C", "F"],
    }:
        raise StringSemanticsError("string-semantics casefold receipt differs")
    if receipt["domain"] != {
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
    }:
        raise StringSemanticsError("string-semantics domain receipt differs")
    generator = _require_exact_keys(
        receipt["generator"],
        {"byteCount", "path", "sha256"},
        label="string-semantics generator receipt",
    )
    if (
        generator["path"] != "scripts/generate-portable-string-semantics.py"
        or not isinstance(generator["byteCount"], int)
        or isinstance(generator["byteCount"], bool)
        or not 1 <= generator["byteCount"] <= 128 * 1024
        or not isinstance(generator["sha256"], str)
        or len(generator["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in generator["sha256"])
    ):
        raise StringSemanticsError("string-semantics generator identity is invalid")
    if receipt["inputs"] != [
        {
            "byteCount": 84_870,
            "path": "bench/vendor/unicode/15.1.0/ucd/CaseFolding.txt",
            "sha256": "4e55acfdc32825a22e87670e9056a3bf94ad7c5400065778e9e10f8314372bcf",
            "unicodeVersion": "15.1.0",
        },
        {
            "byteCount": 136_397,
            "path": "bench/vendor/unicode/15.1.0/ucd/PropList.txt",
            "sha256": "05672956317b6296bc2ec3d6cef1f6452b57ff4f2efc6dc55b0a19277d5fcfd1",
            "unicodeVersion": "15.1.0",
        },
    ]:
        raise StringSemanticsError("string-semantics upstream input receipt differs")
    if receipt["splitConformance"] != {
        "caseCount": 8,
        "corpusFraming": "canonical-json-ensure-ascii-sort-keys-indent-2-lf-v1",
        "corpusSha256": "2ae9846ed89da198e6e14dcc9d5646a8a8919b25acdffcdf5927e2cdd3607749",
        "observedTransitions": [
            "eof:emit",
            "eof:idle",
            "token:continue",
            "token:end",
            "token:start",
            "whitespace:skip",
        ],
        "requiredTransitions": [
            "eof:emit",
            "eof:idle",
            "token:continue",
            "token:end",
            "token:start",
            "whitespace:skip",
        ],
    }:
        raise StringSemanticsError("string-semantics split receipt differs")
    if receipt["tables"] != [casefold_record, whitespace_record]:
        raise StringSemanticsError("string-semantics receipt table bindings differ")
    if receipt["whitespace"] != {
        "predicate": "python-str-isspace-v1",
        "pythonOnlyCodePoints": [0x1C, 0x1D, 0x1E, 0x1F],
        "pythonWhitespaceCount": EXPECTED_PYTHON_WHITESPACE_COUNT,
        "pythonWhitespaceSha256": EXPECTED_WHITESPACE_SHA256,
        "unicodeOnlyCodePoints": [],
        "unicodeWhiteSpaceCount": EXPECTED_UNICODE_WHITESPACE_COUNT,
    }:
        raise StringSemanticsError("string-semantics whitespace receipt differs")


def load_string_semantics(root: Path = ROOT) -> FrozenStringSemantics:
    """Load and independently verify the content-addressed frozen tables."""

    snapshot = _snapshot_data_directory(root)
    manifest_bytes = snapshot.files[MANIFEST_PATH]
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        len(manifest_bytes) != MANIFEST_BYTE_COUNT
        or actual_manifest_sha256 != MANIFEST_SHA256
    ):
        raise StringSemanticsError(
            f"string-semantics manifest differs: expected bytes={MANIFEST_BYTE_COUNT} "
            f"sha256={MANIFEST_SHA256}, found bytes={len(manifest_bytes)} "
            f"sha256={actual_manifest_sha256}"
        )
    manifest = _strict_json(manifest_bytes, label=MANIFEST_PATH)
    if manifest_bytes != _canonical_json(manifest):
        raise StringSemanticsError("string-semantics manifest is not canonical JSON")
    manifest = _require_exact_keys(
        manifest,
        {
            "artifactCount",
            "artifactKind",
            "artifacts",
            "formatVersion",
            "hashAlgorithm",
            "profileVersion",
            "schema",
            "semanticsId",
        },
        label="string-semantics manifest",
    )
    if (
        manifest["artifactCount"] != 3
        or manifest["artifactKind"] != "aleph-portable-string-semantics-manifest"
        or manifest["formatVersion"] != 1
        or manifest["hashAlgorithm"] != "sha256"
        or manifest["profileVersion"] != "0.3.0-provisional"
        or manifest["semanticsId"] != "python-3.13.2-ucd-15.1-string-semantics-v1"
    ):
        raise StringSemanticsError("string-semantics manifest identity differs")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise StringSemanticsError("string-semantics artifact closure differs")
    schema = _require_exact_keys(
        manifest["schema"],
        {"byteCount", "path", "sha256"},
        label="string-semantics schema record",
    )
    if schema != {
        "byteCount": SCHEMA_BYTE_COUNT,
        "path": SCHEMA_PATH,
        "sha256": SCHEMA_SHA256,
    }:
        raise StringSemanticsError("string-semantics schema identity differs")

    schema_bytes = snapshot.files[SCHEMA_PATH]
    if (
        len(schema_bytes) != SCHEMA_BYTE_COUNT
        or hashlib.sha256(schema_bytes).hexdigest() != SCHEMA_SHA256
    ):
        raise StringSemanticsError("string-semantics schema bytes differ")
    schema_value = _strict_json(schema_bytes, label=SCHEMA_PATH)
    if schema_bytes != _canonical_json(schema_value):
        raise StringSemanticsError("string-semantics schema is not canonical JSON")

    casefold_record = _artifact_by_kind(manifest, "casefold-table")
    whitespace_record = _artifact_by_kind(manifest, "whitespace-table")
    receipt_record = _artifact_by_kind(manifest, "generation-receipt")
    casefold_bytes = _read_snapshot_artifact(
        snapshot, casefold_record, max_bytes=512 * 1024
    )
    whitespace_bytes = _read_snapshot_artifact(
        snapshot, whitespace_record, max_bytes=64 * 1024
    )
    receipt_bytes = _read_snapshot_artifact(
        snapshot, receipt_record, max_bytes=128 * 1024
    )
    casefold_value = _strict_json(casefold_bytes, label=casefold_record["path"])
    whitespace_value = _strict_json(whitespace_bytes, label=whitespace_record["path"])
    if casefold_bytes != _canonical_json(casefold_value):
        raise StringSemanticsError("casefold table is not canonical JSON")
    if whitespace_bytes != _canonical_json(whitespace_value):
        raise StringSemanticsError("whitespace table is not canonical JSON")
    receipt_value = _strict_json(receipt_bytes, label=receipt_record["path"])
    if receipt_bytes != _canonical_json(receipt_value):
        raise StringSemanticsError(
            "string-semantics generation receipt is not canonical JSON"
        )
    _validate_generation_receipt(
        receipt_value,
        casefold_record=casefold_record,
        whitespace_record=whitespace_record,
    )
    mappings = _parse_casefold_table(casefold_value)
    whitespace = _parse_whitespace_table(whitespace_value)
    return FrozenStringSemantics(mappings, whitespace)
