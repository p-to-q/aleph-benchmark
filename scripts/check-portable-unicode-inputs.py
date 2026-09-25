#!/usr/bin/env python3
"""Offline verifier for the reviewed portable Unicode 15.1 inputs.

The verifier uses only the Python standard library. It hashes and inspects the
vendored wheels as ZIP archives; it never installs or imports their extension
modules. There is intentionally no manifest generation or ``--write`` mode.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import stat
import sys
import zipfile
from collections import Counter
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = "bench/vendor/portable-unicode-inputs-v1.json"
SCHEMA_PATH = "bench/vendor/portable-unicode-inputs-v1.schema.json"
SCHEMA_BYTE_COUNT = 6_109
SCHEMA_SHA256 = "81ff92bc02b16ffb88a4c1c31f0ad0c10e4b6bbe79aefafce3b73592dbf8ad05"
UNICODE_LICENSE_PATH = "bench/vendor/unicode/UNICODE-LICENSE-v3.txt"
WHEEL_LICENSE_PATH = "bench/vendor/unicodedata2/Apache-2.0.txt"
DIST_INFO = "unicodedata2-15.1.0.dist-info"

UNICODE_LICENSE = {
    "acquiredAt": "2026-09-25T04:05:23Z",
    "byteCount": 1_995,
    "licenseId": "Unicode-3.0",
    "name": "Unicode License v3",
    "path": UNICODE_LICENSE_PATH,
    "primaryUrl": "https://www.unicode.org/license.txt",
    "sha256": "e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96",
    "verifiedInArtifacts": ["CaseFolding.txt", "NormalizationTest.txt", "PropList.txt"],
}

WHEEL_LICENSE = {
    "acquiredAt": "2026-09-25T04:06:22Z",
    "byteCount": 11_325,
    "licenseId": "Apache-2.0",
    "name": "Apache License 2.0",
    "path": WHEEL_LICENSE_PATH,
    "primaryUrl": (
        "https://files.pythonhosted.org/packages/66/5e/"
        "89234dad6ae00379996e2f1c48081faeab679c830dd7018fc6341925c052/"
        "unicodedata2-15.1.0-cp311-cp311-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl"
    ),
    "sha256": "cb5e8e7e5f4a3988e1063c142c60dc2df75605f4c46515e776e3aca6df976e14",
    "sourceArchiveMember": f"{DIST_INFO}/LICENSE",
    "verifiedInArtifacts": [
        "unicodedata2-15.1.0-cp311-cp311-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl",
        "unicodedata2-15.1.0-cp312-cp312-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl",
    ],
}

EXPECTED_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "acquiredAt": "2026-09-25T04:06:27Z",
        "byteCount": 84_870,
        "filename": "CaseFolding.txt",
        "kind": "unicode-data",
        "licenseId": "Unicode-3.0",
        "primaryUrl": "https://www.unicode.org/Public/15.1.0/ucd/CaseFolding.txt",
        "repositoryPath": "bench/vendor/unicode/15.1.0/ucd/CaseFolding.txt",
        "sha256": "4e55acfdc32825a22e87670e9056a3bf94ad7c5400065778e9e10f8314372bcf",
        "unicodeDataFile": "CaseFolding",
        "unicodeVersion": "15.1.0",
    },
    {
        "acquiredAt": "2026-09-25T04:06:26Z",
        "byteCount": 2_625_136,
        "filename": "NormalizationTest.txt",
        "kind": "unicode-data",
        "licenseId": "Unicode-3.0",
        "primaryUrl": "https://www.unicode.org/Public/15.1.0/ucd/NormalizationTest.txt",
        "repositoryPath": "bench/vendor/unicode/15.1.0/ucd/NormalizationTest.txt",
        "sha256": "871238e37e3be0696ec2bd0891119a041b052da1a84485eda05a5438724b223e",
        "unicodeDataFile": "NormalizationTest",
        "unicodeVersion": "15.1.0",
    },
    {
        "acquiredAt": "2026-09-25T04:06:28Z",
        "byteCount": 136_397,
        "filename": "PropList.txt",
        "kind": "unicode-data",
        "licenseId": "Unicode-3.0",
        "primaryUrl": "https://www.unicode.org/Public/15.1.0/ucd/PropList.txt",
        "repositoryPath": "bench/vendor/unicode/15.1.0/ucd/PropList.txt",
        "sha256": "05672956317b6296bc2ec3d6cef1f6452b57ff4f2efc6dc55b0a19277d5fcfd1",
        "unicodeDataFile": "PropList",
        "unicodeVersion": "15.1.0",
    },
    {
        "abiTag": "cp311",
        "acquiredAt": "2026-09-25T04:06:22Z",
        "byteCount": 471_444,
        "distribution": "unicodedata2",
        "extensionBasename": "unicodedata2.cpython-311-x86_64-linux-gnu.so",
        "filename": (
            "unicodedata2-15.1.0-cp311-cp311-manylinux_2_17_x86_64."
            "manylinux2014_x86_64.whl"
        ),
        "kind": "python-wheel",
        "licenseId": "Apache-2.0",
        "platformTags": [
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
        ],
        "primaryUrl": (
            "https://files.pythonhosted.org/packages/66/5e/"
            "89234dad6ae00379996e2f1c48081faeab679c830dd7018fc6341925c052/"
            "unicodedata2-15.1.0-cp311-cp311-manylinux_2_17_x86_64."
            "manylinux2014_x86_64.whl"
        ),
        "pythonTag": "cp311",
        "repositoryPath": (
            "bench/vendor/unicodedata2/15.1.0/"
            "unicodedata2-15.1.0-cp311-cp311-manylinux_2_17_x86_64."
            "manylinux2014_x86_64.whl"
        ),
        "sha256": "a2442a539d1e493486fdbbdf1d08c8f5d4abe51c9b77b7d39df319a96d30abe4",
        "version": "15.1.0",
    },
    {
        "abiTag": "cp312",
        "acquiredAt": "2026-09-25T04:06:24Z",
        "byteCount": 471_953,
        "distribution": "unicodedata2",
        "extensionBasename": "unicodedata2.cpython-312-x86_64-linux-gnu.so",
        "filename": (
            "unicodedata2-15.1.0-cp312-cp312-manylinux_2_17_x86_64."
            "manylinux2014_x86_64.whl"
        ),
        "kind": "python-wheel",
        "licenseId": "Apache-2.0",
        "platformTags": [
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
        ],
        "primaryUrl": (
            "https://files.pythonhosted.org/packages/c5/6c/"
            "17c5143b33ad44529219187c933d0d4d9de18981b034ac182c71b5318923/"
            "unicodedata2-15.1.0-cp312-cp312-manylinux_2_17_x86_64."
            "manylinux2014_x86_64.whl"
        ),
        "pythonTag": "cp312",
        "repositoryPath": (
            "bench/vendor/unicodedata2/15.1.0/"
            "unicodedata2-15.1.0-cp312-cp312-manylinux_2_17_x86_64."
            "manylinux2014_x86_64.whl"
        ),
        "sha256": "bcdcd195484ba8ef19bef04ec14dcd8b1f8d6560fc517462ee0e010f5f545d75",
        "version": "15.1.0",
    },
)


class PortableInputError(RuntimeError):
    """Raised when a retained portable-profile input fails closed."""


def _canonical_json(value: Any) -> bytes:
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


def _strict_json(data: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise PortableInputError(f"{label} contains non-finite JSON value {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PortableInputError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortableInputError(f"{label} is not strict UTF-8 JSON: {error}") from error


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
        raise PortableInputError(f"unsafe or non-canonical repository path: {raw_path!r}")
    return path


def _read_regular_file(root: Path, raw_path: str, *, max_bytes: int) -> bytes:
    """Read below root without following symlinks in any relative component."""

    relative_path = _validated_relative_path(raw_path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise PortableInputError("this verifier requires O_NOFOLLOW and O_DIRECTORY")
    directory_flags = os.O_RDONLY | directory | nofollow
    root_fd = os.open(root, directory_flags)
    parent_fd = root_fd
    file_fd: int | None = None
    try:
        for part in relative_path.parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            if parent_fd != root_fd:
                os.close(parent_fd)
            parent_fd = next_fd
        file_fd = os.open(relative_path.name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PortableInputError(f"{raw_path} must be a singly linked regular file")
        if before.st_size > max_bytes:
            raise PortableInputError(
                f"{raw_path} exceeds the {max_bytes}-byte verification limit"
            )
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
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PortableInputError(f"{raw_path} changed while it was being verified")
        if len(data) > max_bytes:
            raise PortableInputError(
                f"{raw_path} exceeds the {max_bytes}-byte verification limit"
            )
        return data
    except OSError as error:
        raise PortableInputError(f"cannot safely read {raw_path}: {error}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if parent_fd != root_fd:
            os.close(parent_fd)
        os.close(root_fd)


def _expected_manifest() -> dict[str, Any]:
    return {
        "artifactCount": len(EXPECTED_ARTIFACTS),
        "artifactKind": "aleph-bench-portable-unicode-inputs",
        "artifacts": list(EXPECTED_ARTIFACTS),
        "formatVersion": 1,
        "hashAlgorithm": "sha256",
        "licenses": [WHEEL_LICENSE, UNICODE_LICENSE],
        "profileVersion": "0.3.0-provisional",
        "unicodeVersion": "15.1.0",
    }


def _safe_zip_member_name(raw_name: str) -> None:
    if (
        not raw_name
        or "\\" in raw_name
        or any(ord(character) < 0x20 for character in raw_name)
    ):
        raise PortableInputError(f"unsafe wheel member name: {raw_name!r}")
    path_text = raw_name[:-1] if raw_name.endswith("/") else raw_name
    path = PurePosixPath(path_text)
    if (
        not path_text
        or path.is_absolute()
        or path.as_posix() != path_text
        or ".." in path.parts
        or "." in path.parts
    ):
        raise PortableInputError(f"unsafe wheel member name: {raw_name!r}")


def _singleton_header(message: Any, name: str, expected: str) -> None:
    values = message.get_all(name, [])
    if values != [expected]:
        raise PortableInputError(
            f"wheel {name} metadata must be exactly {expected!r}; found {values!r}"
        )


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def _verify_record(member_bytes: dict[str, bytes], record_path: str) -> None:
    try:
        rows = list(
            csv.reader(io.StringIO(member_bytes[record_path].decode("utf-8"), newline=""))
        )
    except (UnicodeDecodeError, csv.Error) as error:
        raise PortableInputError(f"wheel RECORD is not valid UTF-8 CSV: {error}") from error
    if any(len(row) != 3 for row in rows):
        raise PortableInputError("wheel RECORD rows must contain exactly three columns")
    paths = [row[0] for row in rows]
    if len(paths) != len(set(paths)):
        raise PortableInputError("wheel RECORD contains duplicate paths")
    expected_paths = sorted(name for name in member_bytes if not name.endswith("/"))
    if sorted(paths) != expected_paths:
        raise PortableInputError(
            f"wheel RECORD path closure differs: expected {expected_paths!r}, found {sorted(paths)!r}"
        )
    for path, digest, size in rows:
        _safe_zip_member_name(path)
        if path == record_path:
            if digest or size:
                raise PortableInputError("wheel RECORD must leave its own digest and size empty")
            continue
        data = member_bytes[path]
        expected_digest = f"sha256={_record_digest(data)}"
        if digest != expected_digest or size != str(len(data)):
            raise PortableInputError(
                f"wheel RECORD mismatch for {path}: expected "
                f"{expected_digest!r},{len(data)}, found {digest!r},{size!r}"
            )


def inspect_wheel_bytes(
    wheel_bytes: bytes,
    *,
    artifact: dict[str, Any],
    retained_license: bytes,
) -> dict[str, Any]:
    """Inspect one wheel without extracting, installing, or importing it."""

    extension = artifact["extensionBasename"]
    expected_members = {
        "unicodedata2.libs/",
        f"{DIST_INFO}/",
        extension,
        f"{DIST_INFO}/LICENSE",
        f"{DIST_INFO}/METADATA",
        f"{DIST_INFO}/RECORD",
        f"{DIST_INFO}/WHEEL",
        f"{DIST_INFO}/top_level.txt",
    }
    try:
        archive = zipfile.ZipFile(io.BytesIO(wheel_bytes))
    except (OSError, zipfile.BadZipFile) as error:
        raise PortableInputError(f"wheel is not a valid ZIP archive: {error}") from error

    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicates = sorted(name for name, count in Counter(names).items() if count != 1)
        if duplicates:
            raise PortableInputError(f"wheel contains duplicate members: {duplicates!r}")
        for info in infos:
            _safe_zip_member_name(info.filename)
            if info.create_system != 3:
                raise PortableInputError(
                    f"wheel member lacks Unix mode metadata: {info.filename}"
                )
            if info.flag_bits & 0x1:
                raise PortableInputError(f"encrypted wheel member is forbidden: {info.filename}")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise PortableInputError(
                    f"unsupported wheel compression for {info.filename}: {info.compress_type}"
                )
            unix_type = (info.external_attr >> 16) & 0o170000
            if info.is_dir():
                if unix_type != stat.S_IFDIR:
                    raise PortableInputError(
                        f"wheel directory member type is invalid: {info.filename}"
                    )
            elif unix_type != stat.S_IFREG:
                raise PortableInputError(
                    f"wheel non-directory member is not a regular file: {info.filename}"
                )
            if info.file_size > 2_000_000:
                raise PortableInputError(f"wheel member is too large: {info.filename}")
            if info.filename.endswith("/") != info.is_dir():
                raise PortableInputError(f"wheel directory marker is malformed: {info.filename}")
        if set(names) != expected_members:
            raise PortableInputError(
                "wheel member closure differs: "
                f"expected {sorted(expected_members)!r}, found {sorted(names)!r}"
            )
        if sum(info.file_size for info in infos) > 2_000_000:
            raise PortableInputError("wheel uncompressed content exceeds the verification limit")

        member_bytes: dict[str, bytes] = {}
        try:
            for info in infos:
                member_bytes[info.filename] = b"" if info.is_dir() else archive.read(info)
        except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as error:
            raise PortableInputError(f"wheel member data is invalid: {error}") from error

    metadata = BytesParser(policy=policy.compat32).parsebytes(
        member_bytes[f"{DIST_INFO}/METADATA"]
    )
    _singleton_header(metadata, "Metadata-Version", "2.1")
    _singleton_header(metadata, "Name", "unicodedata2")
    _singleton_header(metadata, "Version", "15.1.0")
    _singleton_header(metadata, "License", "Apache License 2.0")
    _singleton_header(metadata, "License-File", "LICENSE")

    wheel_metadata = BytesParser(policy=policy.compat32).parsebytes(
        member_bytes[f"{DIST_INFO}/WHEEL"]
    )
    _singleton_header(wheel_metadata, "Wheel-Version", "1.0")
    _singleton_header(wheel_metadata, "Root-Is-Purelib", "false")
    expected_tags = [
        f"{artifact['pythonTag']}-{artifact['abiTag']}-{platform_tag}"
        for platform_tag in artifact["platformTags"]
    ]
    if wheel_metadata.get_all("Tag", []) != expected_tags:
        raise PortableInputError(
            f"wheel tags differ: expected {expected_tags!r}, "
            f"found {wheel_metadata.get_all('Tag', [])!r}"
        )
    if member_bytes[f"{DIST_INFO}/top_level.txt"] != b"unicodedata2\n":
        raise PortableInputError("wheel top_level.txt must name only unicodedata2")
    if member_bytes[f"{DIST_INFO}/LICENSE"] != retained_license:
        raise PortableInputError("wheel embedded Apache-2.0 license differs from retained bytes")
    _verify_record(member_bytes, f"{DIST_INFO}/RECORD")
    return {
        "extensionBasename": extension,
        "memberCount": len(infos),
        "tags": expected_tags,
    }


def _validate_schema(manifest: Any, schema: Any) -> None:
    if not isinstance(schema, dict):
        raise PortableInputError(f"{SCHEMA_PATH} must contain a JSON object")
    sys.path.insert(0, str(ROOT))
    try:
        from bench.engine.schema_validation import SchemaValidationError, validate

        validate(manifest, schema)
    except SchemaValidationError as error:
        raise PortableInputError(f"portable input manifest schema validation failed: {error}") from error
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)


def verify(root: Path = ROOT) -> dict[str, Any]:
    manifest_bytes = _read_regular_file(root, MANIFEST_PATH, max_bytes=64 * 1024)
    manifest = _strict_json(manifest_bytes, label=MANIFEST_PATH)
    expected_manifest = _expected_manifest()
    if manifest != expected_manifest:
        raise PortableInputError(
            "portable Unicode input manifest differs from the reviewed supply-chain lock"
        )
    if manifest_bytes != _canonical_json(manifest):
        raise PortableInputError(f"{MANIFEST_PATH} is not canonical JSON")

    schema_bytes = _read_regular_file(root, SCHEMA_PATH, max_bytes=SCHEMA_BYTE_COUNT)
    actual_schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    if (
        len(schema_bytes) != SCHEMA_BYTE_COUNT
        or actual_schema_sha256 != SCHEMA_SHA256
    ):
        raise PortableInputError(
            f"portable input schema differs: expected bytes={SCHEMA_BYTE_COUNT} "
            f"sha256={SCHEMA_SHA256}, found bytes={len(schema_bytes)} "
            f"sha256={actual_schema_sha256}"
        )
    schema = _strict_json(schema_bytes, label=SCHEMA_PATH)
    _validate_schema(manifest, schema)

    licenses: dict[str, bytes] = {}
    for license_record in (WHEEL_LICENSE, UNICODE_LICENSE):
        data = _read_regular_file(
            root,
            license_record["path"],
            max_bytes=license_record["byteCount"],
        )
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if (
            len(data) != license_record["byteCount"]
            or actual_sha256 != license_record["sha256"]
        ):
            raise PortableInputError(
                f"retained {license_record['licenseId']} bytes differ: "
                f"expected bytes={license_record['byteCount']} "
                f"sha256={license_record['sha256']}, found bytes={len(data)} "
                f"sha256={actual_sha256}"
            )
        licenses[license_record["licenseId"]] = data

    wheel_count = 0
    for artifact in EXPECTED_ARTIFACTS:
        data = _read_regular_file(
            root,
            artifact["repositoryPath"],
            max_bytes=artifact["byteCount"],
        )
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if len(data) != artifact["byteCount"] or actual_sha256 != artifact["sha256"]:
            raise PortableInputError(
                f"vendored input differs at {artifact['repositoryPath']}: "
                f"expected bytes={artifact['byteCount']} sha256={artifact['sha256']}, "
                f"found bytes={len(data)} sha256={actual_sha256}"
            )
        if PurePosixPath(artifact["repositoryPath"]).name != artifact["filename"]:
            raise PortableInputError(
                f"repository filename differs from upstream filename: {artifact['repositoryPath']}"
            )
        if artifact["kind"] == "python-wheel":
            wheel_count += 1
            inspect_wheel_bytes(
                data,
                artifact=artifact,
                retained_license=licenses[artifact["licenseId"]],
            )
        else:
            expected_header = (
                f"# {artifact['unicodeDataFile']}-{artifact['unicodeVersion']}.txt\n"
            ).encode("ascii")
            if not data.startswith(expected_header):
                raise PortableInputError(
                    f"Unicode data header differs for {artifact['filename']}"
                )
            if b"https://www.unicode.org/terms_of_use.html" not in data[:2_048]:
                raise PortableInputError(
                    f"Unicode terms reference missing from {artifact['filename']}"
                )

    return {
        "artifactCount": len(EXPECTED_ARTIFACTS),
        "licenseCount": len(licenses),
        "status": "ok",
        "wheelCount": wheel_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify vendored Unicode 15.1 and unicodedata2 inputs offline."
    )
    parser.parse_args(argv)
    try:
        report = verify()
    except PortableInputError as error:
        print(f"portable Unicode inputs: FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
