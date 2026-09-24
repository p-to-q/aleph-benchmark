#!/usr/bin/env python3
"""Verify the content-addressed Aleph Bench E2 + E3a + E3b + E3c source import.

The default gate is offline. ``--source-git`` adds provenance verification
against an already-fetched Git object database. This program never fetches and
never executes code from the source repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SOURCE_REPOSITORY = "https://github.com/p-to-q/aleph"
DESTINATION_REPOSITORY = "https://github.com/p-to-q/aleph-benchmark"
SOURCE_COMMIT = "10abdc1368439d1ab454ee3862a59e14b67a9530"
INVENTORY_COMMIT = "d76206bdaf9ba72730b066560862a3d00cac9699"
INVENTORY_UPSTREAM_PATH = "docs/benchmark/extraction/source-v0.2.inventory.json"
INVENTORY_PATH = "provenance/aleph/source-v0.2.inventory.json"
INVENTORY_GIT_BLOB_SHA1 = "46c42b2764f02d4f38b0e735486423f8eaf1c74d"
INVENTORY_RAW_SHA256 = (
    "827d3c4f42f3abac0e34796f20f8202c5f386eb41f09e78f8434e0c572766ed2"
)
INVENTORY_CLASSIFIED_TREE_SHA256 = (
    "a81a92a2075d3dbfbfa8fec606bf3e6c768d4e0e7e5e9506be80d6f6b5ef2a7e"
)
INVENTORY_SEMANTIC_SHA256 = (
    "9e2bdbdc6bde35293b7967dc7903eb929c428b3df0bf039d2bbf7c8b1ea37791"
)
COPY_TREE_SHA256 = "097b6de4ce08e6bd792043a4263097990bb37c561ac6b822e879f7d77722316d"

E3A_RECEIPT_PATH = "provenance/aleph/e3a.metrics-and-protocol.json"
E3A_RECEIPT_BYTES = 1_834
E3A_RECEIPT_GIT_BLOB_SHA1 = "e11275464e3cdbb94ffb06705637587beeb63606"
E3A_RECEIPT_SHA256 = (
    "a19497bb50640135bb8d5e5f9f3e78ec667d276f4431b5c85f681531c7eed333"
)
E3A_SOURCE_PATHS = ("bench/engine/metrics.py", "bench/README.md")
METRICS_OLD_DOC_PATH = b"docs/benchmark/02-design-spec.md"
METRICS_NEW_DOC_PATH = b"docs/protocol-v0.2.md"

E3B_RECEIPT_PATH = "provenance/aleph/e3b.report.json"
E3B_RECEIPT_BYTES = 4_856
E3B_RECEIPT_GIT_BLOB_SHA1 = "22d6c01d1c3938c194c032ffedd79fc3114fcd0c"
E3B_RECEIPT_SHA256 = (
    "2b915eb7b6a967de8b8f5a80bfe8091c1a6bf0d161cdb7386266b0e6602620df"
)
E3B_SOURCE_PATH = "bench/engine/report.py"
E3B_STANDALONE_BASE_COMMIT = "698fdd26e98544816fc25c220def542bc4750a68"

E3C_RECEIPT_PATH = (
    "provenance/aleph/e3c.platform-package-and-protocol-tests.json"
)
E3C_RECEIPT_BYTES = 94_648
E3C_RECEIPT_GIT_BLOB_SHA1 = "0f7bf4a1979f191c0fa7816bde6da44de40b0725"
E3C_RECEIPT_SHA256 = (
    "237c59563ffccdc5b320e648824133e0fc2ce60dcf455b9fc5ddde50189674ec"
)
E3C_SOURCE_PATHS = (
    "bench/engine/platform_package_v0_2.py",
    "bench/tests/test_protocol_v0_2.py",
)
E3C_STANDALONE_BASE_COMMIT = "e9ad706c1eb1cf26b3a33f4f1dca93419be9a80b"
E3C_REVIEW_ISSUE = "https://github.com/p-to-q/aleph-benchmark/issues/8"
E3C_TRANSFORMATION_ALGORITHM = "utf8-codepoint-offset-splice-sequence-v1"
E3C_PACKAGE_TREE_HASH_ALGORITHM = (
    "sha256-length-framed-path-type-mode-and-content-v1"
)
E3C_PACKAGE_TREE_FILES = 10
E3C_PACKAGE_TREE_SHA256 = (
    "be213ca07782f108815bf94264b4d87f48570c29cbe8511f6346e31296bf0cd8"
)

SOURCE_NOTICE_GIT_BLOB_SHA1 = "be3b6c048fee80545b99e83e0a3e089be1a3ee09"
SOURCE_NOTICE_SHA256 = "c6fefd8d70b629b2fd61ea481793dc227d5e59cf8ba7e44e92fa3eef8fab886f"
REVIEWED_NOTICE_SHA256 = (
    "efd7837f12dd3fc495cf16d0a4b08bbe9085e259cd1bb7e1ea103a7ce83ea0ed"
)
REVIEWED_NOTICE_BYTES = 441

EXPECTED_DISPOSITION_COUNTS = {
    "copy": 78,
    "exclude": 89,
    "port": 6,
    "regenerate": 2,
    "rewrite": 6,
}
EXPECTED_COPY_FILES = 78
EXPECTED_COPY_BYTES = 842_943
EXPECTED_COPY_MODES = {"100644": 77, "100755": 1}
EXPECTED_TRACKED_FILES = 181
CLASSIFIED_TREE_DOMAIN = b"aleph-bench-extraction-inventory-tree-v1\0"
COPY_TREE_DOMAIN = b"aleph-bench-copy-import-tree-v1\0"
MANIFEST_ALGORITHM = "sha256-canonical-json-without-manifest-sha256-v1"
CLASSIFIED_TREE_ALGORITHM = "sha256-length-framed-canonical-file-records-v1"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_PATH = re.compile(r"[A-Za-z0-9._/-]+\Z")
MAX_INVENTORY_BYTES = 4 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_E3C_RECEIPT_BYTES = 512 * 1024
MANAGED_PREFIXES = ("bench/", "docs/", "schemas/", "tests/")
SUPPORT_MANAGED_PATHS = {
    "tests/__init__.py",
    "tests/test_metrics_v0_2.py",
    "tests/test_report_v0_2.py",
    "tests/test_source_v0_2_import.py",
}
EXPECTED_ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "README.md",
    "aleph-bench",
}
EXPECTED_ROOT_DIRECTORIES = {
    ".github",
    "bench",
    "docs",
    "platform",
    "provenance",
    "references",
    "schemas",
    "scripts",
    "tests",
}
OPTIONAL_ROOT_ENTRIES = {".git"}
EXPECTED_PLATFORM_ROOT_ENTRIES = {"m0-mock"}


class ImportCheckError(RuntimeError):
    """Raised when an import receipt, source object, or installed file drifts."""


def _canonical_compact_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_file_json(value: Any) -> bytes:
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


def _reject_json_constant(value: str) -> None:
    raise ImportCheckError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ImportCheckError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _exact_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ImportCheckError(
            f"{field} must be an integer >= {minimum}, found {value!r}"
        )
    return value


def _validate_repository_path(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ImportCheckError(f"unsafe {role} path: {value!r}")
    if not value.isascii() or SAFE_PATH.fullmatch(value) is None:
        raise ImportCheckError(f"unsafe {role} path: {value!r}")
    if unicodedata.normalize("NFC", value) != value:
        raise ImportCheckError(f"non-NFC {role} path: {value!r}")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or any(part.casefold() == ".git" for part in candidate.parts)
    ):
        raise ImportCheckError(f"unsafe {role} path: {value!r}")
    return value


def _validate_unique_paths(paths: Iterable[str], *, role: str) -> None:
    observed: set[str] = set()
    portable: dict[str, str] = {}
    parts: list[tuple[str, tuple[str, ...]]] = []
    for path in paths:
        _validate_repository_path(path, role=role)
        if path in observed:
            raise ImportCheckError(f"duplicate {role} path: {path}")
        observed.add(path)
        key = unicodedata.normalize("NFC", path).casefold()
        if key in portable:
            raise ImportCheckError(
                f"portable {role} path collision: {portable[key]!r} and {path!r}"
            )
        portable[key] = path
        parts.append((path, PurePosixPath(path).parts))
    ordered = sorted(parts, key=lambda item: item[1])
    for index, (path, components) in enumerate(ordered[:-1]):
        next_path, next_components = ordered[index + 1]
        if (
            len(components) < len(next_components)
            and next_components[: len(components)] == components
        ):
            raise ImportCheckError(
                f"{role} file/parent collision: {path!r} and {next_path!r}"
            )


def _records_digest(records: Iterable[dict[str, Any]], *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for record in sorted(records, key=lambda row: row["source"]):
        encoded = _canonical_compact_json(record)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _semantic_manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = copy.deepcopy(manifest)
    try:
        del payload["digests"]["manifestSha256"]
    except (KeyError, TypeError) as exc:
        raise ImportCheckError("inventory lacks digests.manifestSha256") from exc
    return _sha256(_canonical_compact_json(payload))


def _validate_file_record(record: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ImportCheckError(f"files[{index}] must be an object")
    disposition = record.get("disposition")
    expected_keys = {
        "bytes",
        "destination",
        "disposition",
        "gitBlobSha1",
        "mode",
        "sha256",
        "source",
    }
    if disposition == "exclude":
        expected_keys.add("exclusion")
    if set(record) != expected_keys:
        raise ImportCheckError(
            f"files[{index}] keys differ: expected {sorted(expected_keys)}, "
            f"found {sorted(record)}"
        )
    if (
        not isinstance(disposition, str)
        or disposition not in EXPECTED_DISPOSITION_COUNTS
    ):
        raise ImportCheckError(
            f"files[{index}] has unknown disposition: {disposition!r}"
        )
    source = _validate_repository_path(record["source"], role="source")
    destination = record["destination"]
    if disposition == "exclude":
        if destination is not None or not isinstance(record["exclusion"], str):
            raise ImportCheckError(f"files[{index}] has invalid exclusion record")
    else:
        _validate_repository_path(destination, role="destination")
    if (
        not isinstance(record["mode"], str)
        or record["mode"] not in {"100644", "100755"}
    ):
        raise ImportCheckError(
            f"files[{index}] has unsupported mode: {record['mode']!r}"
        )
    if (
        not isinstance(record["gitBlobSha1"], str)
        or HEX40.fullmatch(record["gitBlobSha1"]) is None
    ):
        raise ImportCheckError(f"files[{index}] has invalid Git blob SHA-1")
    if (
        not isinstance(record["sha256"], str)
        or HEX64.fullmatch(record["sha256"]) is None
    ):
        raise ImportCheckError(f"files[{index}] has invalid SHA-256")
    _exact_int(record["bytes"], field=f"files[{index}].bytes")
    if disposition == "copy" and destination != source:
        raise ImportCheckError(
            f"E2 copy must preserve its path exactly: {source!r} -> {destination!r}"
        )
    return record


def _validate_inventory(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ImportCheckError("inventory must contain one JSON object")
    expected_keys = {
        "artifactKind",
        "destination",
        "digests",
        "dispositionCounts",
        "exclusions",
        "files",
        "formatVersion",
        "ownedTree",
        "source",
    }
    if set(manifest) != expected_keys:
        raise ImportCheckError("inventory top-level keys differ from E1")
    if manifest["artifactKind"] != "aleph_bench_extraction_inventory":
        raise ImportCheckError("unexpected inventory artifactKind")
    if type(manifest["formatVersion"]) is not int or manifest["formatVersion"] != 1:
        raise ImportCheckError("inventory formatVersion must be the integer 1")
    expected_source = {
        "commit": SOURCE_COMMIT,
        "gitObjectFormat": "sha1",
        "ref": "refs/heads/benchmark/source-v0.2",
        "repository": SOURCE_REPOSITORY,
    }
    if manifest["source"] != expected_source:
        raise ImportCheckError("inventory source identity differs from reviewed E1")
    if manifest["destination"] != {"repository": DESTINATION_REPOSITORY}:
        raise ImportCheckError(
            "inventory destination identity differs from reviewed E1"
        )
    if manifest["dispositionCounts"] != EXPECTED_DISPOSITION_COUNTS:
        raise ImportCheckError("inventory disposition counts differ from reviewed E1")
    owned_tree = manifest["ownedTree"]
    if (
        not isinstance(owned_tree, dict)
        or type(owned_tree.get("trackedFiles")) is not int
        or owned_tree["trackedFiles"] != EXPECTED_TRACKED_FILES
    ):
        raise ImportCheckError("inventory ownedTree differs from reviewed E1")
    files = manifest["files"]
    if not isinstance(files, list) or len(files) != EXPECTED_TRACKED_FILES:
        raise ImportCheckError(
            "inventory files array must contain exactly 181 records"
        )
    validated = [
        _validate_file_record(row, index=index) for index, row in enumerate(files)
    ]
    sources = [row["source"] for row in validated]
    if sources != sorted(sources):
        raise ImportCheckError("inventory records are not sorted by source")
    _validate_unique_paths(sources, role="source")
    destinations = [
        row["destination"]
        for row in validated
        if row["destination"] is not None
    ]
    _validate_unique_paths(destinations, role="destination")

    counts = dict(
        sorted(Counter(row["disposition"] for row in validated).items())
    )
    if counts != EXPECTED_DISPOSITION_COUNTS:
        raise ImportCheckError(f"computed disposition counts differ: {counts}")
    digests = manifest["digests"]
    expected_digest_keys = {
        "classifiedTreeAlgorithm",
        "classifiedTreeSha256",
        "manifestAlgorithm",
        "manifestSha256",
    }
    if not isinstance(digests, dict) or set(digests) != expected_digest_keys:
        raise ImportCheckError("inventory digest record is invalid")
    if digests["classifiedTreeAlgorithm"] != CLASSIFIED_TREE_ALGORITHM:
        raise ImportCheckError("unexpected classified-tree algorithm")
    if digests["manifestAlgorithm"] != MANIFEST_ALGORITHM:
        raise ImportCheckError("unexpected manifest algorithm")
    computed_tree = _records_digest(validated, domain=CLASSIFIED_TREE_DOMAIN)
    if (
        computed_tree != INVENTORY_CLASSIFIED_TREE_SHA256
        or computed_tree != digests["classifiedTreeSha256"]
    ):
        raise ImportCheckError("classified-tree digest differs from reviewed E1")
    computed_semantic = _semantic_manifest_sha256(manifest)
    if (
        computed_semantic != INVENTORY_SEMANTIC_SHA256
        or computed_semantic != digests["manifestSha256"]
    ):
        raise ImportCheckError("semantic inventory digest differs from reviewed E1")

    copy_rows = [row for row in validated if row["disposition"] == "copy"]
    copy_modes = dict(sorted(Counter(row["mode"] for row in copy_rows).items()))
    if len(copy_rows) != EXPECTED_COPY_FILES:
        raise ImportCheckError("copy selection must contain exactly 78 files")
    if sum(row["bytes"] for row in copy_rows) != EXPECTED_COPY_BYTES:
        raise ImportCheckError(
            "copy selection byte count differs from reviewed E1"
        )
    if copy_modes != EXPECTED_COPY_MODES:
        raise ImportCheckError("copy selection mode counts differ from reviewed E1")
    if _records_digest(copy_rows, domain=COPY_TREE_DOMAIN) != COPY_TREE_SHA256:
        raise ImportCheckError("copy selection digest differs from reviewed E1")
    return copy_rows


def _parse_inventory_bytes(
    raw: bytes, *, expected_raw_sha256: str | None = INVENTORY_RAW_SHA256
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(raw) > MAX_INVENTORY_BYTES:
        raise ImportCheckError("inventory is unreasonably large")
    try:
        manifest = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ImportCheckError(f"invalid inventory JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ImportCheckError("inventory must contain one JSON object")
    if type(manifest.get("formatVersion")) is not int:
        raise ImportCheckError(
            "inventory formatVersion must be an integer, not a boolean"
        )
    if raw != _canonical_file_json(manifest):
        raise ImportCheckError(
            "inventory must be canonical sorted, indented ASCII JSON with one trailing newline"
        )
    if expected_raw_sha256 is not None and _sha256(raw) != expected_raw_sha256:
        raise ImportCheckError(
            "raw inventory SHA-256 differs from the reviewed E1 artifact"
        )
    copy_rows = _validate_inventory(manifest)
    return manifest, copy_rows


def _parse_e3a_receipt_bytes(
    raw: bytes,
    manifest: dict[str, Any],
    *,
    expected_raw_sha256: str | None = E3A_RECEIPT_SHA256,
) -> list[dict[str, Any]]:
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ImportCheckError("E3a receipt is unreasonably large")
    try:
        receipt = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ImportCheckError(f"invalid E3a receipt JSON: {exc}") from exc
    if raw != _canonical_file_json(receipt):
        raise ImportCheckError(
            "E3a receipt must be canonical sorted, indented ASCII JSON with one trailing newline"
        )
    if expected_raw_sha256 is not None and _sha256(raw) != expected_raw_sha256:
        raise ImportCheckError("raw E3a receipt SHA-256 differs from the reviewed artifact")
    expected_top_level = {
        "artifactKind",
        "formatVersion",
        "inventoryCommit",
        "slice",
        "sourceCommit",
        "transformations",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_top_level:
        raise ImportCheckError("E3a receipt top-level keys differ")
    if receipt["artifactKind"] != "aleph_bench_source_migration_receipt":
        raise ImportCheckError("unexpected E3a receipt artifactKind")
    if type(receipt["formatVersion"]) is not int or receipt["formatVersion"] != 1:
        raise ImportCheckError("E3a receipt formatVersion must be the integer 1")
    if receipt["slice"] != "E3a":
        raise ImportCheckError("unexpected migration slice in E3a receipt")
    if receipt["sourceCommit"] != SOURCE_COMMIT:
        raise ImportCheckError("E3a receipt source commit differs")
    if receipt["inventoryCommit"] != INVENTORY_COMMIT:
        raise ImportCheckError("E3a receipt inventory commit differs")

    transformations = receipt["transformations"]
    if not isinstance(transformations, list) or len(transformations) != 2:
        raise ImportCheckError("E3a receipt must contain exactly two transformations")
    inventory_by_source = {row["source"]: row for row in manifest["files"]}
    observed_sources: list[str] = []
    observed_destinations: list[str] = []
    for index, entry in enumerate(transformations):
        if not isinstance(entry, dict) or set(entry) != {
            "destination",
            "output",
            "source",
            "transformation",
        }:
            raise ImportCheckError(f"E3a transformations[{index}] keys differ")
        destination = _validate_repository_path(
            entry["destination"], role="E3a destination"
        )
        source = entry["source"]
        expected_source_keys = {
            "bytes",
            "disposition",
            "gitBlobSha1",
            "mode",
            "path",
            "sha256",
        }
        if not isinstance(source, dict) or set(source) != expected_source_keys:
            raise ImportCheckError(f"E3a transformations[{index}].source keys differ")
        source_path = _validate_repository_path(source["path"], role="E3a source")
        inventory_row = inventory_by_source.get(source_path)
        if inventory_row is None:
            raise ImportCheckError(f"E3a source is absent from inventory: {source_path}")
        expected_source = {
            "bytes": inventory_row["bytes"],
            "disposition": inventory_row["disposition"],
            "gitBlobSha1": inventory_row["gitBlobSha1"],
            "mode": inventory_row["mode"],
            "path": inventory_row["source"],
            "sha256": inventory_row["sha256"],
        }
        if source != expected_source or destination != inventory_row["destination"]:
            raise ImportCheckError(
                f"E3a transformation differs from inventory row: {source_path}"
            )
        output = entry["output"]
        if not isinstance(output, dict) or set(output) != {
            "bytes",
            "gitBlobSha1",
            "mode",
            "sha256",
        }:
            raise ImportCheckError(f"E3a transformations[{index}].output keys differ")
        _exact_int(output["bytes"], field=f"transformations[{index}].output.bytes")
        if output["mode"] != "100644":
            raise ImportCheckError("E3a outputs must use Git mode 100644")
        if not isinstance(output["sha256"], str) or HEX64.fullmatch(output["sha256"]) is None:
            raise ImportCheckError("E3a output has invalid SHA-256")
        if (
            not isinstance(output["gitBlobSha1"], str)
            or HEX40.fullmatch(output["gitBlobSha1"]) is None
        ):
            raise ImportCheckError("E3a output has invalid Git blob SHA-1")
        observed_sources.append(source_path)
        observed_destinations.append(destination)

    if tuple(observed_sources) != E3A_SOURCE_PATHS:
        raise ImportCheckError("E3a source selection or order differs")
    if observed_destinations != ["bench/engine/metrics.py", "docs/protocol-v0.2.md"]:
        raise ImportCheckError("E3a destination selection or order differs")

    metrics_transform = transformations[0]["transformation"]
    if metrics_transform != {
        "algorithm": "utf8-replace-once-v1",
        "replacements": [
            {
                "from": METRICS_OLD_DOC_PATH.decode("ascii"),
                "to": METRICS_NEW_DOC_PATH.decode("ascii"),
            }
        ],
    }:
        raise ImportCheckError("E3a metrics transformation contract differs")
    protocol_transform = transformations[1]["transformation"]
    if protocol_transform != {
        "algorithm": "reviewed-standalone-rewrite-v1",
        "reviewIssue": "https://github.com/p-to-q/aleph-benchmark/issues/4",
    }:
        raise ImportCheckError("E3a protocol rewrite contract differs")
    return transformations


def _parse_e3b_receipt_bytes(
    raw: bytes,
    manifest: dict[str, Any],
    *,
    expected_raw_sha256: str | None = E3B_RECEIPT_SHA256,
) -> list[dict[str, Any]]:
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ImportCheckError("E3b receipt is unreasonably large")
    try:
        receipt = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ImportCheckError(f"invalid E3b receipt JSON: {exc}") from exc
    if raw != _canonical_file_json(receipt):
        raise ImportCheckError(
            "E3b receipt must be canonical sorted, indented ASCII JSON with one trailing newline"
        )
    if expected_raw_sha256 is not None and _sha256(raw) != expected_raw_sha256:
        raise ImportCheckError("raw E3b receipt SHA-256 differs from the reviewed artifact")

    expected_top_level = {
        "artifactKind",
        "formatVersion",
        "inventoryCommit",
        "slice",
        "sourceCommit",
        "standaloneBaseCommit",
        "transformations",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_top_level:
        raise ImportCheckError("E3b receipt top-level keys differ")
    if receipt["artifactKind"] != "aleph_bench_source_migration_receipt":
        raise ImportCheckError("unexpected E3b receipt artifactKind")
    if type(receipt["formatVersion"]) is not int or receipt["formatVersion"] != 1:
        raise ImportCheckError("E3b receipt formatVersion must be the integer 1")
    if receipt["slice"] != "E3b":
        raise ImportCheckError("unexpected migration slice in E3b receipt")
    if receipt["sourceCommit"] != SOURCE_COMMIT:
        raise ImportCheckError("E3b receipt source commit differs")
    if receipt["inventoryCommit"] != INVENTORY_COMMIT:
        raise ImportCheckError("E3b receipt inventory commit differs")
    if receipt["standaloneBaseCommit"] != E3B_STANDALONE_BASE_COMMIT:
        raise ImportCheckError("E3b receipt standalone base commit differs")

    transformations = receipt["transformations"]
    if not isinstance(transformations, list) or len(transformations) != 1:
        raise ImportCheckError("E3b receipt must contain exactly one transformation")
    entry = transformations[0]
    if not isinstance(entry, dict) or set(entry) != {
        "destination",
        "output",
        "source",
        "transformation",
    }:
        raise ImportCheckError("E3b transformation keys differ")
    destination = _validate_repository_path(
        entry["destination"], role="E3b destination"
    )
    source = entry["source"]
    expected_source_keys = {
        "bytes",
        "disposition",
        "gitBlobSha1",
        "mode",
        "path",
        "sha256",
    }
    if not isinstance(source, dict) or set(source) != expected_source_keys:
        raise ImportCheckError("E3b transformation source keys differ")
    source_path = _validate_repository_path(source["path"], role="E3b source")
    inventory_by_source = {row["source"]: row for row in manifest["files"]}
    inventory_row = inventory_by_source.get(source_path)
    if inventory_row is None:
        raise ImportCheckError(f"E3b source is absent from inventory: {source_path}")
    expected_source = {
        "bytes": inventory_row["bytes"],
        "disposition": inventory_row["disposition"],
        "gitBlobSha1": inventory_row["gitBlobSha1"],
        "mode": inventory_row["mode"],
        "path": inventory_row["source"],
        "sha256": inventory_row["sha256"],
    }
    if (
        source_path != E3B_SOURCE_PATH
        or source != expected_source
        or destination != inventory_row["destination"]
        or destination != E3B_SOURCE_PATH
    ):
        raise ImportCheckError("E3b report transformation differs from inventory")

    output = entry["output"]
    if not isinstance(output, dict) or set(output) != {
        "bytes",
        "gitBlobSha1",
        "mode",
        "sha256",
    }:
        raise ImportCheckError("E3b transformation output keys differ")
    _exact_int(output["bytes"], field="E3b transformation output.bytes")
    if output["mode"] != "100644":
        raise ImportCheckError("E3b output must use Git mode 100644")
    if not isinstance(output["sha256"], str) or HEX64.fullmatch(output["sha256"]) is None:
        raise ImportCheckError("E3b output has invalid SHA-256")
    if (
        not isinstance(output["gitBlobSha1"], str)
        or HEX40.fullmatch(output["gitBlobSha1"]) is None
    ):
        raise ImportCheckError("E3b output has invalid Git blob SHA-1")

    transformation = entry["transformation"]
    if not isinstance(transformation, dict) or set(transformation) != {
        "algorithm",
        "replacements",
        "reviewIssue",
    }:
        raise ImportCheckError("E3b report transformation contract keys differ")
    if transformation["algorithm"] != "utf8-replace-once-sequence-v1":
        raise ImportCheckError("E3b report transformation algorithm differs")
    if transformation["reviewIssue"] != (
        "https://github.com/p-to-q/aleph-benchmark/issues/6"
    ):
        raise ImportCheckError("E3b report review issue differs")
    replacements = transformation["replacements"]
    if not isinstance(replacements, list) or len(replacements) != 5:
        raise ImportCheckError("E3b report must declare exactly five replacements")
    for index, replacement in enumerate(replacements):
        if not isinstance(replacement, dict) or set(replacement) != {"from", "to"}:
            raise ImportCheckError(f"E3b replacement[{index}] keys differ")
        before = replacement["from"]
        after = replacement["to"]
        if (
            not isinstance(before, str)
            or not isinstance(after, str)
            or not before
            or before == after
        ):
            raise ImportCheckError(f"E3b replacement[{index}] is invalid")
        try:
            before.encode("utf-8")
            after.encode("utf-8")
        except UnicodeError as exc:
            raise ImportCheckError(
                f"E3b replacement[{index}] is not valid UTF-8"
            ) from exc
    return transformations


def _validate_e3c_splice_transformation(
    transformation: Any, *, role: str
) -> None:
    if not isinstance(transformation, dict) or set(transformation) != {
        "algorithm",
        "edits",
        "reviewIssue",
    }:
        raise ImportCheckError(f"{role} transformation contract keys differ")
    if transformation["algorithm"] != E3C_TRANSFORMATION_ALGORITHM:
        raise ImportCheckError(f"{role} transformation algorithm differs")
    if transformation["reviewIssue"] != E3C_REVIEW_ISSUE:
        raise ImportCheckError(f"{role} review issue differs")

    edits = transformation["edits"]
    if not isinstance(edits, list) or not edits:
        raise ImportCheckError(f"{role} must declare at least one splice edit")
    source_cursor = 0
    output_cursor = 0
    cumulative_delta = 0
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict) or set(edit) != {
            "from",
            "outputCharOffset",
            "sourceCharOffset",
            "to",
        }:
            raise ImportCheckError(f"{role} edit[{index}] keys differ")
        source_offset = _exact_int(
            edit["sourceCharOffset"],
            field=f"{role} edit[{index}].sourceCharOffset",
        )
        output_offset = _exact_int(
            edit["outputCharOffset"],
            field=f"{role} edit[{index}].outputCharOffset",
        )
        before = edit["from"]
        after = edit["to"]
        if not isinstance(before, str) or not isinstance(after, str):
            raise ImportCheckError(f"{role} edit[{index}] payloads must be strings")
        if before == after:
            raise ImportCheckError(f"{role} edit[{index}] must change the payload")
        try:
            before.encode("utf-8")
            after.encode("utf-8")
        except UnicodeError as exc:
            raise ImportCheckError(
                f"{role} edit[{index}] payload is not valid UTF-8"
            ) from exc
        if source_offset < source_cursor or output_offset < output_cursor:
            raise ImportCheckError(f"{role} splice edits overlap or are out of order")
        if output_offset != source_offset + cumulative_delta:
            raise ImportCheckError(
                f"{role} edit[{index}] source/output offsets are inconsistent"
            )
        source_cursor = source_offset + len(before)
        output_cursor = output_offset + len(after)
        cumulative_delta += len(after) - len(before)


def _apply_e3c_splice_transformation(
    payload: bytes,
    transformation: dict[str, Any],
    *,
    reverse: bool,
    role: str,
) -> bytes:
    _validate_e3c_splice_transformation(transformation, role=role)
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ImportCheckError(f"{role} input is not valid UTF-8") from exc

    chunks: list[str] = []
    cursor = 0
    offset_key = "outputCharOffset" if reverse else "sourceCharOffset"
    before_key = "to" if reverse else "from"
    after_key = "from" if reverse else "to"
    direction = "reverse" if reverse else "forward"
    for index, edit in enumerate(transformation["edits"]):
        offset = edit[offset_key]
        before = edit[before_key]
        after = edit[after_key]
        if offset < cursor or text[offset : offset + len(before)] != before:
            raise ImportCheckError(
                f"{role} {direction} splice[{index}] does not match its input"
            )
        chunks.append(text[cursor:offset])
        chunks.append(after)
        cursor = offset + len(before)
    chunks.append(text[cursor:])
    return "".join(chunks).encode("utf-8")


def _parse_e3c_receipt_bytes(
    raw: bytes,
    manifest: dict[str, Any],
    *,
    expected_raw_sha256: str | None = E3C_RECEIPT_SHA256,
) -> list[dict[str, Any]]:
    if len(raw) > MAX_E3C_RECEIPT_BYTES:
        raise ImportCheckError("E3c receipt is unreasonably large")
    try:
        receipt = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ImportCheckError(f"invalid E3c receipt JSON: {exc}") from exc
    if raw != _canonical_file_json(receipt):
        raise ImportCheckError(
            "E3c receipt must be canonical sorted, indented ASCII JSON with one trailing newline"
        )
    if expected_raw_sha256 is not None and _sha256(raw) != expected_raw_sha256:
        raise ImportCheckError(
            "raw E3c receipt SHA-256 differs from the reviewed artifact"
        )

    expected_top_level = {
        "artifactKind",
        "formatVersion",
        "inventoryCommit",
        "packageTree",
        "slice",
        "sourceCommit",
        "standaloneBaseCommit",
        "transformations",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_top_level:
        raise ImportCheckError("E3c receipt top-level keys differ")
    if receipt["artifactKind"] != "aleph_bench_source_migration_receipt":
        raise ImportCheckError("unexpected E3c receipt artifactKind")
    if type(receipt["formatVersion"]) is not int or receipt["formatVersion"] != 1:
        raise ImportCheckError("E3c receipt formatVersion must be the integer 1")
    if receipt["slice"] != "E3c":
        raise ImportCheckError("unexpected migration slice in E3c receipt")
    if receipt["sourceCommit"] != SOURCE_COMMIT:
        raise ImportCheckError("E3c receipt source commit differs")
    if receipt["inventoryCommit"] != INVENTORY_COMMIT:
        raise ImportCheckError("E3c receipt inventory commit differs")
    if receipt["standaloneBaseCommit"] != E3C_STANDALONE_BASE_COMMIT:
        raise ImportCheckError("E3c receipt standalone base commit differs")

    package_tree = receipt["packageTree"]
    if not isinstance(package_tree, dict) or set(package_tree) != {
        "algorithm",
        "fileCount",
        "sha256",
    }:
        raise ImportCheckError("E3c packageTree contract keys differ")
    if package_tree != {
        "algorithm": E3C_PACKAGE_TREE_HASH_ALGORITHM,
        "fileCount": E3C_PACKAGE_TREE_FILES,
        "sha256": E3C_PACKAGE_TREE_SHA256,
    }:
        raise ImportCheckError("E3c package tree contract differs")

    transformations = receipt["transformations"]
    if not isinstance(transformations, list) or len(transformations) != 2:
        raise ImportCheckError("E3c receipt must contain exactly two transformations")
    inventory_by_source = {row["source"]: row for row in manifest["files"]}
    observed_sources: list[str] = []
    observed_destinations: list[str] = []
    for index, entry in enumerate(transformations):
        role = f"E3c transformations[{index}]"
        if not isinstance(entry, dict) or set(entry) != {
            "destination",
            "output",
            "source",
            "transformation",
        }:
            raise ImportCheckError(f"{role} keys differ")
        destination = _validate_repository_path(
            entry["destination"], role="E3c destination"
        )
        source = entry["source"]
        expected_source_keys = {
            "bytes",
            "disposition",
            "gitBlobSha1",
            "mode",
            "path",
            "sha256",
        }
        if not isinstance(source, dict) or set(source) != expected_source_keys:
            raise ImportCheckError(f"{role}.source keys differ")
        source_path = _validate_repository_path(source["path"], role="E3c source")
        inventory_row = inventory_by_source.get(source_path)
        if inventory_row is None:
            raise ImportCheckError(f"E3c source is absent from inventory: {source_path}")
        expected_source = {
            "bytes": inventory_row["bytes"],
            "disposition": inventory_row["disposition"],
            "gitBlobSha1": inventory_row["gitBlobSha1"],
            "mode": inventory_row["mode"],
            "path": inventory_row["source"],
            "sha256": inventory_row["sha256"],
        }
        if source != expected_source or destination != inventory_row["destination"]:
            raise ImportCheckError(
                f"E3c transformation differs from inventory row: {source_path}"
            )
        if source["disposition"] != "port" or source["mode"] != "100644":
            raise ImportCheckError(f"E3c source must be a reviewed 100644 port: {source_path}")

        output = entry["output"]
        if not isinstance(output, dict) or set(output) != {
            "bytes",
            "gitBlobSha1",
            "mode",
            "sha256",
        }:
            raise ImportCheckError(f"{role}.output keys differ")
        _exact_int(output["bytes"], field=f"{role}.output.bytes", minimum=1)
        if output["mode"] != "100644":
            raise ImportCheckError("E3c outputs must use Git mode 100644")
        if (
            not isinstance(output["sha256"], str)
            or HEX64.fullmatch(output["sha256"]) is None
        ):
            raise ImportCheckError(f"{role} output has invalid SHA-256")
        if (
            not isinstance(output["gitBlobSha1"], str)
            or HEX40.fullmatch(output["gitBlobSha1"]) is None
        ):
            raise ImportCheckError(f"{role} output has invalid Git blob SHA-1")
        _validate_e3c_splice_transformation(
            entry["transformation"], role=role
        )
        observed_sources.append(source_path)
        observed_destinations.append(destination)

    if tuple(observed_sources) != E3C_SOURCE_PATHS:
        raise ImportCheckError("E3c source selection or order differs")
    if tuple(observed_destinations) != E3C_SOURCE_PATHS:
        raise ImportCheckError("E3c destination selection or order differs")
    return transformations


def _require_secure_io() -> None:
    if (
        os.name != "posix"
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise ImportCheckError(
            "secure import verification requires POSIX O_NOFOLLOW/O_DIRECTORY"
        )


def _open_parent_no_follow(root: Path, relative: str) -> tuple[int, str]:
    _require_secure_io()
    _validate_repository_path(relative, role="installed")
    try:
        root_status = os.lstat(root)
    except OSError as exc:
        raise ImportCheckError(
            f"could not lstat repository root {root}: {exc}"
        ) from exc
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
        raise ImportCheckError(f"repository root must be a real directory: {root}")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(root, flags)
        parts = PurePosixPath(relative).parts
        for component in parts[:-1]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ImportCheckError(
            f"could not securely open parent of {relative}: {exc}"
        ) from exc


def _read_regular_path(
    root: Path,
    relative: str,
    *,
    expected_bytes: int | None = None,
    max_bytes: int | None = None,
) -> tuple[bytes, os.stat_result]:
    parent_fd, name = _open_parent_no_follow(root, relative)
    file_fd: int | None = None
    try:
        try:
            expected = os.lstat(name, dir_fd=parent_fd)
        except OSError as exc:
            raise ImportCheckError(
                f"could not lstat installed file {relative}: {exc}"
            ) from exc
        if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode):
            raise ImportCheckError(
                f"installed path must be a regular file: {relative}"
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        try:
            file_fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ImportCheckError(
                f"could not securely open {relative}: {exc}"
            ) from exc
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ImportCheckError(
                f"installed path must be a regular file: {relative}"
            )
        if (expected.st_dev, expected.st_ino) != (opened.st_dev, opened.st_ino):
            raise ImportCheckError(f"installed file changed while opening: {relative}")
        if expected_bytes is not None and opened.st_size != expected_bytes:
            raise ImportCheckError(
                f"installed byte count differs at {relative}: "
                f"{opened.st_size} != {expected_bytes}"
            )
        if max_bytes is not None and opened.st_size > max_bytes:
            raise ImportCheckError(
                f"installed file exceeds the read limit at {relative}"
            )
        chunks: list[bytes] = []
        observed_bytes = 0
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            observed_bytes += len(chunk)
            if max_bytes is not None and observed_bytes > max_bytes:
                raise ImportCheckError(
                    f"installed file exceeded the read limit at {relative}"
                )
        after = os.fstat(file_fd)

        def snapshot(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if snapshot(opened) != snapshot(after):
            raise ImportCheckError(f"installed file changed while reading: {relative}")
        return b"".join(chunks), opened
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _git_mode_from_stat(status: os.stat_result, *, path: str) -> str:
    if status.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        raise ImportCheckError(f"special permission bits are forbidden: {path}")
    execute_bits = status.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if execute_bits and not status.st_mode & stat.S_IXUSR:
        raise ImportCheckError(
            f"non-canonical execute permissions are forbidden: {path}"
        )
    return "100755" if execute_bits else "100644"


def _verify_root_entries(root: Path) -> None:
    expected = EXPECTED_ROOT_FILES | EXPECTED_ROOT_DIRECTORIES
    try:
        entries = {entry.name: entry for entry in os.scandir(root)}
    except OSError as exc:
        raise ImportCheckError(f"could not scan repository root {root}: {exc}") from exc
    actual = set(entries) - OPTIONAL_ROOT_ENTRIES
    if actual != expected:
        raise ImportCheckError(
            "repository root entry set differs: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for name in sorted(expected):
        status = entries[name].stat(follow_symlinks=False)
        if stat.S_ISLNK(status.st_mode):
            raise ImportCheckError(f"repository root symlink is forbidden: {name}")
        if name in EXPECTED_ROOT_FILES and not stat.S_ISREG(status.st_mode):
            raise ImportCheckError(f"repository root file is not regular: {name}")
        if name in EXPECTED_ROOT_DIRECTORIES and not stat.S_ISDIR(status.st_mode):
            raise ImportCheckError(f"repository root directory is not real: {name}")
    git_entry = entries.get(".git")
    if git_entry is not None:
        status = git_entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(status.st_mode) or not (
            stat.S_ISREG(status.st_mode) or stat.S_ISDIR(status.st_mode)
        ):
            raise ImportCheckError("optional .git entry has an unsafe type")
    try:
        platform_entries = {
            entry.name: entry for entry in os.scandir(root / "platform")
        }
    except OSError as exc:
        raise ImportCheckError(f"could not scan platform root: {exc}") from exc
    if set(platform_entries) != EXPECTED_PLATFORM_ROOT_ENTRIES:
        raise ImportCheckError(
            "platform root entry set differs; unexpected entries could shadow the standard library"
        )
    platform_snapshot = platform_entries["m0-mock"].stat(follow_symlinks=False)
    if not stat.S_ISDIR(platform_snapshot.st_mode):
        raise ImportCheckError("platform/m0-mock must be a real directory")


def _scan_closed_tree(root: Path, prefix: str) -> tuple[set[str], set[str]]:
    root_relative = prefix.rstrip("/")
    start = root / root_relative
    try:
        status = os.lstat(start)
    except OSError as exc:
        raise ImportCheckError(
            f"missing managed directory {root_relative}: {exc}"
        ) from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ImportCheckError(
            f"managed root must be a real directory: {root_relative}"
        )
    files: set[str] = set()
    directories: set[str] = {root_relative}
    pending = [start]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ImportCheckError(
                f"could not scan managed directory {directory}: {exc}"
            ) from exc
        for entry in entries:
            entry_status = entry.stat(follow_symlinks=False)
            relative = Path(entry.path).relative_to(root).as_posix()
            _validate_repository_path(relative, role="installed")
            if stat.S_ISLNK(entry_status.st_mode):
                raise ImportCheckError(
                    f"symlink is forbidden in managed tree: {relative}"
                )
            if stat.S_ISDIR(entry_status.st_mode):
                directories.add(relative)
                pending.append(Path(entry.path))
            elif stat.S_ISREG(entry_status.st_mode):
                files.add(relative)
            else:
                raise ImportCheckError(
                    f"non-regular path is forbidden in managed tree: {relative}"
                )
    return files, directories


def _path_or_parent_exists(root: Path, relative: str) -> bool:
    current = root
    for component in PurePosixPath(relative).parts:
        current /= component
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ImportCheckError(
                f"could not inspect forbidden path {relative}: {exc}"
            ) from exc
        if stat.S_ISLNK(status.st_mode):
            return True
    return True


def _expected_directories(paths: Iterable[str], *, prefix: str) -> set[str]:
    root_relative = prefix.rstrip("/")
    result = {root_relative}
    for path in paths:
        if not path.startswith(prefix):
            continue
        parent = PurePosixPath(path).parent
        while (
            parent.as_posix() != "."
            and parent.as_posix().startswith(root_relative)
        ):
            result.add(parent.as_posix())
            if parent.as_posix() == root_relative:
                break
            parent = parent.parent
    return result


def _verify_installed(
    root: Path,
    manifest: dict[str, Any],
    copy_rows: list[dict[str, Any]],
    transformations: list[dict[str, Any]],
) -> None:
    expected_copy_paths = {row["destination"] for row in copy_rows}
    transformed_paths = {entry["destination"] for entry in transformations}
    expected_managed_paths = (
        expected_copy_paths | transformed_paths | SUPPORT_MANAGED_PATHS
    )
    for prefix in MANAGED_PREFIXES:
        actual_files, actual_directories = _scan_closed_tree(root, prefix)
        expected_files = {
            path for path in expected_managed_paths if path.startswith(prefix)
        }
        expected_directories = _expected_directories(expected_files, prefix=prefix)
        if actual_files != expected_files:
            raise ImportCheckError(
                f"managed file set differs under {prefix}: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}"
            )
        if actual_directories != expected_directories:
            raise ImportCheckError(
                f"managed directory set differs under {prefix}: "
                f"missing={sorted(expected_directories - actual_directories)}, "
                f"extra={sorted(actual_directories - expected_directories)}"
            )

    forbidden = {
        row["destination"]
        for row in manifest["files"]
        if row["disposition"] in {"port", "regenerate", "rewrite"}
        and row["destination"] != "NOTICE"
        and row["destination"] not in transformed_paths
    }
    for path in sorted(forbidden):
        if _path_or_parent_exists(root, path):
            raise ImportCheckError(
                f"non-copy E1 destination is forbidden in E2: {path}"
            )

    for row in copy_rows:
        path = row["destination"]
        payload, status = _read_regular_path(
            root, path, expected_bytes=row["bytes"], max_bytes=row["bytes"]
        )
        observed_mode = _git_mode_from_stat(status, path=path)
        if len(payload) != row["bytes"]:
            raise ImportCheckError(f"installed byte count differs: {path}")
        if _sha256(payload) != row["sha256"]:
            raise ImportCheckError(f"installed SHA-256 differs: {path}")
        if _git_blob_sha1(payload) != row["gitBlobSha1"]:
            raise ImportCheckError(f"installed Git blob SHA-1 differs: {path}")
        if observed_mode != row["mode"]:
            raise ImportCheckError(
                f"installed Git mode differs at {path}: "
                f"{observed_mode} != {row['mode']}"
            )

    for entry in transformations:
        path = entry["destination"]
        output = entry["output"]
        payload, status = _read_regular_path(
            root,
            path,
            expected_bytes=output["bytes"],
            max_bytes=output["bytes"],
        )
        if _sha256(payload) != output["sha256"]:
            raise ImportCheckError(f"installed transformed SHA-256 differs: {path}")
        if _git_blob_sha1(payload) != output["gitBlobSha1"]:
            raise ImportCheckError(
                f"installed transformed Git blob SHA-1 differs: {path}"
            )
        observed_mode = _git_mode_from_stat(status, path=path)
        if observed_mode != output["mode"]:
            raise ImportCheckError(
                f"installed transformed Git mode differs at {path}: "
                f"{observed_mode} != {output['mode']}"
            )

    notice, notice_status = _read_regular_path(
        root,
        "NOTICE",
        expected_bytes=REVIEWED_NOTICE_BYTES,
        max_bytes=REVIEWED_NOTICE_BYTES,
    )
    if _sha256(notice) != REVIEWED_NOTICE_SHA256:
        raise ImportCheckError("reviewed standalone NOTICE SHA-256 differs")
    if _git_mode_from_stat(notice_status, path="NOTICE") != "100644":
        raise ImportCheckError("standalone NOTICE must have Git mode 100644")
    if _sha256(notice) == SOURCE_NOTICE_SHA256:
        raise ImportCheckError(
            "standalone NOTICE must be the reviewed rewrite, not a copy"
        )


def _git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    return environment


def _git(repository: Path, arguments: list[str]) -> bytes:
    result = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-c",
            "core.hooksPath=/dev/null",
            *arguments,
        ],
        cwd=repository,
        check=False,
        env=_git_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ImportCheckError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _verify_commit(repository: Path, commit: str) -> None:
    resolved = _git(
        repository, ["rev-parse", "--verify", f"{commit}^{{commit}}"]
    ).decode("ascii").strip()
    if resolved != commit:
        raise ImportCheckError(
            f"source commit did not resolve exactly: {resolved!r}"
        )
    object_type = _git(repository, ["cat-file", "-t", commit]).decode(
        "ascii"
    ).strip()
    if object_type != "commit":
        raise ImportCheckError(f"source object is not a commit: {commit}")


def _parse_ls_tree(raw: bytes) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    records = raw.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    for record in records:
        try:
            header, encoded_path = record.split(b"\t", 1)
            mode_raw, object_type_raw, oid_raw, size_raw = header.split()
            path = encoded_path.decode("utf-8")
            mode = mode_raw.decode("ascii")
            object_type = object_type_raw.decode("ascii")
            oid = oid_raw.decode("ascii")
            size_text = size_raw.decode("ascii")
            size = None if size_text == "-" else int(size_text)
        except (UnicodeError, ValueError) as exc:
            raise ImportCheckError(
                f"could not parse git ls-tree record: {record!r}"
            ) from exc
        _validate_repository_path(path, role="Git tree")
        if path in entries:
            raise ImportCheckError(f"duplicate Git tree path: {path}")
        entries[path] = {
            "mode": mode,
            "objectType": object_type,
            "oid": oid,
            "bytes": size,
        }
    return entries


def _verify_rows_at_commit(
    repository: Path, commit: str, rows: Iterable[dict[str, Any]]
) -> dict[str, bytes]:
    tree = _parse_ls_tree(
        _git(
            repository,
            ["ls-tree", "-rz", "-r", "--full-tree", "-l", commit],
        )
    )
    payloads: dict[str, bytes] = {}
    for row in rows:
        path = row["source"]
        entry = tree.get(path)
        if entry is None:
            raise ImportCheckError(
                f"source path is missing from pinned commit: {path}"
            )
        if (
            entry["objectType"] != "blob"
            or entry["mode"] not in {"100644", "100755"}
        ):
            raise ImportCheckError(
                f"source path is not a supported regular blob: {path} "
                f"({entry['mode']} {entry['objectType']})"
            )
        if entry["mode"] != row["mode"]:
            raise ImportCheckError(f"source Git mode differs at {path}")
        if entry["oid"] != row["gitBlobSha1"]:
            raise ImportCheckError(f"source Git blob SHA-1 differs at {path}")
        if entry["bytes"] != row["bytes"]:
            raise ImportCheckError(f"source byte count differs at {path}")
        payload = _git(repository, ["cat-file", "blob", entry["oid"]])
        if len(payload) != row["bytes"]:
            raise ImportCheckError(f"source blob byte count differs at {path}")
        if _git_blob_sha1(payload) != row["gitBlobSha1"]:
            raise ImportCheckError(f"source blob Git SHA-1 differs at {path}")
        if _sha256(payload) != row["sha256"]:
            raise ImportCheckError(f"source blob SHA-256 differs at {path}")
        payloads[path] = payload
    return payloads


def _verify_inventory_object(repository: Path, installed_raw: bytes) -> None:
    raw_tree = _git(
        repository,
        [
            "ls-tree",
            "-z",
            "-l",
            INVENTORY_COMMIT,
            "--",
            INVENTORY_UPSTREAM_PATH,
        ],
    )
    tree = _parse_ls_tree(raw_tree)
    if set(tree) != {INVENTORY_UPSTREAM_PATH}:
        raise ImportCheckError(
            "pinned inventory tree entry is missing or ambiguous"
        )
    entry = tree[INVENTORY_UPSTREAM_PATH]
    if entry["objectType"] != "blob" or entry["mode"] != "100644":
        raise ImportCheckError("pinned inventory object is not a 100644 blob")
    if entry["oid"] != INVENTORY_GIT_BLOB_SHA1:
        raise ImportCheckError("pinned inventory Git blob SHA-1 differs")
    raw = _git(repository, ["cat-file", "blob", entry["oid"]])
    if raw != installed_raw or _sha256(raw) != INVENTORY_RAW_SHA256:
        raise ImportCheckError(
            "installed inventory differs from pinned upstream object"
        )
    if _git_blob_sha1(raw) != INVENTORY_GIT_BLOB_SHA1:
        raise ImportCheckError(
            "pinned inventory blob failed Git SHA-1 verification"
        )


def _validate_source_root(repository: Path) -> Path:
    try:
        status = os.lstat(repository)
    except OSError as exc:
        raise ImportCheckError(f"could not lstat source Git directory: {exc}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ImportCheckError(
            "source Git path must be a real directory, not a symlink"
        )
    return repository


def _verify_e3a_transformations_at_source(
    source_payloads: dict[str, bytes],
    root: Path,
    transformations: list[dict[str, Any]],
) -> None:
    metrics_source = source_payloads["bench/engine/metrics.py"]
    if metrics_source.count(METRICS_OLD_DOC_PATH) != 1:
        raise ImportCheckError(
            "pinned metrics source must contain the old documentation path exactly once"
        )
    if METRICS_NEW_DOC_PATH in metrics_source:
        raise ImportCheckError(
            "pinned metrics source already contains the standalone documentation path"
        )
    expected_metrics = metrics_source.replace(
        METRICS_OLD_DOC_PATH, METRICS_NEW_DOC_PATH, 1
    )
    metrics_entry = transformations[0]
    installed_metrics, _ = _read_regular_path(
        root,
        metrics_entry["destination"],
        expected_bytes=metrics_entry["output"]["bytes"],
        max_bytes=metrics_entry["output"]["bytes"],
    )
    if installed_metrics != expected_metrics:
        raise ImportCheckError(
            "installed metrics port differs from the one reviewed path replacement"
        )


def _verify_e3b_transformations_at_source(
    source_payloads: dict[str, bytes],
    root: Path,
    transformations: list[dict[str, Any]],
) -> None:
    entry = transformations[0]
    expected = source_payloads[E3B_SOURCE_PATH]
    for index, replacement in enumerate(
        entry["transformation"]["replacements"]
    ):
        before = replacement["from"].encode("utf-8")
        after = replacement["to"].encode("utf-8")
        if expected.count(before) != 1:
            raise ImportCheckError(
                f"pinned report source does not match E3b replacement[{index}] exactly once"
            )
        expected = expected.replace(before, after, 1)

    installed, _ = _read_regular_path(
        root,
        entry["destination"],
        expected_bytes=entry["output"]["bytes"],
        max_bytes=entry["output"]["bytes"],
    )
    if installed != expected:
        raise ImportCheckError(
            "installed report port differs from the reviewed replacement sequence"
        )


def _reconstruct_e3c_sources(
    root: Path, transformations: list[dict[str, Any]]
) -> dict[str, bytes]:
    reconstructed: dict[str, bytes] = {}
    for index, entry in enumerate(transformations):
        role = f"E3c transformations[{index}]"
        output = entry["output"]
        installed, _ = _read_regular_path(
            root,
            entry["destination"],
            expected_bytes=output["bytes"],
            max_bytes=output["bytes"],
        )
        if (
            len(installed) != output["bytes"]
            or _sha256(installed) != output["sha256"]
            or _git_blob_sha1(installed) != output["gitBlobSha1"]
        ):
            raise ImportCheckError(f"{role} installed output metadata differs")
        source_payload = _apply_e3c_splice_transformation(
            installed,
            entry["transformation"],
            reverse=True,
            role=role,
        )
        source = entry["source"]
        if (
            len(source_payload) != source["bytes"]
            or _sha256(source_payload) != source["sha256"]
            or _git_blob_sha1(source_payload) != source["gitBlobSha1"]
        ):
            raise ImportCheckError(
                f"{role} reverse replay does not reconstruct the pinned source"
            )
        replayed = _apply_e3c_splice_transformation(
            source_payload,
            entry["transformation"],
            reverse=False,
            role=role,
        )
        if replayed != installed:
            raise ImportCheckError(
                f"{role} forward replay does not reconstruct the installed output"
            )
        reconstructed[source["path"]] = source_payload
    return reconstructed


def _verify_e3c_transformations_at_source(
    source_payloads: dict[str, bytes],
    root: Path,
    transformations: list[dict[str, Any]],
) -> None:
    reconstructed = _reconstruct_e3c_sources(root, transformations)
    for source_path in E3C_SOURCE_PATHS:
        if reconstructed[source_path] != source_payloads[source_path]:
            raise ImportCheckError(
                f"installed E3c port does not reconstruct pinned source: {source_path}"
            )


def _verify_source(
    repository: Path,
    root: Path,
    manifest: dict[str, Any],
    copy_rows: list[dict[str, Any]],
    e3a_transformations: list[dict[str, Any]],
    e3b_transformations: list[dict[str, Any]],
    e3c_transformations: list[dict[str, Any]],
    installed_inventory_raw: bytes,
) -> None:
    repository = _validate_source_root(repository)
    object_format = _git(
        repository, ["rev-parse", "--show-object-format"]
    ).decode("ascii").strip()
    if object_format != "sha1":
        raise ImportCheckError(
            f"source Git object format must be sha1, found {object_format!r}"
        )
    _verify_commit(repository, SOURCE_COMMIT)
    _verify_commit(repository, INVENTORY_COMMIT)
    _verify_inventory_object(repository, installed_inventory_raw)
    _verify_rows_at_commit(repository, SOURCE_COMMIT, copy_rows)

    inventory_by_source = {row["source"]: row for row in manifest["files"]}
    all_transformations = (
        e3a_transformations + e3b_transformations + e3c_transformations
    )
    transformation_rows = [
        inventory_by_source[entry["source"]["path"]]
        for entry in all_transformations
    ]
    source_payloads = _verify_rows_at_commit(
        repository, SOURCE_COMMIT, transformation_rows
    )
    _verify_e3a_transformations_at_source(
        source_payloads, root, e3a_transformations
    )
    _verify_e3b_transformations_at_source(
        source_payloads, root, e3b_transformations
    )
    _verify_e3c_transformations_at_source(
        source_payloads, root, e3c_transformations
    )

    notice_rows = [
        row
        for row in manifest["files"]
        if row["source"] == "NOTICE" and row["disposition"] == "rewrite"
    ]
    if len(notice_rows) != 1:
        raise ImportCheckError(
            "inventory must contain exactly one NOTICE rewrite record"
        )
    notice = notice_rows[0]
    if (
        notice["destination"] != "NOTICE"
        or notice["gitBlobSha1"] != SOURCE_NOTICE_GIT_BLOB_SHA1
        or notice["sha256"] != SOURCE_NOTICE_SHA256
        or notice["mode"] != "100644"
    ):
        raise ImportCheckError(
            "source NOTICE receipt differs from the reviewed rewrite input"
        )
    _verify_rows_at_commit(repository, SOURCE_COMMIT, notice_rows)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_repository(
    root: Path, *, source_git: Path | None = None
) -> dict[str, Any]:
    _verify_root_entries(root)
    inventory_raw, inventory_status = _read_regular_path(
        root, INVENTORY_PATH, max_bytes=MAX_INVENTORY_BYTES
    )
    if _git_mode_from_stat(inventory_status, path=INVENTORY_PATH) != "100644":
        raise ImportCheckError("installed inventory must have Git mode 100644")
    if _git_blob_sha1(inventory_raw) != INVENTORY_GIT_BLOB_SHA1:
        raise ImportCheckError("installed inventory Git blob SHA-1 differs from E1")
    manifest, copy_rows = _parse_inventory_bytes(inventory_raw)
    receipt_raw, receipt_status = _read_regular_path(
        root,
        E3A_RECEIPT_PATH,
        expected_bytes=E3A_RECEIPT_BYTES,
        max_bytes=MAX_RECEIPT_BYTES,
    )
    if _git_mode_from_stat(receipt_status, path=E3A_RECEIPT_PATH) != "100644":
        raise ImportCheckError("installed E3a receipt must have Git mode 100644")
    if _git_blob_sha1(receipt_raw) != E3A_RECEIPT_GIT_BLOB_SHA1:
        raise ImportCheckError("installed E3a receipt Git blob SHA-1 differs")
    e3a_transformations = _parse_e3a_receipt_bytes(receipt_raw, manifest)
    e3b_receipt_raw, e3b_receipt_status = _read_regular_path(
        root,
        E3B_RECEIPT_PATH,
        expected_bytes=E3B_RECEIPT_BYTES,
        max_bytes=MAX_RECEIPT_BYTES,
    )
    if (
        _git_mode_from_stat(e3b_receipt_status, path=E3B_RECEIPT_PATH)
        != "100644"
    ):
        raise ImportCheckError("installed E3b receipt must have Git mode 100644")
    if _git_blob_sha1(e3b_receipt_raw) != E3B_RECEIPT_GIT_BLOB_SHA1:
        raise ImportCheckError("installed E3b receipt Git blob SHA-1 differs")
    e3b_transformations = _parse_e3b_receipt_bytes(e3b_receipt_raw, manifest)
    e3c_receipt_raw, e3c_receipt_status = _read_regular_path(
        root,
        E3C_RECEIPT_PATH,
        expected_bytes=E3C_RECEIPT_BYTES,
        max_bytes=MAX_E3C_RECEIPT_BYTES,
    )
    if (
        _git_mode_from_stat(e3c_receipt_status, path=E3C_RECEIPT_PATH)
        != "100644"
    ):
        raise ImportCheckError("installed E3c receipt must have Git mode 100644")
    if _git_blob_sha1(e3c_receipt_raw) != E3C_RECEIPT_GIT_BLOB_SHA1:
        raise ImportCheckError("installed E3c receipt Git blob SHA-1 differs")
    e3c_transformations = _parse_e3c_receipt_bytes(
        e3c_receipt_raw, manifest
    )
    all_transformations = (
        e3a_transformations + e3b_transformations + e3c_transformations
    )
    _verify_installed(root, manifest, copy_rows, all_transformations)
    _reconstruct_e3c_sources(root, e3c_transformations)
    if source_git is not None:
        _verify_source(
            source_git,
            root,
            manifest,
            copy_rows,
            e3a_transformations,
            e3b_transformations,
            e3c_transformations,
            inventory_raw,
        )
    return {
        "copyBytes": sum(row["bytes"] for row in copy_rows),
        "copyFiles": len(copy_rows),
        "copyTreeSha256": COPY_TREE_SHA256,
        "e3aFiles": len(e3a_transformations),
        "e3bFiles": len(e3b_transformations),
        "e3cFiles": len(e3c_transformations),
        "e3cPackageTreeSha256": E3C_PACKAGE_TREE_SHA256,
        "sourceVerified": source_git is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the pinned Aleph Bench E2 + E3a + E3b + E3c source import."
        )
    )
    parser.add_argument(
        "--source-git",
        type=Path,
        help=(
            "Also verify provenance against an already-fetched Git object "
            "database. No network fetch is performed."
        ),
    )
    args = parser.parse_args(argv)
    result = verify_repository(_repository_root(), source_git=args.source_git)
    mode = "source+installed" if result["sourceVerified"] else "offline-installed"
    print(
        "source import ok: "
        f"mode={mode} commit={SOURCE_COMMIT} files={result['copyFiles']} "
        f"bytes={result['copyBytes']} "
        f"e3aFiles={result['e3aFiles']} "
        f"e3bFiles={result['e3bFiles']} "
        f"e3cFiles={result['e3cFiles']} "
        f"e3cPackageTreeSha256={result['e3cPackageTreeSha256']} "
        f"copyTreeSha256={result['copyTreeSha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportCheckError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
