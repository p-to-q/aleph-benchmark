#!/usr/bin/env python3
"""Verify the closed, byte-exact Aleph-Bench v0.2 scoring authority.

This checker intentionally has no generation mode. Updating an authority byte
requires an explicit review of both the expectation below and the checked-in
lock manifest; editing the manifest alone cannot bless drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = "bench/config/v0.2-scoring-authority-lock.json"
SOURCE_COMMIT = "8fb57eb921d792dbd245d849ab08ac5c7cf11c6e"
AGGREGATE_DOMAIN = b"aleph-bench-v0.2-scoring-authority-v1\0"
EXPECTED_AGGREGATE_SHA256 = (
    "be5aaa575fcc6314aa4f5518d2314b2fecfa5efe320ab84ec595d3fc586a59f2"
)
METRICS_PATH = "bench/engine/metrics.py"
METRICS_SOURCE_DOC_PATH = b"docs/benchmark/02-design-spec.md"
METRICS_STANDALONE_DOC_PATH = b"docs/protocol-v0.2.md"
METRICS_STANDALONE_BYTES = 4_796
METRICS_STANDALONE_SHA256 = (
    "3628f3bf60998707eeadb6e2ab72a1942ca0459d1c018fa5cf7887d2904461d7"
)

# Explicit paths are deliberate: recursive discovery or globs could silently
# enlarge or shrink the authority closure.
EXPECTED_FILES: tuple[tuple[str, str, int, str], ...] = (
    (
        "protocol-config",
        "bench/config/frozen_ladder-v0.2.json",
        703,
        "c4cc5d97655de91fcf22674d5133a689faee8ac27dd8c60a1a67caccae648ba8",
    ),
    (
        "conformance",
        "bench/conformance/scorer-v0.2.json",
        18_057,
        "044925b055db9051329e1556ed524f165d6c8e1e47f02eaf9bdb129aeadf67c6",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-001.json",
        3_413,
        "6a250424862f0859d49ef9e32fe3de18f66c243e4c6337c6cbc65396b48aad42",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-002.json",
        3_943,
        "650ad28c3b99582f6326b8e3a23057a3aca2722477d33f01bb70e2af5e9f0c53",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-003.json",
        3_481,
        "74e537781c76d31ffd91c3e50317376d64e8f19000244aacd880ad6aa8a21522",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-004.json",
        3_462,
        "32b950beebfcf0cef91cec0e52b665ccf30e18d1e81dbaab3ed351abef5eb411",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-005.json",
        3_909,
        "06050f574694be118e91ee04ff3f3c54bf29ac42780ea69c01673c8d267070b9",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-006.json",
        3_473,
        "9d18c86c39da485add5eed01955e914d5ccf952a28b22a7c590f9780e21947ed",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-007.json",
        3_429,
        "a86e12aebee7710a6598d9393b5500c01bc3fa49ec830808f2ed3468da240f1d",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-008.json",
        3_898,
        "13650545523abfa5225d53b16a26ad6f18eac336b711a7e656fc63f2e23a54f3",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-009.json",
        3_497,
        "8885ddb89f7681d3d8b45b045ab8d0da031c7686396b81d1617e096dd56f0f32",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-010.json",
        3_445,
        "f5eec7c0bd25a36afc7c94ae29f9f1e4a9ebcee867349cb2b0ca58aa69c6eb3a",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-011.json",
        3_932,
        "c639ad89f4ba752d00de610a41880932dd581f4c53c98b17864a8e787bd904c9",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-012.json",
        3_481,
        "15e9c41169a885e2c854891d7704973cb9b35f3a8ace3a030f6c0eb99dd3d003",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-013.json",
        3_467,
        "7a7a27ca46f87e3872e2a8ac6464ddd9806b3ecf8c983507733d52624f46f6d6",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-014.json",
        3_902,
        "64cfdc7da9d9c68d87e81cef5a9e21dab628aa713aa87ab29bf9a2ba938a38f0",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-015.json",
        3_473,
        "64fd15feacc9d694883dc337d88e91ddbecfee683a611d851797db32049cd0d1",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-016.json",
        3_437,
        "e899a0f06f9b090b60d4bc2f338a16b0705cd293c8e4eb31e281614c0b9c55c9",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-017.json",
        3_932,
        "64ed023716763a75c245cc04e494c0e65974f75285ffce3420e86a3a4a63670b",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-018.json",
        3_465,
        "38296476678e202ad90de8d0c5b26886f28a7be6f1316156fdefc221f1e356b0",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-019.json",
        3_454,
        "93aae617d26a1f2f23c83e0bf952a1634458ccfbd33e5fbe22e9f6bd68a0d410",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-020.json",
        3_942,
        "dd0406f2f5107973d8b35f00afaa184241305fbc16089285c604bf3c07408caf",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-021.json",
        3_473,
        "0dedc5bc327dbd021648cd3c29abc3c48848e94ef5cb7faddca07d080610fa0e",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-022.json",
        3_462,
        "bb3243677496eb3f9ea5e407ffd1a92ae4fa944fa5336830638da31b998ea13c",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-023.json",
        3_909,
        "e8798d797a55fdb6d206cd2a7d52b686c32e1f6a3d044521cdfbe704d1b4f6f0",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-024.json",
        3_465,
        "f4718837d6340a523635c383b24a984959af7b608bf50db63d6e400dad009b6e",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-025.json",
        3_429,
        "1efe43a2d3893006a09b66d3c4490b11c411391b6afb4f2e1be74d6f3b29f914",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-026.json",
        3_943,
        "0a89d35f0087bc9b9845c95f4fbf983e0c9eda2f51ba65d328b6c21afad17a22",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-027.json",
        3_481,
        "f971dd77213df4b8f7bb4e124afb921de915b882fa396649570cbd5f9b439c6c",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-028.json",
        3_478,
        "d44517e68dd7e727015cfa726d2408266b39d125aad9134fcce0e36669c9655e",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-029.json",
        3_909,
        "a8bf2681c65853787b29f57740eacaf249643b097bd799f97e3d59ccf762dd94",
    ),
    (
        "public-s2-item",
        "bench/data/v0.2/public/s2/s2-030.json",
        3_481,
        "fd6aea421aa87f927e53242af60099c6bdf91c9df130eaff428ad6db0aaee095",
    ),
    (
        "aggregation-module",
        "bench/engine/frozen_ladder.py",
        21_866,
        "42415e6858a57cc7562000cb4a21f28b274a9c875a9e752a064288d2d4c96753",
    ),
    (
        "leakage-module",
        "bench/engine/leakage_gate.py",
        940,
        "f54d5a5cb201524d67ec4321cbbed09aae92933a19901a602733559731a0815d",
    ),
    (
        "metrics-module",
        "bench/engine/metrics.py",
        4_807,
        "23ce9f04cdbc358e7984d346bde1c22679f0902ffc2bbd0afcd597ee0a17c002",
    ),
    (
        "protocol-module",
        "bench/engine/protocol.py",
        10_690,
        "0f5fcfe2e16003f516de260fba5c6c6fdc19a1ccf2eced7b2b25dbac911887fc",
    ),
    (
        "scorer",
        "bench/engine/scoring_core.py",
        16_621,
        "bf38b1ac35bc1cdb99fb59f26de5d0e6ac953ca7f29a07a2b90e7cf3b028dd01",
    ),
    (
        "public-schema",
        "schemas/v0.2/aleph-bench-item.schema.json",
        2_446,
        "ad2b014b897084552b481aa73eb0e399f137f16edbd2248a2c8a1578fd8a1bc9",
    ),
    (
        "public-schema",
        "schemas/v0.2/aleph-bench-manifest.schema.json",
        11_498,
        "81360d210a95846f3b17321f6c1694748cadb64b7bdac1e38becb42dd3981c42",
    ),
    (
        "public-schema",
        "schemas/v0.2/aleph-bench-platform-package.schema.json",
        3_494,
        "48a395a6ba930785556fe356c8eb2a142a8042ba12a38c4e5fdd9649ffc765bf",
    ),
    (
        "public-schema",
        "schemas/v0.2/aleph-bench-result.schema.json",
        20_453,
        "05414d02ef0d12a678e22387777adab4fb720967f8dafc58faa908b4b57f9bfb",
    ),
)


class AuthorityError(RuntimeError):
    """Raised when the authority closure differs from its reviewed bytes."""


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
        raise AuthorityError(f"{label} contains non-finite JSON value {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuthorityError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorityError(f"{label} is not strict UTF-8 JSON: {error}") from error


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
        raise AuthorityError(f"unsafe or non-canonical authority path: {raw_path!r}")
    return path


def _read_regular_file(root: Path, raw_path: str, *, max_bytes: int) -> bytes:
    """Read below root without following symlinks in any relative component."""

    relative_path = _validated_relative_path(raw_path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise AuthorityError("this verifier requires O_NOFOLLOW and O_DIRECTORY")
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
            raise AuthorityError(f"{raw_path} must be a singly linked regular file")
        if before.st_size > max_bytes:
            raise AuthorityError(
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
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise AuthorityError(f"{raw_path} changed while it was being verified")
        if len(data) > max_bytes:
            raise AuthorityError(
                f"{raw_path} exceeds the {max_bytes}-byte verification limit"
            )
        return data
    except OSError as error:
        raise AuthorityError(f"cannot safely read {raw_path}: {error}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if parent_fd != root_fd:
            os.close(parent_fd)
        os.close(root_fd)


def _expected_manifest() -> dict[str, Any]:
    files = [
        {"bytes": byte_count, "path": path, "role": role, "sha256": sha256}
        for role, path, byte_count, sha256 in EXPECTED_FILES
    ]
    return {
        "aggregate": {
            "algorithm": "sha256-length-framed-path-and-content-v1",
            "domain": "aleph-bench-v0.2-scoring-authority-v1",
            "sha256": EXPECTED_AGGREGATE_SHA256,
        },
        "artifactKind": "aleph-bench-v0.2-scoring-authority-lock",
        "fileCount": len(files),
        "files": files,
        "formatVersion": 1,
        "hashAlgorithm": "sha256",
        "protocolVersion": "0.2.0",
        "sourceCommit": SOURCE_COMMIT,
    }


def _reconstruct_authority_bytes(path: str, installed: bytes) -> bytes:
    """Return the pinned source bytes represented by an installed authority file.

    The standalone metrics port differs from the v0.2 authority by one reviewed
    documentation-path replacement.  Verify the installed side first, reverse
    that replacement in memory, and let the normal byte and aggregate checks
    below validate the reconstructed source.  No checked-in authority byte is
    rewritten by this verifier.
    """

    if path != METRICS_PATH:
        return installed
    installed_sha256 = hashlib.sha256(installed).hexdigest()
    if (
        len(installed) != METRICS_STANDALONE_BYTES
        or installed_sha256 != METRICS_STANDALONE_SHA256
    ):
        raise AuthorityError(
            "standalone metrics port differs from its reviewed output: "
            f"expected bytes={METRICS_STANDALONE_BYTES} "
            f"sha256={METRICS_STANDALONE_SHA256}, found bytes={len(installed)} "
            f"sha256={installed_sha256}"
        )
    if installed.count(METRICS_STANDALONE_DOC_PATH) != 1:
        raise AuthorityError(
            "standalone metrics port must contain its reviewed documentation "
            "path exactly once"
        )
    if METRICS_SOURCE_DOC_PATH in installed:
        raise AuthorityError(
            "standalone metrics port unexpectedly contains the source path"
        )
    reconstructed = installed.replace(
        METRICS_STANDALONE_DOC_PATH,
        METRICS_SOURCE_DOC_PATH,
        1,
    )
    if reconstructed.replace(
        METRICS_SOURCE_DOC_PATH,
        METRICS_STANDALONE_DOC_PATH,
        1,
    ) != installed:
        raise AuthorityError("standalone metrics transform is not reversible")
    return reconstructed


def verify(root: Path = ROOT) -> dict[str, Any]:
    manifest_bytes = _read_regular_file(root, LOCK_PATH, max_bytes=64 * 1024)
    manifest = _strict_json(manifest_bytes, label=LOCK_PATH)
    expected_manifest = _expected_manifest()
    if manifest != expected_manifest:
        raise AuthorityError(
            "v0.2 scoring-authority lock differs from the reviewed explicit closure"
        )
    if manifest_bytes != _canonical_json(manifest):
        raise AuthorityError(f"{LOCK_PATH} is not canonical JSON")

    aggregate = hashlib.sha256(AGGREGATE_DOMAIN)
    for _role, path, expected_bytes, expected_sha256 in EXPECTED_FILES:
        installed_max_bytes = (
            METRICS_STANDALONE_BYTES if path == METRICS_PATH else expected_bytes
        )
        installed = _read_regular_file(
            root,
            path,
            max_bytes=installed_max_bytes,
        )
        data = _reconstruct_authority_bytes(path, installed)
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if len(data) != expected_bytes or actual_sha256 != expected_sha256:
            raise AuthorityError(
                f"authority byte drift at {path}: expected bytes={expected_bytes} "
                f"sha256={expected_sha256}, found bytes={len(data)} "
                f"sha256={actual_sha256}"
            )
        path_bytes = path.encode("utf-8")
        aggregate.update(len(path_bytes).to_bytes(4, "big"))
        aggregate.update(path_bytes)
        aggregate.update(len(data).to_bytes(8, "big"))
        aggregate.update(data)

    actual_aggregate = aggregate.hexdigest()
    if actual_aggregate != EXPECTED_AGGREGATE_SHA256:
        raise AuthorityError(
            "v0.2 scoring-authority aggregate differs: "
            f"expected {EXPECTED_AGGREGATE_SHA256}, found {actual_aggregate}"
        )
    return {
        "fileCount": len(EXPECTED_FILES),
        "protocolVersion": "0.2.0",
        "sha256": actual_aggregate,
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the immutable Aleph-Bench v0.2 scoring authority."
    )
    parser.parse_args(argv)
    try:
        report = verify()
    except AuthorityError as error:
        print(f"v0.2 scoring authority: FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
