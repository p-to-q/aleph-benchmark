from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .kaggle_diagnostic_receipt import (
    MAX_RECEIPT_BYTES,
    parse_kaggle_diagnostic_receipt_bytes,
)


RECEIPT_FILENAME = "aleph-bench-v0.2-kaggle-diagnostic-receipt.json"
MAX_ARCHIVE_BYTES = 268_435_456
MAX_ARCHIVE_ENTRIES = 4_096
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1_073_741_824
MAX_RUNS_TO_INSPECT = 100
MAX_RUN_PAGES_TO_INSPECT = 10
_SLUG_SEGMENT = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,98}[a-z0-9])?$")
_TERMINAL_TASK_STATES = {
    "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED",
    "BENCHMARK_TASK_VERSION_CREATION_STATE_ERRORED",
}
_TERMINAL_RUN_STATES = {
    "BENCHMARK_TASK_RUN_STATE_COMPLETED",
    "BENCHMARK_TASK_RUN_STATE_ERRORED",
}


class KaggleCreationOutputError(ValueError):
    """Raised when a task-creation output cannot be bound and verified."""


def _fail(message: str) -> None:
    raise KaggleCreationOutputError(message)


def _write_exclusive(path: Path, data: bytes) -> None:
    # Exclusive creation closes the race between a preflight exists check and
    # the write; an older task version must never be silently overwritten.
    handle = path.open("xb")
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A failed durable write is not evidence. Remove the file created by
        # this call so an operator can retry after addressing the I/O failure.
        # Opening an existing path fails before this block and is never unlinked.
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _parse_task_slug(value: str) -> tuple[str | None, str]:
    parts = value.split("/")
    if len(parts) == 1:
        owner = None
        task = parts[0]
    elif len(parts) == 2:
        owner, task = parts
    else:
        _fail("task must be TASK or OWNER/TASK")
    for role, segment in (("owner", owner), ("task", task)):
        if segment is not None and not _SLUG_SEGMENT.fullmatch(segment):
            _fail(f"invalid Kaggle {role} slug: {segment!r}")
    return owner, task


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(value).rsplit(".", 1)[-1]


def _optional_positive_id(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"Kaggle {label} is invalid")
    if value == 0:
        return None
    if value < 0:
        _fail(f"Kaggle {label} is invalid")
    return value


def _task_metadata(
    task_info: Any,
    *,
    requested_task: str,
    expected_version: int,
    expected_source_kernel_id: int | None,
    expected_datasets: tuple[str, ...],
) -> dict[str, Any]:
    requested_owner, requested_slug = _parse_task_slug(requested_task)
    response_slug = getattr(task_info, "slug", None)
    if response_slug is None:
        _fail("Kaggle task response has no version slug")
    actual_owner = getattr(response_slug, "owner_slug", None) or None
    actual_slug = getattr(response_slug, "task_slug", None)
    actual_version = getattr(response_slug, "version_number", None)
    if actual_slug != requested_slug:
        _fail(
            f"Kaggle returned task {actual_slug!r}, expected {requested_slug!r}"
        )
    if requested_owner is not None and actual_owner != requested_owner:
        _fail(
            f"Kaggle returned owner {actual_owner!r}, expected {requested_owner!r}"
        )
    if (
        isinstance(actual_version, bool)
        or not isinstance(actual_version, int)
        or actual_version <= 0
    ):
        _fail("Kaggle task response has no positive integer version")
    if actual_version != expected_version:
        _fail(
            f"Kaggle returned task version {actual_version!r}, "
            f"expected {expected_version}"
        )

    state = _enum_name(getattr(task_info, "creation_state", None))
    if state not in _TERMINAL_TASK_STATES:
        error = getattr(task_info, "creation_error_message", None) or getattr(
            task_info, "error", None
        )
        suffix = f": {error}" if error else ""
        _fail(f"task creation is not complete ({state}){suffix}")

    source_kernel_id = _optional_positive_id(
        getattr(task_info, "source_kernel_id", None),
        label="source kernel ID",
    )
    if (
        expected_source_kernel_id is not None
        and source_kernel_id != expected_source_kernel_id
    ):
        _fail(
            f"Kaggle returned source kernel {source_kernel_id}, "
            f"expected {expected_source_kernel_id}"
        )

    options = getattr(task_info, "options", None)
    actual_datasets = tuple(
        sorted(getattr(options, "dataset_data_sources", None) or ())
    )
    expected_sorted = tuple(sorted(expected_datasets))
    if actual_datasets != expected_sorted:
        _fail(
            "attached Kaggle datasets differ: "
            f"expected {list(expected_sorted)!r}, found {list(actual_datasets)!r}"
        )

    create_time_value = _datetime_value(getattr(task_info, "create_time", None))
    return {
        "requested": requested_task,
        "owner": actual_owner,
        "slug": actual_slug,
        "version": actual_version,
        "sourceKernelId": source_kernel_id,
        "creationState": state,
        "creationError": (
            getattr(task_info, "creation_error_message", None)
            or getattr(task_info, "error", None)
            or None
        ),
        "createTime": create_time_value,
        "url": getattr(task_info, "url", None) or None,
        "datasets": list(actual_datasets),
    }


def _datetime_value(value: Any) -> str | None:
    if isinstance(value, datetime):
        # kagglesdk 0.1.37 deserializes benchmark API UTC timestamps into
        # naive datetime objects even though the service values are UTC.
        # Restore that documented transport context only for typed datetimes;
        # arbitrary strings remain untrusted and are validated downstream.
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if value is None:
        return None
    return str(value)


def _run_identity(run_info: Any) -> tuple[str | None, str | None, Any]:
    slug = getattr(run_info, "task_slug", None)
    if slug is None:
        return None, None, None
    owner = getattr(slug, "owner_slug", None) or None
    task = getattr(slug, "task_slug", None)
    version = getattr(slug, "version_number", None)
    return owner, task, version


def _run_metadata(
    run_info: Any,
    *,
    requested_task: str,
    expected_version: int,
    expected_run_id: int | None,
) -> dict[str, Any]:
    requested_owner, requested_slug = _parse_task_slug(requested_task)
    actual_owner, actual_slug, actual_version = _run_identity(run_info)
    if actual_slug != requested_slug:
        _fail(
            f"Kaggle returned run task {actual_slug!r}, "
            f"expected {requested_slug!r}"
        )
    if requested_owner is not None and actual_owner != requested_owner:
        _fail(
            f"Kaggle returned run owner {actual_owner!r}, "
            f"expected {requested_owner!r}"
        )
    if (
        isinstance(actual_version, bool)
        or not isinstance(actual_version, int)
        or actual_version <= 0
    ):
        _fail("Kaggle task run has no positive integer task version")
    if actual_version != expected_version:
        _fail(
            f"Kaggle returned run task version {actual_version!r}, "
            f"expected {expected_version}"
        )

    run_id = getattr(run_info, "id", None)
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        _fail("Kaggle task run has no positive run ID")
    if expected_run_id is not None and run_id != expected_run_id:
        _fail(f"Kaggle returned run {run_id}, expected {expected_run_id}")

    state = _enum_name(getattr(run_info, "state", None))
    if state not in _TERMINAL_RUN_STATES:
        _fail(
            "task run is not terminal; wait and retry this read-only "
            f"download ({state})"
        )
    model = getattr(run_info, "model_version_slug", None)
    if not isinstance(model, str) or not model.strip():
        _fail("Kaggle task run has no model version slug")
    return {
        "id": run_id,
        "owner": actual_owner,
        "task": actual_slug,
        "version": actual_version,
        "modelVersionSlug": model,
        "state": state,
        "error": getattr(run_info, "error_message", None) or None,
        "startTime": _datetime_value(getattr(run_info, "start_time", None)),
        "endTime": _datetime_value(getattr(run_info, "end_time", None)),
    }


def _select_exact_run(
    runs: list[Any],
    *,
    requested_task: str,
    expected_version: int,
    expected_run_id: int | None,
) -> Any:
    requested_owner, requested_slug = _parse_task_slug(requested_task)
    matches: list[Any] = []
    for run in runs:
        owner, task, version = _run_identity(run)
        if (
            task == requested_slug
            and isinstance(version, int)
            and not isinstance(version, bool)
            and version == expected_version
            and (requested_owner is None or owner == requested_owner)
        ):
            matches.append(run)

    if expected_run_id is not None:
        matches = [
            run for run in matches if getattr(run, "id", None) == expected_run_id
        ]

    if len(matches) != 1:
        authority = (
            f"run ID {expected_run_id} for the creation version"
            if expected_run_id is not None
            else "task run for the creation version"
        )
        _fail(
            f"expected exactly one {authority}; "
            f"found {len(matches)}. Do not guess which run is authoritative."
        )
    return matches[0]


def _safe_archive_entry(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if not name or "\\" in name or "\x00" in name:
        _fail(f"unsafe creation archive entry: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        _fail(f"unsafe creation archive entry: {name!r}")
    unix_mode = (info.external_attr >> 16) & 0o170000
    if unix_mode == 0o120000:
        _fail(f"symlink creation archive entry is forbidden: {name!r}")


def _extract_verified_receipt(
    archive_bytes: bytes,
) -> tuple[bytes, dict[str, Any], str]:
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        _fail(
            f"creation archive exceeds the {MAX_ARCHIVE_BYTES}-byte safety limit"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_ENTRIES:
                _fail("creation archive has too many entries")
            total_size = 0
            matches: list[zipfile.ZipInfo] = []
            for info in infos:
                _safe_archive_entry(info)
                total_size += info.file_size
                if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    _fail("creation archive uncompressed size exceeds the safety limit")
                if PurePosixPath(info.filename).name == RECEIPT_FILENAME:
                    matches.append(info)
            if len(matches) != 1:
                _fail(
                    "creation archive must contain exactly one diagnostic receipt; "
                    f"found {len(matches)}"
                )
            receipt_info = matches[0]
            if receipt_info.file_size > MAX_RECEIPT_BYTES:
                _fail("creation receipt exceeds the safety limit")
            receipt_bytes = archive.read(receipt_info)
    except zipfile.BadZipFile as exc:
        raise KaggleCreationOutputError(
            f"Kaggle creation output is not a valid zip archive: {exc}"
        ) from exc
    receipt = parse_kaggle_diagnostic_receipt_bytes(receipt_bytes)
    return receipt_bytes, receipt, receipt_info.filename


def _gate_result(receipt: dict[str, Any], mode: str) -> tuple[bool, str]:
    calls = receipt["calls"]
    if mode == "zero-call":
        zero_calls_proven = (
            calls["attempted"] == 0
            and calls["completed"] == 0
            and calls["activeCall"] is None
            and receipt["rows"] == []
        )
        if (
            receipt["status"] == "blocked"
            and receipt["phase"] == "runtime_preflight"
            and zero_calls_proven
        ):
            return (
                False,
                "zero calls proven, but runtime is incompatible and the "
                "package gate was not reached",
            )
        passed = (
            receipt["status"] == "blocked"
            and receipt["phase"] == "package_preflight"
            and zero_calls_proven
        )
        return passed, (
            "package preflight blocked before every model call"
            if passed
            else "receipt did not prove the zero-call package gate"
        )
    if mode == "six-call":
        passed = (
            receipt["status"] == "complete"
            and calls["attempted"] == 6
            and calls["completed"] == 6
            and calls["activeCall"] is None
            and len(receipt["rows"]) == 6
        )
        return passed, (
            "all six diagnostic calls completed"
            if passed
            else "receipt retained but the six-call canary did not complete"
        )
    _fail(f"unknown gate mode: {mode!r}")


def _validate_gate_datasets(mode: str, datasets: tuple[str, ...]) -> None:
    if mode == "zero-call":
        if datasets:
            _fail("zero-call gate must expect no attached datasets")
        return
    if mode == "six-call":
        if len(datasets) != 1:
            _fail("six-call gate must expect exactly one attached dataset")
        return
    _fail(f"unknown gate mode: {mode!r}")


def _read_bounded_download(response: Any) -> bytes:
    """Consume a streamed SDK download without an unbounded `.content` read."""

    close = getattr(response, "close", None)
    try:
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        headers = getattr(response, "headers", {}) or {}
        content_length = headers.get("Content-Length") or headers.get(
            "content-length"
        )
        announced_bytes: int | None = None
        if content_length is not None:
            try:
                announced_bytes = int(content_length)
            except (TypeError, ValueError) as exc:
                raise KaggleCreationOutputError(
                    "Kaggle creation output has an invalid Content-Length"
                ) from exc
            if announced_bytes < 0:
                _fail("Kaggle creation output has a negative Content-Length")
            if announced_bytes > MAX_ARCHIVE_BYTES:
                _fail("Kaggle creation output exceeds the archive safety limit")

        iter_content = getattr(response, "iter_content", None)
        if not callable(iter_content):
            _fail("Kaggle creation output response is not stream-readable")
        body = bytearray()
        for chunk in iter_content(chunk_size=1_048_576):
            if not chunk:
                continue
            if not isinstance(chunk, (bytes, bytearray)):
                _fail("Kaggle creation output stream yielded non-byte content")
            if len(body) + len(chunk) > MAX_ARCHIVE_BYTES:
                _fail("Kaggle creation output exceeds the archive safety limit")
            body.extend(chunk)
        if announced_bytes is not None and len(body) != announced_bytes:
            _fail(
                "Kaggle creation output length differs from Content-Length: "
                f"expected {announced_bytes}, received {len(body)}"
            )
        return bytes(body)
    finally:
        if callable(close):
            close()


def write_creation_bundle(
    *,
    task_info: Any,
    run_info: Any,
    requested_task: str,
    expected_version: int,
    expected_source_kernel_id: int | None,
    expected_run_id: int | None,
    expected_datasets: tuple[str, ...],
    archive_bytes: bytes,
    gate_mode: str,
    output_dir: Path,
) -> tuple[dict[str, Any], bool]:
    """Validate and persist one exact task-creation archive and receipt."""

    _validate_gate_datasets(gate_mode, expected_datasets)
    task = _task_metadata(
        task_info,
        requested_task=requested_task,
        expected_version=expected_version,
        expected_source_kernel_id=expected_source_kernel_id,
        expected_datasets=expected_datasets,
    )
    run = _run_metadata(
        run_info,
        requested_task=requested_task,
        expected_version=expected_version,
        expected_run_id=expected_run_id,
    )
    receipt_bytes, receipt, receipt_entry = _extract_verified_receipt(
        archive_bytes
    )
    gate_passed, gate_message = _gate_result(receipt, gate_mode)
    if task["creationState"] != "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED":
        gate_passed = False
        gate_message = (
            "creation kernel errored; receipt retained for manual review"
        )
    if run["state"] != "BENCHMARK_TASK_RUN_STATE_COMPLETED":
        gate_passed = False
        gate_message = "creation task run errored; receipt retained for manual review"
    stem = f"{task['slug']}-v{expected_version}-run-{run['id']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{stem}.zip"
    receipt_path = output_dir / f"{stem}-receipt.json"
    metadata_path = output_dir / f"{stem}-metadata.json"
    for path in (archive_path, receipt_path, metadata_path):
        if path.exists():
            _fail(f"refusing to overwrite existing creation artifact: {path}")

    metadata = {
        "artifactKind": "kaggle_task_creation_bundle",
        "task": task,
        "run": run,
        "archive": {
            "file": archive_path.name,
            "bytes": len(archive_bytes),
            "sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "receiptEntry": receipt_entry,
        },
        "receipt": {
            "file": receipt_path.name,
            "id": receipt["id"],
            "status": receipt["status"],
            "phase": receipt["phase"],
            "attempted": receipt["calls"]["attempted"],
            "completed": receipt["calls"]["completed"],
            "activeCall": receipt["calls"]["activeCall"],
            "manualReviewRequired": (
                task["creationState"]
                != "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED"
                or run["state"] != "BENCHMARK_TASK_RUN_STATE_COMPLETED"
                or receipt["calls"]["activeCall"] is not None
                or (
                    receipt["status"] == "blocked"
                    and receipt["calls"]["attempted"] > 0
                )
            ),
        },
        "gate": {
            "mode": gate_mode,
            "passed": gate_passed,
            "message": gate_message,
        },
    }
    metadata_bytes = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    written: list[Path] = []
    try:
        for path, data in (
            (archive_path, archive_bytes),
            (receipt_path, receipt_bytes),
            (metadata_path, metadata_bytes),
        ):
            _write_exclusive(path, data)
            written.append(path)
    except BaseException:
        # Only remove files created by this invocation; pre-existing evidence
        # is protected by exclusive creation above.
        for path in reversed(written):
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return metadata, gate_passed


def _load_kaggle_sdk() -> tuple[Any, Any, Any, Any, Any]:
    """Load the optional Kaggle client only when the CLI path is used."""

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.benchmarks.types.benchmark_tasks_api_service import (
            ApiBenchmarkTaskSlug,
            ApiDownloadBenchmarkTaskRunOutputRequest,
            ApiGetBenchmarkTaskRequest,
            ApiListBenchmarkTaskRunsRequest,
        )
    except ImportError as exc:
        raise KaggleCreationOutputError(
            "Kaggle CLI 2.2.4 or newer is required to download creation output"
        ) from exc
    return (
        KaggleApi,
        ApiBenchmarkTaskSlug,
        ApiDownloadBenchmarkTaskRunOutputRequest,
        ApiGetBenchmarkTaskRequest,
        ApiListBenchmarkTaskRunsRequest,
    )


def _download_creation_archive(
    task: str,
    version: int,
    expected_run_id: int | None,
    expected_source_kernel_id: int | None,
    expected_datasets: tuple[str, ...],
) -> tuple[Any, Any, bytes]:
    (
        KaggleApi,
        ApiBenchmarkTaskSlug,
        ApiDownloadBenchmarkTaskRunOutputRequest,
        ApiGetBenchmarkTaskRequest,
        ApiListBenchmarkTaskRunsRequest,
    ) = _load_kaggle_sdk()

    owner, task_slug = _parse_task_slug(task)
    slug = ApiBenchmarkTaskSlug()
    if owner is not None:
        slug.owner_slug = owner
    slug.task_slug = task_slug
    slug.version_number = version
    get_request = ApiGetBenchmarkTaskRequest()
    get_request.slug = slug

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        task_client = client.benchmarks.benchmark_tasks_api_client
        get_task = task_client.get_benchmark_task
        task_info = api.with_retry(get_task)(get_request)
        task_record = _task_metadata(
            task_info,
            requested_task=task,
            expected_version=version,
            expected_source_kernel_id=expected_source_kernel_id,
            expected_datasets=expected_datasets,
        )
        resolved_task = (
            f"{task_record['owner']}/{task_record['slug']}"
            if task_record["owner"] is not None
            else task_record["slug"]
        )

        list_request = ApiListBenchmarkTaskRunsRequest()
        list_request.task_slug = slug
        list_request.page_size = MAX_RUNS_TO_INSPECT
        runs: list[Any] = []
        seen_page_tokens: set[str] = set()
        pages_inspected = 0
        while True:
            pages_inspected += 1
            if pages_inspected > MAX_RUN_PAGES_TO_INSPECT:
                _fail(
                    "too many task-run pages to bind creation output safely; "
                    "manual reconciliation is required"
                )
            # These calls are read-only. Retrying them cannot create another
            # task version or schedule another model execution.
            response = api.with_retry(task_client.list_benchmark_task_runs)(
                list_request
            )
            runs.extend(list(getattr(response, "runs", None) or ()))
            if len(runs) > MAX_RUNS_TO_INSPECT:
                _fail(
                    "too many task runs to bind creation output safely; "
                    "manual reconciliation is required"
                )
            next_page_token = getattr(response, "next_page_token", None) or ""
            if not next_page_token:
                break
            if next_page_token in seen_page_tokens:
                _fail("Kaggle task-run pagination repeated a page token")
            if pages_inspected >= MAX_RUN_PAGES_TO_INSPECT:
                _fail(
                    "too many task-run pages to bind creation output safely; "
                    "manual reconciliation is required"
                )
            if len(runs) >= MAX_RUNS_TO_INSPECT:
                _fail(
                    "too many task runs to bind creation output safely; "
                    "manual reconciliation is required"
                )
            seen_page_tokens.add(next_page_token)
            list_request.page_token = next_page_token

        run_info = _select_exact_run(
            runs,
            requested_task=resolved_task,
            expected_version=version,
            expected_run_id=expected_run_id,
        )
        _run_metadata(
            run_info,
            requested_task=resolved_task,
            expected_version=version,
            expected_run_id=expected_run_id,
        )
        output_request = ApiDownloadBenchmarkTaskRunOutputRequest()
        output_request.run_id = getattr(run_info, "id")
        # Retain the executed notebook beside the receipt so a later audit can
        # compare its code cells with the submitted, content-addressed source.
        output_request.include_source = True
        download = task_client.download_benchmark_task_run_output
        response = api.with_retry(download)(output_request)
        return task_info, run_info, _read_bounded_download(response)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify the unique creation-triggered run output for "
            "one exact Kaggle benchmark task version without scheduling a run."
        )
    )
    parser.add_argument("task", help="TASK or OWNER/TASK slug")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument(
        "--source-kernel-id",
        type=int,
        help=(
            "Optional backing-kernel ID to cross-check. Kaggle's create "
            "response may omit it; the exact version readback remains required."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help=(
            "Optional run ID to cross-check. Without it, exactly one run must "
            "exist for the requested task version."
        ),
    )
    parser.add_argument(
        "--gate", choices=("zero-call", "six-call"), required=True
    )
    datasets = parser.add_mutually_exclusive_group(required=True)
    datasets.add_argument(
        "--expect-no-datasets", action="store_true"
    )
    datasets.add_argument(
        "--expect-dataset", action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.version <= 0:
        parser.error("--version must be positive")
    if args.source_kernel_id is not None and args.source_kernel_id <= 0:
        parser.error("--source-kernel-id must be positive")
    if args.run_id is not None and args.run_id <= 0:
        parser.error("--run-id must be positive")
    expected_datasets = (
        () if args.expect_no_datasets else tuple(args.expect_dataset)
    )
    try:
        task_info, run_info, archive_bytes = _download_creation_archive(
            args.task,
            args.version,
            args.run_id,
            args.source_kernel_id,
            expected_datasets,
        )
        metadata, passed = write_creation_bundle(
            task_info=task_info,
            run_info=run_info,
            requested_task=args.task,
            expected_version=args.version,
            expected_source_kernel_id=args.source_kernel_id,
            expected_run_id=args.run_id,
            expected_datasets=expected_datasets,
            archive_bytes=archive_bytes,
            gate_mode=args.gate,
            output_dir=args.output,
        )
    except (OSError, ValueError) as exc:
        print(f"invalid Kaggle creation output: {exc}", file=sys.stderr)
        return 1
    print(
        f"retained task {metadata['task']['slug']} v{metadata['task']['version']} "
        f"run {metadata['run']['id']} (creation kernel "
        f"{metadata['task']['sourceKernelId']}): "
        f"{metadata['gate']['message']}"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
