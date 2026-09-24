from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bench.tasks.kaggle.generate_v0_2_capture import (
    OUTPUT_PATH as CAPTURE_OUTPUT_PATH,
    render_task_source as render_capture_task_source,
)
from bench.tasks.kaggle.generate_v0_2_diagnostic import (
    OUTPUT_PATH,
    render_task_source,
)


TASK_SLUG = "aleph-bench-deployment-diagnostic-2048-none"
CAPTURE_TASK_SLUG = "aleph-bench-v0-2-capture-canary"
MAX_SOURCE_BYTES = 2_097_152
REQUIRED_PYTHON_MAJOR_MINOR = (3, 13)
REQUIRED_CLIENT_VERSIONS = {
    "kaggle": "2.2.4",
    "kagglesdk": "0.1.37",
    "jupytext": "1.19.5",
}
_DATASET_SLUG = re.compile(
    r"^[a-z0-9](?:[a-z0-9_-]{0,98}[a-z0-9])?/"
    r"[a-z0-9](?:[a-z0-9_-]{0,98}[a-z0-9])?$"
)
_TERMINAL_CREATION_STATES = {
    "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED",
    "BENCHMARK_TASK_VERSION_CREATION_STATE_ERRORED",
    "BENCHMARK_TASK_VERSION_CREATION_STATE_KERNEL_WITHOUT_RUN",
    "BENCHMARK_TASK_VERSION_CREATION_STATE_VALIDATION_FAILED",
    "BENCHMARK_TASK_VERSION_CREATION_STATE_NO_MODEL_SPECIFIED",
}


class KagglePushOnceError(ValueError):
    """Raised when the diagnostic task cannot be submitted safely once."""


class KagglePushOutcomeAmbiguous(KagglePushOnceError):
    """Raised after dispatch when task creation may have reached Kaggle."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_parent_directory(path: Path) -> None:
    # File fsync alone does not guarantee that a newly created or replaced
    # directory entry survives a machine-level crash. The dispatch journal is
    # the retry barrier, so persist its parent before crossing that boundary.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_initial_journal(path: Path, journal: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is the process-level guard. Two operators using the
    # same reviewed journal path cannot both reach the dispatch boundary.
    try:
        with path.open("xb") as handle:
            handle.write(_canonical_json_bytes(journal))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent_directory(path)
    except FileExistsError as exc:
        raise KagglePushOnceError(
            f"push journal already exists; inspect it and do not retry: {path}"
        ) from exc


def _replace_journal(path: Path, journal: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(journal))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_parent_directory(path)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _exception_record(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__[:200],
        "message": str(exc)[:2000],
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verified_client_versions() -> dict[str, str]:
    observed = {
        "python": sys.version.split()[0],
        **{
            name: _distribution_version(name)
            for name in REQUIRED_CLIENT_VERSIONS
        },
    }
    mismatches: list[str] = []
    if sys.version_info[:2] != REQUIRED_PYTHON_MAJOR_MINOR:
        mismatches.append(f"python={observed['python']} (requires 3.13.x)")
    mismatches.extend(
        f"{name}={observed[name]!r} (requires {required!r})"
        for name, required in REQUIRED_CLIENT_VERSIONS.items()
        if observed[name] != required
    )
    if mismatches:
        raise KagglePushOnceError(
            "unreviewed Kaggle client environment; no remote request was made: "
            + "; ".join(mismatches)
        )
    # The comparison above proves these optional metadata values are strings.
    return {name: str(version) for name, version in observed.items()}


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(value).rsplit(".", 1)[-1]


def _optional_positive_id(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise KagglePushOnceError(f"Kaggle response has an invalid {label}")
    if value == 0:
        return None
    if value < 0:
        raise KagglePushOnceError(f"Kaggle response has an invalid {label}")
    return value


def _is_http_not_found(exc: BaseException) -> bool:
    """Return whether a Kaggle request failed because the task does not exist."""

    return (
        getattr(getattr(exc, "response", None), "status_code", None) == 404
    )


def _prior_task_record(
    task_info: Any | None, *, expected_task: str = TASK_SLUG
) -> dict[str, Any] | None:
    if task_info is None:
        return None
    slug = getattr(task_info, "slug", None)
    actual_task = getattr(slug, "task_slug", None)
    version = getattr(slug, "version_number", None)
    if actual_task != expected_task:
        raise KagglePushOnceError(
            f"Kaggle preflight returned task {actual_task!r}, expected {expected_task!r}"
        )
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise KagglePushOnceError(
            "Kaggle preflight returned no positive prior task version"
        )
    state = _enum_name(getattr(task_info, "creation_state", None))
    if state not in _TERMINAL_CREATION_STATES:
        raise KagglePushOnceError(
            "latest Kaggle task creation is pending or has an unknown state; "
            f"reconcile it before creating another version ({state})"
        )
    source_kernel_id = _optional_positive_id(
        getattr(task_info, "source_kernel_id", None),
        label="source kernel ID",
    )
    options = getattr(task_info, "options", None)
    datasets = tuple(sorted(getattr(options, "dataset_data_sources", None) or ()))
    return {
        "owner": getattr(slug, "owner_slug", None) or None,
        "task": actual_task,
        "version": version,
        "creationState": state,
        "sourceKernelId": source_kernel_id,
        "datasets": list(datasets),
    }


def preflight_and_dispatch_once(
    *,
    fetch_latest_task: Callable[[], Any | None],
    create_task: Callable[[Any], Any],
    request: Any,
    journal_path: Path,
    prepared_journal: dict[str, Any],
    response_record: Callable[[Any], dict[str, Any]],
    expected_task: str = TASK_SLUG,
    clock: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Require a terminal prior version, then cross one create boundary."""

    prior_task = _prior_task_record(
        fetch_latest_task(), expected_task=expected_task
    )
    prepared = dict(prepared_journal)
    prepared["remotePreflight"] = {
        "observedAt": clock(),
        "priorTask": prior_task,
    }
    return dispatch_create_once(
        create_task=create_task,
        request=request,
        journal_path=journal_path,
        prepared_journal=prepared,
        response_record=response_record,
        clock=clock,
    )


def dispatch_create_once(
    *,
    create_task: Callable[[Any], Any],
    request: Any,
    journal_path: Path,
    prepared_journal: dict[str, Any],
    response_record: Callable[[Any], dict[str, Any]],
    clock: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Dispatch exactly once and persist ambiguity before propagating failure."""

    journal = dict(prepared_journal)
    journal["state"] = "prepared"
    journal["updatedAt"] = clock()
    journal["response"] = None
    journal["failure"] = None
    _write_initial_journal(journal_path, journal)

    # This durable boundary precedes the only non-idempotent call. Once it is
    # written, any exception is ambiguous: a server may have accepted the
    # request even when the response never reached this process.
    journal["state"] = "dispatching"
    journal["updatedAt"] = clock()
    _replace_journal(journal_path, journal)
    try:
        response = create_task(request)
        normalized_response = response_record(response)
    except BaseException as exc:
        journal["state"] = "ambiguous"
        journal["updatedAt"] = clock()
        journal["failure"] = _exception_record(exc)
        _replace_journal(journal_path, journal)
        if isinstance(exc, Exception):
            raise KagglePushOutcomeAmbiguous(
                "task creation outcome is ambiguous; inspect Kaggle task "
                "versions, source kernels, quota, and this journal; do not retry"
            ) from exc
        raise

    journal["state"] = "returned"
    journal["updatedAt"] = clock()
    journal["response"] = normalized_response
    _replace_journal(journal_path, journal)
    return journal


def _source_and_notebook(source_path: Path) -> tuple[bytes, str]:
    return _exact_source_and_notebook(
        source_path,
        expected_path=OUTPUT_PATH,
        expected_source=render_task_source(),
        source_label="diagnostic",
    )


def _exact_source_and_notebook(
    source_path: Path,
    *,
    expected_path: Path,
    expected_source: str,
    source_label: str,
) -> tuple[bytes, str]:
    try:
        source = source_path.read_bytes()
    except OSError as exc:
        raise KagglePushOnceError(
            f"could not read {source_label} source: {exc}"
        ) from exc
    if len(source) > MAX_SOURCE_BYTES:
        raise KagglePushOnceError(f"{source_label} source exceeds the safety limit")
    expected = expected_source.encode("utf-8")
    if source != expected or source_path.resolve() != expected_path.resolve():
        raise KagglePushOnceError(
            f"only the current checked-in generated {source_label} source may be pushed"
        )
    try:
        import jupytext
    except ImportError as exc:
        raise KagglePushOnceError(
            "Kaggle CLI 2.2.4 or newer is required for jupytext conversion"
        ) from exc
    notebook = jupytext.reads(source.decode("utf-8"), fmt="py:percent")
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook_text = jupytext.writes(notebook, fmt="ipynb")
    return source, notebook_text


def _normalize_response(
    response: Any,
    *,
    expected_datasets: tuple[str, ...],
    expected_task: str = TASK_SLUG,
) -> dict[str, Any]:
    error = getattr(response, "error", None)
    if error:
        raise KagglePushOnceError(f"Kaggle rejected task creation: {error}")
    slug = getattr(response, "slug", None)
    actual_task = getattr(slug, "task_slug", None)
    version = getattr(slug, "version_number", None)
    source_kernel_id = getattr(response, "source_kernel_id", None)
    if actual_task != expected_task:
        raise KagglePushOnceError(
            f"Kaggle returned task {actual_task!r}, expected {expected_task!r}"
        )
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise KagglePushOnceError("Kaggle response has no positive task version")
    # Kaggle 2.2.4 can acknowledge the exact task version before its backing
    # kernel ID is populated. The task/version pair is the durable create
    # acknowledgement; the read-only output step resolves and verifies the
    # eventual kernel and the unique run for that exact version.
    source_kernel_id = _optional_positive_id(
        source_kernel_id,
        label="source kernel ID",
    )
    options = getattr(response, "options", None)
    actual_datasets = tuple(
        sorted(getattr(options, "dataset_data_sources", None) or ())
    )
    if actual_datasets != tuple(sorted(expected_datasets)):
        raise KagglePushOnceError(
            "Kaggle response dataset attachments differ from the request"
        )
    return {
        "owner": getattr(slug, "owner_slug", None) or None,
        "task": actual_task,
        "version": version,
        "sourceKernelId": source_kernel_id,
        "creationState": _enum_name(getattr(response, "creation_state", None)),
        "url": getattr(response, "url", None) or None,
        "datasets": list(actual_datasets),
    }


def push_diagnostic_once(
    *,
    source_path: Path,
    datasets: tuple[str, ...],
    gate: str,
    journal_path: Path,
) -> dict[str, Any]:
    """Submit one canonical task version without application-level retries."""

    if gate not in {"zero-call", "six-call"}:
        raise KagglePushOnceError(f"unknown diagnostic gate: {gate!r}")
    if any(not _DATASET_SLUG.fullmatch(dataset) for dataset in datasets):
        raise KagglePushOnceError(
            "each Kaggle dataset must use a lowercase OWNER/DATASET slug"
        )
    if len(set(datasets)) != len(datasets):
        raise KagglePushOnceError("duplicate Kaggle dataset attachments are forbidden")
    if gate == "zero-call" and datasets:
        raise KagglePushOnceError("zero-call gate must not attach a dataset")
    if gate == "six-call" and len(datasets) != 1:
        raise KagglePushOnceError(
            "six-call gate requires exactly one reviewed private dataset"
        )
    client_versions = _verified_client_versions()
    source, notebook_text = _source_and_notebook(source_path)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.benchmarks.types.benchmark_types import (
            BenchmarkTaskOptions,
        )
        from kagglesdk.benchmarks.types.benchmark_tasks_api_service import (
            ApiBenchmarkTaskSlug,
            ApiCreateBenchmarkTaskRequest,
            ApiGetBenchmarkTaskRequest,
        )
        from requests.exceptions import HTTPError
    except ImportError as exc:
        raise KagglePushOnceError(
            "Kaggle CLI 2.2.4 or newer is required for one-shot push"
        ) from exc

    request = ApiCreateBenchmarkTaskRequest()
    request.slug = TASK_SLUG
    request.text = notebook_text
    if datasets:
        options = BenchmarkTaskOptions()
        options.dataset_data_sources = list(datasets)
        request.options = options

    created_at = _utc_now()
    prepared = {
        "journalVersion": 2,
        "artifactKind": "kaggle_task_creation_dispatch",
        "operationId": str(uuid.uuid4()),
        "createdAt": created_at,
        "task": TASK_SLUG,
        "gate": gate,
        "datasets": list(datasets),
        "client": client_versions,
        "source": {
            "path": str(source_path.resolve()),
            "bytes": len(source),
            "sha256": hashlib.sha256(source).hexdigest(),
            "notebookSha256": hashlib.sha256(
                notebook_text.encode("utf-8")
            ).hexdigest(),
        },
    }

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        slug = ApiBenchmarkTaskSlug()
        slug.task_slug = TASK_SLUG
        get_request = ApiGetBenchmarkTaskRequest()
        get_request.slug = slug
        get_task = client.benchmarks.benchmark_tasks_api_client.get_benchmark_task

        def fetch_latest_task() -> Any | None:
            try:
                # Retrying this read is safe; only the create below is one-shot.
                return api.with_retry(get_task)(get_request)
            except HTTPError as exc:
                if _is_http_not_found(exc):
                    return None
                raise

        create = client.benchmarks.benchmark_tasks_api_client.create_benchmark_task
        # Deliberately do not wrap this non-idempotent call in KaggleApi.with_retry.
        return preflight_and_dispatch_once(
            fetch_latest_task=fetch_latest_task,
            create_task=create,
            request=request,
            journal_path=journal_path,
            prepared_journal=prepared,
            response_record=lambda response: _normalize_response(
                response, expected_datasets=datasets
            ),
        )


def push_capture_once(
    *,
    source_path: Path,
    datasets: tuple[str, ...],
    journal_path: Path,
) -> dict[str, Any]:
    """Submit the fixed six-call capture canary exactly once."""

    if any(not _DATASET_SLUG.fullmatch(dataset) for dataset in datasets):
        raise KagglePushOnceError(
            "each Kaggle dataset must use a lowercase OWNER/DATASET slug"
        )
    if len(datasets) != 1:
        raise KagglePushOnceError(
            "capture canary requires exactly one reviewed private dataset"
        )
    client_versions = _verified_client_versions()
    source, notebook_text = _exact_source_and_notebook(
        source_path,
        expected_path=CAPTURE_OUTPUT_PATH,
        expected_source=render_capture_task_source(),
        source_label="capture",
    )
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.benchmarks.types.benchmark_types import BenchmarkTaskOptions
        from kagglesdk.benchmarks.types.benchmark_tasks_api_service import (
            ApiBenchmarkTaskSlug,
            ApiCreateBenchmarkTaskRequest,
            ApiGetBenchmarkTaskRequest,
        )
        from requests.exceptions import HTTPError
    except ImportError as exc:
        raise KagglePushOnceError(
            "Kaggle CLI 2.2.4 or newer is required for one-shot push"
        ) from exc

    request = ApiCreateBenchmarkTaskRequest()
    request.slug = CAPTURE_TASK_SLUG
    request.text = notebook_text
    options = BenchmarkTaskOptions()
    options.dataset_data_sources = list(datasets)
    request.options = options

    prepared = {
        "journalVersion": 2,
        "artifactKind": "kaggle_task_creation_dispatch",
        "operationId": str(uuid.uuid4()),
        "createdAt": _utc_now(),
        "task": CAPTURE_TASK_SLUG,
        "gate": "six-call-capture",
        "datasets": list(datasets),
        "client": client_versions,
        "source": {
            "path": str(source_path.resolve()),
            "bytes": len(source),
            "sha256": hashlib.sha256(source).hexdigest(),
            "notebookSha256": hashlib.sha256(
                notebook_text.encode("utf-8")
            ).hexdigest(),
        },
    }

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        slug = ApiBenchmarkTaskSlug()
        slug.task_slug = CAPTURE_TASK_SLUG
        get_request = ApiGetBenchmarkTaskRequest()
        get_request.slug = slug
        get_task = client.benchmarks.benchmark_tasks_api_client.get_benchmark_task

        def fetch_latest_task() -> Any | None:
            try:
                return api.with_retry(get_task)(get_request)
            except HTTPError as exc:
                if _is_http_not_found(exc):
                    return None
                raise

        create = client.benchmarks.benchmark_tasks_api_client.create_benchmark_task
        return preflight_and_dispatch_once(
            fetch_latest_task=fetch_latest_task,
            create_task=create,
            request=request,
            journal_path=journal_path,
            prepared_journal=prepared,
            response_record=lambda response: _normalize_response(
                response,
                expected_datasets=datasets,
                expected_task=CAPTURE_TASK_SLUG,
            ),
            expected_task=CAPTURE_TASK_SLUG,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create exactly one Aleph Kaggle diagnostic task version without "
            "automatic retries. An ambiguous outcome must never be retried."
        )
    )
    parser.add_argument(
        "--source", type=Path, default=OUTPUT_PATH
    )
    parser.add_argument(
        "--task-kind", choices=("diagnostic", "capture"), default="diagnostic"
    )
    parser.add_argument("--gate", choices=("zero-call", "six-call"), required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.task_kind == "capture":
            if args.gate != "six-call":
                raise KagglePushOnceError(
                    "capture task-kind requires the six-call gate"
                )
            source_path = (
                CAPTURE_OUTPUT_PATH if args.source == OUTPUT_PATH else args.source
            )
            journal = push_capture_once(
                source_path=source_path,
                datasets=tuple(args.dataset),
                journal_path=args.journal,
            )
        else:
            journal = push_diagnostic_once(
                source_path=args.source,
                datasets=tuple(args.dataset),
                gate=args.gate,
                journal_path=args.journal,
            )
    except (OSError, ValueError) as exc:
        print(f"Kaggle one-shot push failed: {exc}", file=sys.stderr)
        return 1
    response = journal["response"]
    source_note = (
        f"source kernel {response['sourceKernelId']}"
        if response["sourceKernelId"] is not None
        else "source kernel pending exact-version readback"
    )
    print(
        f"submitted {response['task']} v{response['version']} once; "
        f"{source_note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
