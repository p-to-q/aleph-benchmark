from __future__ import annotations

import argparse
import math
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from bench.engine.kaggle_push_once import (
    KagglePushOnceError,
    _enum_name,
    _exception_record,
    _replace_journal,
    _utc_now,
    _verified_client_versions,
    _write_initial_journal,
)


_SLUG_PART = re.compile(
    r"^[a-z0-9](?:[a-z0-9_-]{0,98}[a-z0-9])?$"
)
_MODEL_SLUG = re.compile(
    r"^[a-z0-9](?:[a-z0-9._-]{0,198}[a-z0-9])?$"
)
_COMPLETED_TASK_STATE = "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED"
_TERMINAL_RUN_STATES = {
    "BENCHMARK_TASK_RUN_STATE_COMPLETED",
    "BENCHMARK_TASK_RUN_STATE_ERRORED",
}
DEFAULT_RECONCILE_DELAYS_SECONDS = (0.0, 1.0, 2.0, 4.0, 8.0)


class KaggleRunOnceError(KagglePushOnceError):
    """Raised before dispatch, or after a conclusively skipped schedule."""


class KaggleRunOutcomeAmbiguous(KaggleRunOnceError):
    """Raised after dispatch when the unique paid run cannot be proved."""


def _validate_target(
    *, owner: str, task: str, version: int, model: str, journal_path: Path
) -> None:
    if not _SLUG_PART.fullmatch(owner):
        raise KaggleRunOnceError("owner must be a lowercase Kaggle slug")
    if not _SLUG_PART.fullmatch(task):
        raise KaggleRunOnceError("task must be a lowercase Kaggle slug")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise KaggleRunOnceError("version must be a positive integer")
    if not _MODEL_SLUG.fullmatch(model):
        raise KaggleRunOnceError(
            "model must be one exact canonical BenchmarkModelVersion slug; "
            "provider paths and @ aliases are not accepted"
        )
    if journal_path.exists():
        raise KaggleRunOnceError(
            f"run journal already exists; inspect it and do not retry: {journal_path}"
        )


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KaggleRunOnceError(f"Kaggle returned an invalid {label}")
    return value


def _iso_datetime(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise KaggleRunOnceError(f"Kaggle returned an invalid {label}")
    if value.tzinfo is None:
        # kagglesdk 0.1.37 drops the offset from service timestamps whose
        # documented semantics are UTC. Retain that compatibility repair in
        # the receipt instead of serializing a misleading local time.
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _task_record(
    task_info: Any,
    *,
    expected_owner: str,
    expected_task: str,
    expected_version: int,
) -> dict[str, Any]:
    slug = getattr(task_info, "slug", None)
    owner = getattr(slug, "owner_slug", None)
    task = getattr(slug, "task_slug", None)
    version = getattr(slug, "version_number", None)
    if (owner, task, version) != (
        expected_owner,
        expected_task,
        expected_version,
    ):
        raise KaggleRunOnceError(
            "Kaggle exact-version preflight returned a different task identity"
        )
    state = _enum_name(getattr(task_info, "creation_state", None))
    if state != _COMPLETED_TASK_STATE:
        raise KaggleRunOnceError(
            "exact Kaggle task version is not ready to run "
            f"({state})"
        )
    return {
        "owner": owner,
        "task": task,
        "version": version,
        "creationState": state,
        "sourceKernelId": (
            _positive_int(
                getattr(task_info, "source_kernel_id", None),
                label="source kernel ID",
            )
            if getattr(task_info, "source_kernel_id", None)
            else None
        ),
        "url": getattr(task_info, "url", None) or None,
    }


def _run_record(
    run: Any,
    *,
    expected_owner: str,
    expected_task: str,
    expected_version: int,
) -> dict[str, Any]:
    slug = getattr(run, "task_slug", None)
    identity = (
        getattr(slug, "owner_slug", None),
        getattr(slug, "task_slug", None),
        getattr(slug, "version_number", None),
    )
    if identity != (expected_owner, expected_task, expected_version):
        raise KaggleRunOnceError(
            "Kaggle run listing escaped the requested exact task version"
        )
    model = getattr(run, "model_version_slug", None)
    if not isinstance(model, str) or not _MODEL_SLUG.fullmatch(model):
        raise KaggleRunOnceError("Kaggle returned an invalid run model slug")
    error_message = getattr(run, "error_message", None)
    if error_message is not None and not isinstance(error_message, str):
        raise KaggleRunOnceError("Kaggle returned an invalid run error message")
    return {
        "id": _positive_int(getattr(run, "id", None), label="run ID"),
        "model": model,
        "state": _enum_name(getattr(run, "state", None)),
        "startTime": _iso_datetime(
            getattr(run, "start_time", None), label="run start time"
        ),
        "endTime": _iso_datetime(
            getattr(run, "end_time", None), label="run end time"
        ),
        "errorMessage": error_message or None,
    }


def _run_set(
    runs: Iterable[Any],
    *,
    expected_owner: str,
    expected_task: str,
    expected_version: int,
) -> list[dict[str, Any]]:
    records = [
        _run_record(
            run,
            expected_owner=expected_owner,
            expected_task=expected_task,
            expected_version=expected_version,
        )
        for run in runs
    ]
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise KaggleRunOnceError("Kaggle returned duplicate run IDs")
    return sorted(records, key=lambda record: record["id"])


def _model_record(model_info: Any, *, expected_model: str) -> dict[str, Any]:
    version = getattr(model_info, "version", None)
    slug = getattr(version, "slug", None)
    if slug != expected_model:
        raise KaggleRunOnceError(
            "Kaggle model preflight returned a different canonical version slug"
        )
    if getattr(version, "published", None) is not True:
        raise KaggleRunOnceError("requested Kaggle model version is not published")
    if getattr(version, "allow_model_proxy", None) is not True:
        raise KaggleRunOnceError(
            "requested Kaggle model version does not allow Model Proxy execution"
        )
    proxy_slug = getattr(version, "model_proxy_slug", None)
    if not isinstance(proxy_slug, str) or not proxy_slug:
        raise KaggleRunOnceError(
            "requested Kaggle model version has no Model Proxy identity"
        )
    return {
        "benchmarkModelId": _positive_int(
            getattr(model_info, "id", None), label="benchmark model ID"
        ),
        "benchmarkModelVersionId": _positive_int(
            getattr(version, "id", None), label="benchmark model version ID"
        ),
        "slug": slug,
        "modelProxySlug": proxy_slug,
        "displayName": getattr(version, "display_name", None) or None,
        "deprecatedAt": _iso_datetime(
            getattr(version, "deprecation_time", None),
            label="model deprecation time",
        ),
    }


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KaggleRunOnceError(f"Kaggle returned an invalid {label}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise KaggleRunOnceError(f"Kaggle returned an invalid {label}")
    return number


def _quota_record(response: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    periods: set[str] = set()
    for balance in getattr(response, "quota_balances", None) or ():
        period = _enum_name(getattr(balance, "refill_period", None))
        if period not in {"DAILY", "MONTHLY"} or period in periods:
            raise KaggleRunOnceError(
                "Kaggle returned duplicate or unknown Model Proxy quota periods"
            )
        periods.add(period)
        used = _finite_number(
            getattr(balance, "quota_used", None), label="quota used"
        )
        allowed = _finite_number(
            getattr(balance, "total_quota_allowed", None),
            label="quota allowance",
        )
        if used > allowed:
            raise KaggleRunOnceError("Kaggle quota used exceeds its allowance")
        records.append(
            {
                "period": period,
                "usedUsd": used,
                "allowedUsd": allowed,
                "remainingUsd": allowed - used,
                "refillTime": _iso_datetime(
                    getattr(balance, "refill_time", None),
                    label="quota refill time",
                ),
            }
        )
    if periods != {"DAILY", "MONTHLY"}:
        raise KaggleRunOnceError(
            "Kaggle did not return both daily and monthly Model Proxy quotas"
        )
    return sorted(records, key=lambda record: record["period"])


def _schedule_response_record(
    response: Any, *, expected_model_version_id: int
) -> dict[str, Any]:
    results = list(getattr(response, "results", None) or ())
    if len(results) != 1:
        raise KaggleRunOnceError(
            "Kaggle schedule response did not contain exactly one result"
        )
    result = results[0]
    scheduled = getattr(result, "run_scheduled", None)
    if not isinstance(scheduled, bool):
        raise KaggleRunOnceError("Kaggle returned an invalid scheduled flag")
    task_version_id = _positive_int(
        getattr(result, "benchmark_task_version_id", None),
        label="internal task version ID",
    )
    model_version_id = _positive_int(
        getattr(result, "benchmark_model_version_id", None),
        label="internal model version ID",
    )
    if model_version_id != expected_model_version_id:
        raise KaggleRunOnceError(
            "Kaggle schedule response identified a different model version"
        )
    parent_id = getattr(result, "parent_task_version_id", None)
    if parent_id:
        _positive_int(parent_id, label="parent task version ID")
        raise KaggleRunOnceError(
            "Kaggle redirected scheduling to a parent task version"
        )
    reason = getattr(result, "run_skipped_reason", None)
    if reason is not None and not isinstance(reason, str):
        raise KaggleRunOnceError("Kaggle returned an invalid skip reason")
    return {
        "runScheduled": scheduled,
        "runSkippedReason": reason or None,
        "benchmarkTaskVersionId": task_version_id,
        "benchmarkModelVersionId": model_version_id,
        "parentTaskVersionId": None,
    }


def _new_runs(
    before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    before_ids = {record["id"] for record in before}
    if not before_ids.issubset({record["id"] for record in after}):
        raise KaggleRunOnceError(
            "Kaggle post-dispatch run listing lost pre-existing run IDs"
        )
    return [record for record in after if record["id"] not in before_ids]


def schedule_and_reconcile_once(
    *,
    owner: str,
    task: str,
    version: int,
    model: str,
    journal_path: Path,
    fetch_task: Callable[[], Any],
    fetch_model: Callable[[], Any],
    fetch_runs: Callable[[], Iterable[Any]],
    fetch_quota: Callable[[], Any],
    schedule_run: Callable[[], Any],
    client_versions: dict[str, str],
    reconcile_delays: Sequence[float] = DEFAULT_RECONCILE_DELAYS_SECONDS,
    clock: Callable[[], str] = _utc_now,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Cross one paid scheduling boundary and bind its unique run ID."""

    _validate_target(
        owner=owner,
        task=task,
        version=version,
        model=model,
        journal_path=journal_path,
    )
    if not reconcile_delays or any(
        isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not math.isfinite(float(delay))
        or delay < 0
        for delay in reconcile_delays
    ):
        raise KaggleRunOnceError("reconciliation delays must be finite and non-negative")

    exact_task = _task_record(
        fetch_task(),
        expected_owner=owner,
        expected_task=task,
        expected_version=version,
    )
    exact_model = _model_record(fetch_model(), expected_model=model)
    before_runs = _run_set(
        fetch_runs(),
        expected_owner=owner,
        expected_task=task,
        expected_version=version,
    )
    active = [
        record
        for record in before_runs
        if record["state"] not in _TERMINAL_RUN_STATES
    ]
    if active:
        raise KaggleRunOnceError(
            "exact task version has unresolved queued, running, or unknown runs"
        )
    quota_before = _quota_record(fetch_quota())

    journal: dict[str, Any] = {
        "journalVersion": 1,
        "artifactKind": "kaggle_benchmark_run_dispatch",
        "operationId": str(uuid.uuid4()),
        "createdAt": clock(),
        "updatedAt": clock(),
        "state": "prepared",
        "target": {
            "owner": owner,
            "task": task,
            "version": version,
            "model": model,
        },
        "client": dict(client_versions),
        "remotePreflight": {
            "observedAt": clock(),
            "task": exact_task,
            "model": exact_model,
            "runs": before_runs,
            "quota": quota_before,
        },
        "response": None,
        "reconciliation": None,
        "dispatchFailure": None,
        "responseFailure": None,
        "failure": None,
    }
    _write_initial_journal(journal_path, journal)

    # This durable state is the no-retry barrier for the only paid call.
    journal["state"] = "dispatching"
    journal["updatedAt"] = clock()
    _replace_journal(journal_path, journal)

    dispatch_failure: dict[str, str] | None = None
    response_failure: dict[str, str] | None = None
    try:
        response = schedule_run()
    except BaseException as exc:
        dispatch_failure = _exception_record(exc)
        journal["dispatchFailure"] = dispatch_failure
        journal["failure"] = dispatch_failure
        journal["state"] = "ambiguous"
        journal["updatedAt"] = clock()
        _replace_journal(journal_path, journal)
        if not isinstance(exc, Exception):
            raise
    else:
        try:
            journal["response"] = _schedule_response_record(
                response,
                expected_model_version_id=exact_model[
                    "benchmarkModelVersionId"
                ],
            )
        except Exception as exc:
            response_failure = _exception_record(exc)
            journal["responseFailure"] = response_failure
            journal["failure"] = response_failure
            journal["state"] = "ambiguous"
            journal["updatedAt"] = clock()
            _replace_journal(journal_path, journal)
        else:
            if not journal["response"]["runScheduled"]:
                journal["state"] = "not_scheduled"
                journal["updatedAt"] = clock()
                journal["failure"] = {
                    "type": "RunSkipped",
                    "message": journal["response"]["runSkippedReason"]
                    or "Kaggle did not schedule the requested run",
                }
                try:
                    journal["quotaAfter"] = _quota_record(fetch_quota())
                except Exception as exc:
                    journal["quotaAfterFailure"] = _exception_record(exc)
                _replace_journal(journal_path, journal)
                raise KaggleRunOnceError(
                    "Kaggle conclusively skipped the requested run; "
                    "inspect the journal"
                )
            journal["state"] = "returned_unreconciled"
            journal["updatedAt"] = clock()
            _replace_journal(journal_path, journal)

    observations: list[dict[str, Any]] = []
    for attempt, delay in enumerate(reconcile_delays, start=1):
        if delay:
            sleeper(float(delay))
        observed_at = clock()
        try:
            after_runs = _run_set(
                fetch_runs(),
                expected_owner=owner,
                expected_task=task,
                expected_version=version,
            )
            delta = _new_runs(before_runs, after_runs)
        except Exception as exc:
            observations.append(
                {
                    "attempt": attempt,
                    "observedAt": observed_at,
                    "delaySeconds": float(delay),
                    "failure": _exception_record(exc),
                }
            )
            continue

        observations.append(
            {
                "attempt": attempt,
                "observedAt": observed_at,
                "delaySeconds": float(delay),
                "runIds": [record["id"] for record in after_runs],
                "newRuns": delta,
            }
        )
        if not delta:
            continue
        if len(delta) != 1 or delta[0]["model"] != model:
            journal["reconciliation"] = {
                "observations": observations,
                "run": None,
            }
            journal["state"] = "ambiguous"
            journal["updatedAt"] = clock()
            journal["failure"] = {
                "type": "RunSetMismatch",
                "message": (
                    "post-dispatch run-set difference was not exactly one "
                    "run for the requested model"
                ),
            }
            _replace_journal(journal_path, journal)
            raise KaggleRunOutcomeAmbiguous(
                "paid scheduling outcome is ambiguous; inspect the journal and "
                "do not schedule again"
            )

        if response_failure is not None:
            journal["reconciliation"] = {
                "observations": observations,
                "run": delta[0],
            }
            journal["state"] = "ambiguous"
            journal["updatedAt"] = clock()
            journal["failure"] = response_failure
            _replace_journal(journal_path, journal)
            raise KaggleRunOutcomeAmbiguous(
                "Kaggle returned a contradictory schedule acknowledgement; "
                "the new run is recorded but must not be promoted or retried"
            )

        journal["reconciliation"] = {
            "observations": observations,
            "run": delta[0],
        }
        journal["state"] = "reconciled"
        journal["updatedAt"] = clock()
        journal["failure"] = None
        try:
            journal["quotaAfter"] = _quota_record(fetch_quota())
        except Exception as exc:
            journal["quotaAfter"] = None
            journal["quotaAfterFailure"] = _exception_record(exc)
        _replace_journal(journal_path, journal)
        return journal

    journal["reconciliation"] = {
        "observations": observations,
        "run": None,
    }
    journal["state"] = "ambiguous"
    journal["updatedAt"] = clock()
    if dispatch_failure is None and response_failure is None:
        journal["failure"] = {
            "type": "RunNotObserved",
            "message": "no new run became visible within the reconciliation window",
        }
    try:
        journal["quotaAfter"] = _quota_record(fetch_quota())
    except Exception as exc:
        journal["quotaAfter"] = None
        journal["quotaAfterFailure"] = _exception_record(exc)
    _replace_journal(journal_path, journal)
    raise KaggleRunOutcomeAmbiguous(
        "paid scheduling outcome is ambiguous; inspect the exact task version, "
        "quota, and journal; do not schedule again"
    )


def run_once(
    *,
    owner: str,
    task: str,
    version: int,
    model: str,
    journal_path: Path,
    reconcile_delays: Sequence[float] = DEFAULT_RECONCILE_DELAYS_SECONDS,
) -> dict[str, Any]:
    """Schedule one exact Kaggle Task/model pair without paid-call retry."""

    _validate_target(
        owner=owner,
        task=task,
        version=version,
        model=model,
        journal_path=journal_path,
    )
    client_versions = _verified_client_versions()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.benchmarks.types.benchmark_tasks_api_service import (
            ApiBatchScheduleBenchmarkTaskRunsRequest,
            ApiBenchmarkTaskSlug,
            ApiGetBenchmarkTaskRequest,
            ApiListBenchmarkTaskRunsRequest,
        )
        from kagglesdk.benchmarks.types.benchmarks_api_service import (
            ApiListBenchmarkModelsRequest,
        )
        from kagglesdk.models.types.model_proxy_api_service import (
            ApiGetModelProxyQuotasRequest,
        )
    except ImportError as exc:
        raise KaggleRunOnceError(
            "Kaggle CLI 2.2.4 or newer is required for one-shot scheduling"
        ) from exc

    slug = ApiBenchmarkTaskSlug()
    slug.owner_slug = owner
    slug.task_slug = task
    slug.version_number = version

    get_request = ApiGetBenchmarkTaskRequest()
    get_request.slug = slug
    schedule_request = ApiBatchScheduleBenchmarkTaskRunsRequest()
    schedule_request.task_slugs = [slug]
    schedule_request.model_version_slugs = [model]

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        tasks = client.benchmarks.benchmark_tasks_api_client

        def fetch_task() -> Any:
            return api.with_retry(tasks.get_benchmark_task)(get_request)

        def fetch_model() -> Any:
            matches: list[Any] = []
            page_token = ""
            seen_tokens: set[str] = set()
            models = client.benchmarks.benchmarks_api_client
            for _ in range(100):
                request = ApiListBenchmarkModelsRequest()
                request.page_size = 100
                if page_token:
                    request.page_token = page_token
                response = api.with_retry(models.list_benchmark_models)(request)
                for candidate in getattr(response, "benchmark_models", None) or ():
                    if getattr(getattr(candidate, "version", None), "slug", None) == model:
                        matches.append(candidate)
                page_token = getattr(response, "next_page_token", None) or ""
                if not page_token:
                    break
                if page_token in seen_tokens:
                    raise KaggleRunOnceError(
                        "Kaggle model pagination repeated a page token"
                    )
                seen_tokens.add(page_token)
            else:
                raise KaggleRunOnceError(
                    "Kaggle model pagination exceeded 100 pages"
                )
            if len(matches) != 1:
                raise KaggleRunOnceError(
                    "exact canonical Kaggle model version was missing or duplicated"
                )
            return matches[0]

        def fetch_runs() -> list[Any]:
            runs: list[Any] = []
            page_token = ""
            seen_tokens: set[str] = set()
            for _ in range(100):
                request = ApiListBenchmarkTaskRunsRequest()
                request.task_slug = slug
                request.page_size = 100
                if page_token:
                    request.page_token = page_token
                response = api.with_retry(tasks.list_benchmark_task_runs)(request)
                runs.extend(getattr(response, "runs", None) or ())
                page_token = getattr(response, "next_page_token", None) or ""
                if not page_token:
                    return runs
                if page_token in seen_tokens:
                    raise KaggleRunOnceError(
                        "Kaggle run pagination repeated a page token"
                    )
                seen_tokens.add(page_token)
            raise KaggleRunOnceError("Kaggle run pagination exceeded 100 pages")

        def fetch_quota() -> Any:
            quotas = client.models.model_proxy_api_client
            return api.with_retry(quotas.get_model_proxy_quotas)(
                ApiGetModelProxyQuotasRequest()
            )

        schedule = tasks.batch_schedule_benchmark_task_runs
        return schedule_and_reconcile_once(
            owner=owner,
            task=task,
            version=version,
            model=model,
            journal_path=journal_path,
            fetch_task=fetch_task,
            fetch_model=fetch_model,
            fetch_runs=fetch_runs,
            fetch_quota=fetch_quota,
            # Deliberately never wrap this paid, non-idempotent POST in
            # KaggleApi.with_retry.
            schedule_run=lambda: schedule(schedule_request),
            client_versions=client_versions,
            reconcile_delays=reconcile_delays,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Schedule one exact Kaggle benchmark Task version/model pair. "
            "The paid call is never retried."
        )
    )
    parser.add_argument("--owner", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--journal", required=True, type=Path)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        journal = run_once(
            owner=args.owner,
            task=args.task,
            version=args.version,
            model=args.model,
            journal_path=args.journal,
        )
    except (OSError, ValueError) as exc:
        print(f"Kaggle one-shot run failed: {exc}", file=sys.stderr)
        return 1
    run = journal["reconciliation"]["run"]
    print(
        f"scheduled {args.owner}/{args.task} v{args.version} once against "
        f"{args.model}; reconciled run {run['id']} ({run['state']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
