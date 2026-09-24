from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .kaggle_capture import MAX_CAPTURE_BYTES, parse_capture_payload
from .kaggle_creation_output import (
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    _download_creation_archive,
    _parse_task_slug,
    _run_metadata,
    _safe_archive_entry,
    _task_metadata,
    _write_exclusive,
)
from .schema_validation import SchemaValidationError, load_schema, validate


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_SCHEMA_PATH = (
    REPO_ROOT
    / "schemas/v0.2/aleph-bench-kaggle-capture-evidence.schema.json"
)
EVIDENCE_SCHEMA_VERSION = "1.1.0"
LEGACY_EVIDENCE_SCHEMA_VERSION = "1.0.0"
SUPPORTED_EVIDENCE_SCHEMA_VERSIONS = {
    LEGACY_EVIDENCE_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
}
ARTIFACT_KIND = "aleph_bench_kaggle_capture_evidence"
TARGET_PROTOCOL_VERSION = "0.2.0"
ARTIFACT_ID_PREFIX = (
    "aleph-bench-kaggle-capture-evidence-v1-artifact-"
)
CAPTURE_FILENAME = "aleph-bench-v0.2-kaggle-capture-canary.json"
EXPECTED_DATASET_SLUG = "aleph-bench-v02-scorer-conformance"
MAX_SOURCE_BYTES = 1_048_576
MAX_NOTEBOOK_BYTES = 16_777_216
MAX_DISPATCH_JOURNAL_BYTES = 16_777_216


class KaggleCaptureEvidenceError(ValueError):
    """Raised when a Kaggle capture cannot be bound to one exact run."""


def _fail(message: str) -> None:
    raise KaggleCaptureEvidenceError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise KaggleCaptureEvidenceError(
            "capture evidence cannot be canonically encoded"
        ) from None


def _artifact_id(evidence: dict[str, Any]) -> str:
    body = {key: value for key, value in evidence.items() if key != "id"}
    return ARTIFACT_ID_PREFIX + _sha256(_canonical_json_bytes(body))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_time(value: Any, *, role: str) -> datetime:
    if not isinstance(value, str):
        raise KaggleCaptureEvidenceError(
            f"{role} is not a valid timestamp"
        )
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except (TypeError, ValueError):
        raise KaggleCaptureEvidenceError(
            f"{role} is not a valid timestamp"
        ) from None
    if parsed.utcoffset() is None:
        _fail(f"{role} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            _fail("notebook source contains a duplicate JSON key")
        value[key] = child
    return value


def _reject_json_constant(_value: str) -> None:
    _fail("notebook source contains a non-finite JSON constant")


def _reject_dispatch_journal_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            _fail("dispatch journal contains a duplicate JSON key")
        value[key] = child
    return value


def _reject_dispatch_journal_json_constant(_value: str) -> None:
    _fail("dispatch journal contains a non-finite JSON constant")


def _dispatch_journal_field(
    record: dict[str, Any], key: str, *, role: str
) -> Any:
    if key not in record:
        _fail(f"dispatch journal {role} is missing {key}")
    return record[key]


def _dispatch_journal_object(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"dispatch journal {role} must be an object")
    return value


def _dispatch_journal_positive_int(value: Any, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"dispatch journal {role} must be a positive integer")
    return value


def _dispatch_journal_failure_record(
    value: Any, *, role: str
) -> dict[str, str]:
    record = _dispatch_journal_object(value, role=role)
    if set(record) != {"type", "message"}:
        _fail(f"dispatch journal has an invalid {role} record")
    failure_type = record["type"]
    failure_message = record["message"]
    if (
        not isinstance(failure_type, str)
        or not failure_type
        or len(failure_type) > 200
        or not isinstance(failure_message, str)
        or len(failure_message) > 2_000
    ):
        _fail(f"dispatch journal has an invalid {role} record")
    return record


def _dispatch_journal_run_record(
    value: Any, *, role: str
) -> dict[str, Any]:
    record = _dispatch_journal_object(value, role=role)
    expected_fields = {
        "id",
        "model",
        "state",
        "startTime",
        "endTime",
        "errorMessage",
    }
    if set(record) != expected_fields:
        _fail(f"dispatch journal {role} has invalid fields")
    _dispatch_journal_positive_int(record["id"], role=f"{role} ID")
    for field in ("model", "state"):
        if not isinstance(record[field], str) or not record[field]:
            _fail(f"dispatch journal {role} has an invalid {field}")
    for field in ("startTime", "endTime"):
        if record[field] is not None:
            _parse_time(record[field], role=f"dispatch journal {role}.{field}")
    if record["errorMessage"] is not None and not isinstance(
        record["errorMessage"], str
    ):
        _fail(f"dispatch journal {role} has an invalid errorMessage")
    return record


def _dispatch_journal_reconciliation_observations(
    value: Any,
    *,
    preflight_runs: list[dict[str, Any]],
    reconciled_run: dict[str, Any],
) -> None:
    if not isinstance(value, list) or not value:
        _fail("dispatch journal reconciliation has no observations")
    preflight_ids = [record["id"] for record in preflight_runs]
    preflight_id_set = set(preflight_ids)
    reconciled_id = reconciled_run["id"]
    for index, raw_observation in enumerate(value, start=1):
        role = f"reconciliation observation {index}"
        observation = _dispatch_journal_object(raw_observation, role=role)
        if _dispatch_journal_positive_int(
            _dispatch_journal_field(observation, "attempt", role=role),
            role=f"{role} attempt",
        ) != index:
            _fail("dispatch journal reconciliation attempts are not sequential")
        _parse_time(
            _dispatch_journal_field(observation, "observedAt", role=role),
            role=f"dispatch journal {role}.observedAt",
        )
        delay = _dispatch_journal_field(
            observation, "delaySeconds", role=role
        )
        if (
            isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not math.isfinite(float(delay))
            or delay < 0
        ):
            _fail(f"dispatch journal {role} has an invalid delaySeconds")

        if "failure" in observation:
            if set(observation) != {
                "attempt",
                "observedAt",
                "delaySeconds",
                "failure",
            }:
                _fail(f"dispatch journal {role} has invalid failure fields")
            _dispatch_journal_failure_record(
                observation["failure"], role=f"{role} failure"
            )
            continue

        if set(observation) != {
            "attempt",
            "observedAt",
            "delaySeconds",
            "runIds",
            "newRuns",
        }:
            _fail(f"dispatch journal {role} has invalid success fields")
        run_ids = _dispatch_journal_field(observation, "runIds", role=role)
        if not isinstance(run_ids, list):
            _fail(f"dispatch journal {role} runIds must be an array")
        parsed_run_ids = [
            _dispatch_journal_positive_int(
                run_id, role=f"{role} run ID"
            )
            for run_id in run_ids
        ]
        if parsed_run_ids != sorted(set(parsed_run_ids)):
            _fail(f"dispatch journal {role} runIds are not ordered and unique")
        if not preflight_id_set.issubset(parsed_run_ids):
            _fail(f"dispatch journal {role} lost a preflight run ID")
        new_runs_value = _dispatch_journal_field(
            observation, "newRuns", role=role
        )
        if not isinstance(new_runs_value, list):
            _fail(f"dispatch journal {role} newRuns must be an array")
        new_runs = [
            _dispatch_journal_run_record(
                record, role=f"{role} newRuns entry"
            )
            for record in new_runs_value
        ]
        new_ids = [record["id"] for record in new_runs]
        expected_new_ids = [
            run_id for run_id in parsed_run_ids if run_id not in preflight_id_set
        ]
        if new_ids != expected_new_ids:
            _fail(f"dispatch journal {role} newRuns differs from its run-set delta")
        if index < len(value) and new_runs:
            _fail("dispatch journal continued after observing a new run")

    final_observation = value[-1]
    if "failure" in final_observation:
        _fail("dispatch journal final observation is not successful")
    expected_final_ids = sorted([*preflight_ids, reconciled_id])
    if final_observation["runIds"] != expected_final_ids:
        _fail("dispatch journal final runIds differs from preflight plus bound run")
    if final_observation["newRuns"] != [reconciled_run]:
        _fail("dispatch journal final observation is not one exact new run")


def _read_dispatch_journal(path: Path) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(MAX_DISPATCH_JOURNAL_BYTES + 1)
    if not data or len(data) > MAX_DISPATCH_JOURNAL_BYTES:
        _fail("dispatch journal exceeds the safety limit")
    return data


def _dispatch_journal_catalog_record(
    journal_bytes: bytes,
    *,
    task: dict[str, Any],
    run: dict[str, Any],
    observed_model: str,
) -> dict[str, Any]:
    """Verify one reconciled scheduler receipt and return its exact alias record."""

    if not journal_bytes or len(journal_bytes) > MAX_DISPATCH_JOURNAL_BYTES:
        _fail("dispatch journal exceeds the safety limit")
    try:
        journal = json.loads(
            journal_bytes.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_dispatch_journal_duplicate_pairs,
            parse_constant=_reject_dispatch_journal_json_constant,
        )
    except KaggleCaptureEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        _fail("dispatch journal is not bounded strict JSON")
    journal = _dispatch_journal_object(journal, role="root")
    journal_version = _dispatch_journal_field(
        journal, "journalVersion", role="root"
    )
    if (
        isinstance(journal_version, bool)
        or not isinstance(journal_version, int)
        or journal_version != 1
    ):
        _fail("dispatch journal version is unsupported")
    if (
        _dispatch_journal_field(journal, "artifactKind", role="root")
        != "kaggle_benchmark_run_dispatch"
    ):
        _fail("dispatch journal artifact kind is invalid")
    if _dispatch_journal_field(journal, "state", role="root") != "reconciled":
        _fail("dispatch journal is not reconciled")
    dispatch_failure = _dispatch_journal_field(
        journal, "dispatchFailure", role="root"
    )
    if (
        _dispatch_journal_field(journal, "responseFailure", role="root")
        is not None
    ):
        _fail("dispatch journal retains responseFailure")
    if _dispatch_journal_field(journal, "failure", role="root") is not None:
        _fail("dispatch journal retains failure")

    operation_id = _dispatch_journal_field(
        journal, "operationId", role="root"
    )
    if not isinstance(operation_id, str):
        _fail("dispatch journal operationId is invalid")
    try:
        parsed_operation_id = str(uuid.UUID(operation_id))
    except (ValueError, AttributeError):
        _fail("dispatch journal operationId is invalid")
    if parsed_operation_id != operation_id:
        _fail("dispatch journal operationId is not canonical")

    target = _dispatch_journal_object(
        _dispatch_journal_field(journal, "target", role="root"),
        role="target",
    )
    target_identity = (
        _dispatch_journal_field(target, "owner", role="target"),
        _dispatch_journal_field(target, "task", role="target"),
        _dispatch_journal_positive_int(
            _dispatch_journal_field(target, "version", role="target"),
            role="target version",
        ),
    )
    if target_identity != (task["owner"], task["slug"], task["version"]):
        _fail("dispatch journal targets a different Kaggle task version")
    scheduled_slug = _dispatch_journal_field(target, "model", role="target")
    if scheduled_slug != run["modelVersionSlug"]:
        _fail("dispatch journal targets a different Kaggle run model")

    remote = _dispatch_journal_object(
        _dispatch_journal_field(journal, "remotePreflight", role="root"),
        role="remotePreflight",
    )
    remote_task = _dispatch_journal_object(
        _dispatch_journal_field(remote, "task", role="remotePreflight"),
        role="remotePreflight.task",
    )
    remote_task_identity = (
        _dispatch_journal_field(
            remote_task, "owner", role="remotePreflight.task"
        ),
        _dispatch_journal_field(
            remote_task, "task", role="remotePreflight.task"
        ),
        _dispatch_journal_positive_int(
            _dispatch_journal_field(
                remote_task, "version", role="remotePreflight.task"
            ),
            role="preflight task version",
        ),
    )
    if remote_task_identity != target_identity:
        _fail("dispatch journal preflight task differs from its target")
    remote_model = _dispatch_journal_object(
        _dispatch_journal_field(remote, "model", role="remotePreflight"),
        role="remotePreflight.model",
    )
    if (
        _dispatch_journal_field(
            remote_model, "slug", role="remotePreflight.model"
        )
        != scheduled_slug
    ):
        _fail("dispatch journal model catalog slug differs from its target")
    benchmark_model_id = _dispatch_journal_positive_int(
        _dispatch_journal_field(
            remote_model, "benchmarkModelId", role="remotePreflight.model"
        ),
        role="benchmark model ID",
    )
    benchmark_model_version_id = _dispatch_journal_positive_int(
        _dispatch_journal_field(
            remote_model,
            "benchmarkModelVersionId",
            role="remotePreflight.model",
        ),
        role="benchmark model version ID",
    )
    proxy_slug = _dispatch_journal_field(
        remote_model, "modelProxySlug", role="remotePreflight.model"
    )
    if not isinstance(proxy_slug, str) or not proxy_slug:
        _fail("dispatch journal Model Proxy slug is invalid")
    if proxy_slug != observed_model:
        _fail("dispatch journal Model Proxy slug differs from runtime observation")

    before_runs = _dispatch_journal_field(
        remote, "runs", role="remotePreflight"
    )
    if not isinstance(before_runs, list):
        _fail("dispatch journal preflight runs must be an array")
    before_runs = [
        _dispatch_journal_run_record(
            before_run, role="remotePreflight.runs entry"
        )
        for before_run in before_runs
    ]
    before_ids = [before_run["id"] for before_run in before_runs]
    if before_ids != sorted(set(before_ids)):
        _fail("dispatch journal preflight run IDs are not ordered and unique")
    if run["id"] in before_ids:
        _fail("dispatch journal preflight already contained the bound run")

    response_value = _dispatch_journal_field(journal, "response", role="root")
    if dispatch_failure is None:
        response = _dispatch_journal_object(response_value, role="response")
        if (
            _dispatch_journal_field(response, "runScheduled", role="response")
            is not True
        ):
            _fail("dispatch journal does not prove a scheduled run")
        response_model_version_id = _dispatch_journal_positive_int(
            _dispatch_journal_field(
                response, "benchmarkModelVersionId", role="response"
            ),
            role="response model version ID",
        )
        if response_model_version_id != benchmark_model_version_id:
            _fail("dispatch journal response model version ID drifted")
        if (
            _dispatch_journal_field(
                response, "runSkippedReason", role="response"
            )
            is not None
            or _dispatch_journal_field(
                response, "parentTaskVersionId", role="response"
            )
            is not None
        ):
            _fail(
                "dispatch journal response did not schedule the exact task version"
            )
        _dispatch_journal_positive_int(
            _dispatch_journal_field(
                response, "benchmarkTaskVersionId", role="response"
            ),
            role="internal task version ID",
        )
    else:
        _dispatch_journal_failure_record(
            dispatch_failure, role="dispatchFailure"
        )
        if response_value is not None:
            _fail("dispatch journal has both a response and dispatchFailure")

    reconciliation = _dispatch_journal_object(
        _dispatch_journal_field(journal, "reconciliation", role="root"),
        role="reconciliation",
    )
    reconciled_run = _dispatch_journal_run_record(
        _dispatch_journal_field(reconciliation, "run", role="reconciliation"),
        role="reconciliation.run",
    )
    reconciled_identity = (
        _dispatch_journal_positive_int(
            _dispatch_journal_field(
                reconciled_run, "id", role="reconciliation.run"
            ),
            role="reconciled run ID",
        ),
        _dispatch_journal_field(
            reconciled_run, "model", role="reconciliation.run"
        ),
    )
    if reconciled_identity != (run["id"], scheduled_slug):
        _fail("dispatch journal reconciles a different Kaggle run")
    _dispatch_journal_reconciliation_observations(
        _dispatch_journal_field(
            reconciliation, "observations", role="reconciliation"
        ),
        preflight_runs=before_runs,
        reconciled_run=reconciled_run,
    )

    return {
        "operationId": operation_id,
        "benchmarkModelId": benchmark_model_id,
        "benchmarkModelVersionId": benchmark_model_version_id,
        "scheduledSlug": scheduled_slug,
        "modelProxySlug": proxy_slug,
    }


def _expected_capture_contract() -> tuple[dict[str, Any], bytes]:
    """Return the frozen generated-task contract on the authority runtime."""

    from bench.tasks.kaggle.generate_v0_2_capture import (
        TASK_NAME,
        TASK_SOURCE_PATH,
        TASK_VERSION,
        _generation_context,
        _render_task_source,
    )

    try:
        context = _generation_context()
    except RuntimeError:
        raise KaggleCaptureEvidenceError(
            "capture evidence binding requires the frozen authority runtime"
        ) from None
    source = _render_task_source(
        definition_sha256=context["definitionSha256"],
        implementation_sha256=context["implementationSha256"],
        dataset_identity=context["datasetIdentity"],
        package_identity=context["packageIdentity"],
        call_plan_identity=context["callPlanIdentity"],
        request_policy=context["requestPolicy"],
        shard=context["shard"],
    ).encode("utf-8")
    checked_in = REPO_ROOT / TASK_SOURCE_PATH
    if not checked_in.is_file() or checked_in.read_bytes() != source:
        _fail("checked-in generated capture task is stale")
    expected = {
        "datasetIdentity": context["datasetIdentity"],
        "packageIdentity": {
            key: context["packageIdentity"][key]
            for key in ("id", "manifestPath", "sha256", "bytes")
        },
        "callPlanIdentity": context["callPlanIdentity"],
        "taskIdentity": {
            "name": TASK_NAME,
            "version": str(TASK_VERSION),
            "sourcePath": TASK_SOURCE_PATH,
            "definitionSha256": context["definitionSha256"],
            "implementationSha256": context["implementationSha256"],
        },
        "requestPolicy": context["requestPolicy"],
        "shard": context["shard"],
    }
    return expected, source


def _notebook_source_candidates(
    data: bytes,
    *,
    expected_source: bytes,
    entry: str,
) -> list[dict[str, Any]]:
    try:
        notebook = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except KaggleCaptureEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        _fail("archive notebook source is not bounded strict JSON")
    if not isinstance(notebook, dict):
        _fail("archive notebook source must be a JSON object")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or len(cells) > 10_000:
        _fail("archive notebook has an invalid code-cell list")
    matches: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source")
        if isinstance(source, str):
            text = source
        elif isinstance(source, list) and all(
            isinstance(part, str) for part in source
        ):
            text = "".join(source)
        else:
            continue
        try:
            source_bytes = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            continue
        # Jupytext intentionally removes a Python shebang when converting the
        # exact percent-script source into a notebook code cell. Accept only
        # that single deterministic transformation; all other bytes remain
        # bound to the checked-in generated source.
        notebook_expected_source = expected_source
        shebang = b"#!/usr/bin/env python3\n"
        if notebook_expected_source.startswith(shebang):
            notebook_expected_source = notebook_expected_source[len(shebang) :]
        if notebook_expected_source.endswith(b"\n"):
            notebook_expected_source = notebook_expected_source[:-1]
        if source_bytes in {expected_source, notebook_expected_source}:
            matches.append(
                {
                    "sourceContainerEntry": entry,
                    "archiveKind": "notebookCodeCell",
                    "notebookCellIndex": index,
                }
            )
    return matches


def _extract_capture_and_source(
    archive_bytes: bytes,
    *,
    expected_source: bytes,
) -> tuple[bytes, dict[str, Any]]:
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        _fail("capture archive exceeds the safety limit")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_ENTRIES:
                _fail("capture archive has too many entries")
            names: set[str] = set()
            total_size = 0
            payload_infos: list[zipfile.ZipInfo] = []
            source_matches: list[dict[str, Any]] = []
            for info in infos:
                _safe_archive_entry(info)
                if info.filename in names:
                    _fail("capture archive contains duplicate entry names")
                names.add(info.filename)
                total_size += info.file_size
                if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    _fail("capture archive uncompressed size exceeds the safety limit")
                if PurePosixPath(info.filename).name == CAPTURE_FILENAME:
                    payload_infos.append(info)
                suffix = PurePosixPath(info.filename).suffix.lower()
                if info.is_dir() or suffix not in {".py", ".ipynb"}:
                    continue
                if suffix == ".py":
                    if info.file_size > MAX_SOURCE_BYTES:
                        continue
                    if archive.read(info) == expected_source:
                        source_matches.append(
                            {
                                "sourceContainerEntry": info.filename,
                                "archiveKind": "directFile",
                                "notebookCellIndex": None,
                            }
                        )
                elif info.file_size <= MAX_NOTEBOOK_BYTES:
                    source_matches.extend(
                        _notebook_source_candidates(
                            archive.read(info),
                            expected_source=expected_source,
                            entry=info.filename,
                        )
                    )
            if len(payload_infos) != 1:
                _fail(
                    "capture archive must contain exactly one raw payload; "
                    f"found {len(payload_infos)}"
                )
            payload_info = payload_infos[0]
            if payload_info.file_size > MAX_CAPTURE_BYTES:
                _fail("archived capture payload exceeds the safety limit")
            payload_bytes = archive.read(payload_info)
            if len(source_matches) != 1:
                _fail(
                    "capture archive must contain exactly one matching task source; "
                    f"found {len(source_matches)}"
                )
    except KaggleCaptureEvidenceError:
        raise
    except (zipfile.BadZipFile, RuntimeError, KeyError, ValueError):
        raise KaggleCaptureEvidenceError(
            "Kaggle capture output is not a valid bounded zip archive"
        ) from None
    source = source_matches[0]
    source["payloadEntry"] = payload_info.filename
    return payload_bytes, source


def _verify_capture_contract(
    payload: dict[str, Any], expected: dict[str, Any]
) -> None:
    for field in (
        "datasetIdentity",
        "packageIdentity",
        "callPlanIdentity",
        "taskIdentity",
        "requestPolicy",
        "shard",
    ):
        if payload[field] != expected[field]:
            _fail(f"capture {field} drifted from the generated task")


def _hash_optional_message(value: Any) -> str | None:
    if value is None:
        return None
    try:
        encoded = str(value).encode("utf-8", errors="backslashreplace")
    except Exception:
        encoded = b"<unprintable-platform-message>"
    return _sha256(encoded)


def _redacted_task(record: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(record)
    error = value.pop("creationError")
    value["creationErrorStringSha256"] = _hash_optional_message(error)
    if value["owner"] is None:
        _fail("Kaggle task readback has no owner")
    return value


def _redacted_run(record: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(record)
    error = value.pop("error")
    value["errorStringSha256"] = _hash_optional_message(error)
    if value["owner"] is None:
        _fail("Kaggle run readback has no owner")
    return value


def _run_model_matches_observation(run_slug: str, observed_slug: str) -> bool:
    if run_slug == observed_slug:
        return True
    # Kaggle's task-run API currently returns the model basename while the
    # runtime actor and downloaded ATIF preserve the provider-qualified slug.
    # Accept only that one lossy representation. A conflicting qualified
    # provider remains a hard failure, and both raw values stay in evidence.
    if "/" in run_slug:
        return False
    observed_parts = observed_slug.split("/")
    if len(observed_parts) < 2 or not all(observed_parts):
        return False
    observed_basename = observed_parts[-1]
    if observed_basename == run_slug:
        return True
    # For versioned proxy aliases, the runtime actor retains `model@revision`
    # while the task-run API serializes the same identity as `model-revision`.
    # Accept only one non-empty revision suffix and one exact transformation;
    # never normalize arbitrary punctuation or conflicting providers.
    if observed_basename.count("@") != 1:
        return False
    model, revision = observed_basename.split("@", 1)
    return bool(model and revision and f"{model}-{revision}" == run_slug)


def _verify_platform_binding(
    *,
    task: dict[str, Any],
    run: dict[str, Any],
    payload: dict[str, Any],
    downloaded_at: str,
    dispatch_journal_bytes: bytes | None = None,
) -> dict[str, Any] | None:
    if (task["owner"], task["slug"], task["version"]) != (
        run["owner"],
        run["task"],
        run["version"],
    ):
        _fail("task and run identities disagree")
    observed_model = payload["modelObservation"]["slug"]
    if observed_model == "unavailable/unobserved":
        if dispatch_journal_bytes is not None:
            _fail("dispatch journal cannot bind an unavailable runtime model")
        if (
            payload["calls"]["attemptedCallCount"] != 0
            or payload["canonicalReplayEligible"]
        ):
            _fail("unavailable capture model cannot claim a dispatched call")
        catalog_record = None
    elif dispatch_journal_bytes is not None:
        catalog_record = _dispatch_journal_catalog_record(
            dispatch_journal_bytes,
            task=task,
            run=run,
            observed_model=observed_model,
        )
    else:
        catalog_record = None
        if not _run_model_matches_observation(
            run["modelVersionSlug"], observed_model
        ):
            _fail("Kaggle run model differs from the capture model observation")

    capture_start = _parse_time(payload["startedAt"], role="capture.startedAt")
    capture_end = (
        _parse_time(payload["endedAt"], role="capture.endedAt")
        if payload["endedAt"] is not None
        else None
    )
    if run["startTime"] is not None:
        run_start = _parse_time(run["startTime"], role="run.startTime")
        if capture_start < run_start:
            _fail("capture interval starts before the bound Kaggle run")
    if run["endTime"] is not None:
        run_end = _parse_time(run["endTime"], role="run.endTime")
        if capture_end is not None and capture_end > run_end:
            _fail("capture interval ends after the bound Kaggle run")
        if _parse_time(downloaded_at, role="downloadedAt") < run_end:
            _fail("capture evidence was timestamped before the run ended")
    return catalog_record


def _assembly_eligible(
    task: dict[str, Any], run: dict[str, Any], payload: dict[str, Any]
) -> bool:
    return (
        task["creationState"]
        == "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED"
        and run["state"] == "BENCHMARK_TASK_RUN_STATE_COMPLETED"
        and payload["canonicalReplayEligible"]
    )


def _expected_bundle_names(
    task: dict[str, Any], run: dict[str, Any]
) -> dict[str, str]:
    stem = f"{task['slug']}-v{task['version']}-run-{run['id']}"
    return {
        "archive": f"{stem}.zip",
        "payload": f"{stem}-capture.json",
        "source": f"{stem}-task-source.py",
        "dispatchJournal": f"{stem}-dispatch-journal.json",
    }


def verify_capture_evidence(
    evidence: dict[str, Any],
    *,
    archive_bytes: bytes,
    payload_bytes: bytes,
    source_bytes: bytes,
    dispatch_journal_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Verify one envelope and all exact bytes it claims to bind."""

    try:
        validate(evidence, load_schema(EVIDENCE_SCHEMA_PATH))
    except (SchemaValidationError, OSError, json.JSONDecodeError, RecursionError):
        raise KaggleCaptureEvidenceError(
            "capture evidence schema validation failed"
        ) from None
    if (
        evidence["evidenceSchemaVersion"]
        not in SUPPORTED_EVIDENCE_SCHEMA_VERSIONS
    ):
        _fail("unsupported capture evidence schema version")
    if (
        evidence["evidenceSchemaVersion"] == LEGACY_EVIDENCE_SCHEMA_VERSION
        and "modelCatalogBinding" in evidence
    ):
        _fail("legacy capture evidence cannot contain a model catalog binding")
    if (
        evidence["evidenceSchemaVersion"] == EVIDENCE_SCHEMA_VERSION
        and "modelCatalogBinding" not in evidence
    ):
        _fail("capture evidence schema 1.1 requires a model catalog binding")
    if evidence["artifactKind"] != ARTIFACT_KIND:
        _fail("capture evidence artifact kind mismatch")
    if evidence["targetProtocolVersion"] != TARGET_PROTOCOL_VERSION:
        _fail("capture evidence target protocol mismatch")
    if evidence["leaderboardEligible"] is not False:
        _fail("capture evidence can never be leaderboard eligible")
    if evidence["publicationEligible"] is not False:
        _fail("capture evidence can never be publication eligible")

    expected, expected_source = _expected_capture_contract()
    if source_bytes != expected_source:
        _fail("retained task source differs from the generated source")
    extracted_payload, archive_source = _extract_capture_and_source(
        archive_bytes,
        expected_source=expected_source,
    )
    if extracted_payload != payload_bytes:
        _fail("retained payload bytes differ from the archive member")
    payload = parse_capture_payload(payload_bytes)
    _verify_capture_contract(payload, expected)
    catalog_record = _verify_platform_binding(
        task=evidence["task"],
        run=evidence["run"],
        payload=payload,
        downloaded_at=evidence["downloadedAt"],
        dispatch_journal_bytes=dispatch_journal_bytes,
    )
    catalog_binding = evidence.get("modelCatalogBinding")
    if (catalog_binding is None) != (dispatch_journal_bytes is None):
        _fail("model catalog binding and dispatch journal must be supplied together")
    if catalog_binding is not None:
        expected_catalog_binding = {
            **catalog_record,
            "dispatchJournal": {
                "file": _expected_bundle_names(
                    evidence["task"], evidence["run"]
                )["dispatchJournal"],
                "bytes": len(dispatch_journal_bytes),
                "sha256": _sha256(dispatch_journal_bytes),
            },
        }
        if catalog_binding != expected_catalog_binding:
            _fail("model catalog binding disagrees with dispatch journal bytes")

    for field, data in (
        ("archive", archive_bytes),
        ("payload", payload_bytes),
        ("source", source_bytes),
    ):
        binding = evidence[field]
        if binding["bytes"] != len(data) or binding["sha256"] != _sha256(data):
            _fail(f"{field} byte binding mismatch")
    if evidence["archive"]["payloadEntry"] != archive_source["payloadEntry"]:
        _fail("archive payload entry binding mismatch")
    if (
        evidence["archive"]["sourceContainerEntry"]
        != archive_source["sourceContainerEntry"]
    ):
        _fail("archive source-container binding mismatch")
    if evidence["source"]["archiveKind"] != archive_source["archiveKind"]:
        _fail("archive source kind binding mismatch")
    if (
        evidence["source"]["notebookCellIndex"]
        != archive_source["notebookCellIndex"]
    ):
        _fail("archive notebook-cell binding mismatch")
    if evidence["source"]["sourcePath"] != payload["taskIdentity"]["sourcePath"]:
        _fail("retained source path differs from the capture task identity")
    expected_names = _expected_bundle_names(evidence["task"], evidence["run"])
    for field, expected_name in expected_names.items():
        if field == "dispatchJournal":
            continue
        if evidence[field]["file"] != expected_name:
            _fail(f"{field} file name disagrees with the bound task run")
    expected_payload_summary = {
        "artifactId": payload["id"],
        "captureComplete": payload["captureComplete"],
        "canonicalReplayEligible": payload["canonicalReplayEligible"],
    }
    if {
        key: evidence["payload"][key] for key in expected_payload_summary
    } != expected_payload_summary:
        _fail("payload summary disagrees with retained payload bytes")
    if evidence["assemblyEligible"] is not _assembly_eligible(
        evidence["task"], evidence["run"], payload
    ):
        _fail("assemblyEligible disagrees with platform and capture state")
    if evidence["id"] != _artifact_id(evidence):
        _fail("capture evidence artifact id mismatch")
    return evidence


def write_capture_evidence_bundle(
    *,
    task_info: Any,
    run_info: Any,
    requested_task: str,
    expected_version: int,
    expected_source_kernel_id: int | None,
    expected_run_id: int | None,
    expected_datasets: tuple[str, ...],
    archive_bytes: bytes,
    output_dir: Path,
    downloaded_at: str | None = None,
    dispatch_journal_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Bind and exclusively persist one exact Kaggle capture run."""

    if len(expected_datasets) != 1:
        _fail("capture evidence requires exactly one attached dataset")
    dataset_owner, dataset_slug = _parse_task_slug(expected_datasets[0])
    if dataset_owner is None or dataset_slug != EXPECTED_DATASET_SLUG:
        _fail("capture evidence dataset slug is not the frozen package slug")
    task = _redacted_task(
        _task_metadata(
            task_info,
            requested_task=requested_task,
            expected_version=expected_version,
            expected_source_kernel_id=expected_source_kernel_id,
            expected_datasets=expected_datasets,
        )
    )
    run = _redacted_run(
        _run_metadata(
            run_info,
            requested_task=requested_task,
            expected_version=expected_version,
            expected_run_id=expected_run_id,
        )
    )
    expected, expected_source = _expected_capture_contract()
    payload_bytes, archive_source = _extract_capture_and_source(
        archive_bytes,
        expected_source=expected_source,
    )
    payload = parse_capture_payload(payload_bytes)
    _verify_capture_contract(payload, expected)
    observed_downloaded_at = downloaded_at or _utc_now()
    catalog_record = _verify_platform_binding(
        task=task,
        run=run,
        payload=payload,
        downloaded_at=observed_downloaded_at,
        dispatch_journal_bytes=dispatch_journal_bytes,
    )

    bundle_names = _expected_bundle_names(task, run)
    stem = f"{task['slug']}-v{expected_version}-run-{run['id']}"
    archive_path = output_dir / bundle_names["archive"]
    payload_path = output_dir / bundle_names["payload"]
    source_path = output_dir / bundle_names["source"]
    dispatch_journal_path = output_dir / bundle_names["dispatchJournal"]
    evidence_path = output_dir / f"{stem}-evidence.json"
    evidence = {
        "evidenceSchemaVersion": (
            EVIDENCE_SCHEMA_VERSION
            if catalog_record is not None
            else LEGACY_EVIDENCE_SCHEMA_VERSION
        ),
        "artifactKind": ARTIFACT_KIND,
        "targetProtocolVersion": TARGET_PROTOCOL_VERSION,
        "leaderboardEligible": False,
        "publicationEligible": False,
        "assemblyEligible": _assembly_eligible(task, run, payload),
        "downloadedAt": observed_downloaded_at,
        "task": task,
        "run": run,
        "archive": {
            "file": archive_path.name,
            "bytes": len(archive_bytes),
            "sha256": _sha256(archive_bytes),
            "payloadEntry": archive_source["payloadEntry"],
            "sourceContainerEntry": archive_source["sourceContainerEntry"],
        },
        "payload": {
            "file": payload_path.name,
            "bytes": len(payload_bytes),
            "sha256": _sha256(payload_bytes),
            "artifactId": payload["id"],
            "captureComplete": payload["captureComplete"],
            "canonicalReplayEligible": payload["canonicalReplayEligible"],
        },
        "source": {
            "file": source_path.name,
            "bytes": len(expected_source),
            "sha256": _sha256(expected_source),
            "sourcePath": payload["taskIdentity"]["sourcePath"],
            "archiveKind": archive_source["archiveKind"],
            "notebookCellIndex": archive_source["notebookCellIndex"],
        },
        "id": "",
    }
    if catalog_record is not None and dispatch_journal_bytes is not None:
        evidence["modelCatalogBinding"] = {
            **catalog_record,
            "dispatchJournal": {
                "file": dispatch_journal_path.name,
                "bytes": len(dispatch_journal_bytes),
                "sha256": _sha256(dispatch_journal_bytes),
            },
        }
    evidence["id"] = _artifact_id(evidence)
    verify_capture_evidence(
        evidence,
        archive_bytes=archive_bytes,
        payload_bytes=payload_bytes,
        source_bytes=expected_source,
        dispatch_journal_bytes=dispatch_journal_bytes,
    )
    evidence_bytes = (
        json.dumps(
            evidence,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths = [archive_path, payload_path, source_path, evidence_path]
    if dispatch_journal_bytes is not None:
        bundle_paths.append(dispatch_journal_path)
    for path in bundle_paths:
        if path.exists():
            _fail(f"refusing to overwrite existing capture evidence: {path}")
    written: list[Path] = []
    try:
        for path, data in (
            (archive_path, archive_bytes),
            (payload_path, payload_bytes),
            (source_path, expected_source),
            *(
                ((dispatch_journal_path, dispatch_journal_bytes),)
                if dispatch_journal_bytes is not None
                else ()
            ),
            (evidence_path, evidence_bytes),
        ):
            _write_exclusive(path, data)
            written.append(path)
    except Exception:
        for path in reversed(written):
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download and bind one exact Kaggle raw capture "
            "without scheduling a model run."
        )
    )
    parser.add_argument("task", help="OWNER/TASK slug")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--source-kernel-id", type=int)
    parser.add_argument("--run-id", type=int)
    parser.add_argument(
        "--dispatch-journal",
        type=Path,
        help="exact reconciled kaggle_run_once journal for a scheduled run",
    )
    parser.add_argument("--expect-dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.version <= 0:
        parser.error("--version must be positive")
    if args.source_kernel_id is not None and args.source_kernel_id <= 0:
        parser.error("--source-kernel-id must be positive")
    if args.run_id is not None and args.run_id <= 0:
        parser.error("--run-id must be positive")
    expected_datasets = (args.expect_dataset,)
    try:
        dispatch_journal_bytes = (
            _read_dispatch_journal(args.dispatch_journal)
            if args.dispatch_journal is not None
            else None
        )
        task_info, run_info, archive_bytes = _download_creation_archive(
            args.task,
            args.version,
            args.run_id,
            args.source_kernel_id,
            expected_datasets,
        )
        evidence = write_capture_evidence_bundle(
            task_info=task_info,
            run_info=run_info,
            requested_task=args.task,
            expected_version=args.version,
            expected_source_kernel_id=args.source_kernel_id,
            expected_run_id=args.run_id,
            expected_datasets=expected_datasets,
            archive_bytes=archive_bytes,
            output_dir=args.output,
            dispatch_journal_bytes=dispatch_journal_bytes,
        )
    except (OSError, ValueError) as exc:
        print(f"invalid Kaggle capture evidence: {exc}", file=sys.stderr)
        return 1
    print(
        f"retained task {evidence['task']['slug']} v{evidence['task']['version']} "
        f"run {evidence['run']['id']}: assemblyEligible="
        f"{str(evidence['assemblyEligible']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
