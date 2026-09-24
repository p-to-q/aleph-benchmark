from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .schema_validation import SchemaValidationError, load_schema, validate


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCHEMA_PATH = (
    REPO_ROOT / "schemas/v0.2/aleph-bench-kaggle-capture-payload.schema.json"
)
CAPTURE_SCHEMA_VERSION = "1.1.0"
ARTIFACT_KIND = "aleph_bench_kaggle_raw_capture"
TARGET_PROTOCOL_VERSION = "0.2.0"
ARTIFACT_ID_PREFIX = "aleph-bench-kaggle-capture-v1-artifact-"
FROZEN_DATASET_HASH_ALGORITHM = "sha256-length-framed-filename-and-content-v1"

MAX_CAPTURE_BYTES = 16 * 1024 * 1024
MAX_SCORING_TEXT_CODEPOINTS = 16_384
MAX_RAW_OUTPUT_CODEPOINTS = 65_536
MAX_AGGREGATE_RAW_OUTPUT_CODEPOINTS = 4_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CLEAN_FINISH_REASONS = frozenset({"stop", "end_turn", "eos", "completed"})
# Recognized limit reasons are retained as evidence but are never replay-safe.
_TOKEN_LIMIT_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "token_limit", "MAX_TOKENS"}
)
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_ROW_OUTCOMES = frozenset(
    {"returnedString", "missingOutput", "nonStringOutput", "oversizedOutput", "failed"}
)


class KaggleCaptureError(ValueError):
    """Raised when a raw-capture payload is unsafe or internally inconsistent."""


def _fail(message: str) -> None:
    raise KaggleCaptureError(message)


def _has_any_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _has_adjacent_surrogate_pair(value: str) -> bool:
    return any(
        0xD800 <= ord(value[index]) <= 0xDBFF
        and 0xDC00 <= ord(value[index + 1]) <= 0xDFFF
        for index in range(len(value) - 1)
    )


def _has_lone_surrogate(value: str) -> bool:
    """Return true for a surrogate code unit that is not part of an adjacent pair."""

    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(value) and 0xDC00 <= ord(value[index + 1]) <= 0xDFFF:
                index += 2
                continue
            return True
        if 0xDC00 <= codepoint <= 0xDFFF:
            return True
        index += 1
    return False


def _validate_json_tree(value: Any) -> None:
    """Reject cycles and strings whose JSON identity is unstable.

    Only a row's direct ``rawOutput`` value may contain a lone surrogate code
    unit. The explicit enter/exit stack tracks ancestors rather than every
    object ever seen, so a harmless shared alias is serialized twice while a
    true dict/list cycle fails closed.
    """

    stack: list[tuple[Any, bool, bool]] = [(value, False, False)]
    ancestors: set[int] = set()
    while stack:
        current, exiting, allow_lone_surrogate = stack.pop()
        if isinstance(current, float) and not math.isfinite(current):
            _fail("capture contains a non-finite number")
        if isinstance(current, str):
            # CPython may collapse an escaped adjacent surrogate pair into one
            # scalar during parsing. Reject the pair before hashing or writing.
            if _has_adjacent_surrogate_pair(current):
                _fail("capture contains an adjacent surrogate pair")
            if _has_any_surrogate(current) and not allow_lone_surrogate:
                _fail("capture contains a non-Unicode-scalar string")
        elif isinstance(current, (dict, list)):
            identity = id(current)
            if exiting:
                ancestors.remove(identity)
                continue
            if identity in ancestors:
                _fail("capture contains a cyclic object graph")
            ancestors.add(identity)
            stack.append((current, True, False))

            if isinstance(current, list):
                for child in reversed(current):
                    stack.append((child, False, False))
                continue
            for key, child in current.items():
                if not isinstance(key, str):
                    _fail("capture contains a non-string object key")
                if _has_any_surrogate(key):
                    _fail("capture contains a non-Unicode-scalar string")
                stack.append(
                    (
                        child,
                        False,
                        key == "rawOutput" and isinstance(child, str),
                    )
                )


def _canonical_json_bytes(value: Any) -> bytes:
    """Return the artifact identity encoding: sorted, compact, escaped ASCII JSON."""

    _validate_json_tree(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise KaggleCaptureError(
            "capture contains data that cannot be canonically encoded"
        ) from None


def _encode_capture_bytes(value: Any) -> bytes:
    """Encode the persisted representation while never buffering over 16 MiB."""

    output = bytearray()
    encoder = json.JSONEncoder(
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    try:
        for chunk in encoder.iterencode(value):
            encoded_chunk = chunk.encode("ascii")
            # Reserve one final byte for the canonical trailing newline.
            if len(output) + len(encoded_chunk) + 1 > MAX_CAPTURE_BYTES:
                _fail(
                    f"serialized capture exceeds the {MAX_CAPTURE_BYTES}-byte payload limit"
                )
            output.extend(encoded_chunk)
    except KaggleCaptureError:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise KaggleCaptureError("capture cannot be serialized") from None
    output.extend(b"\n")
    return bytes(output)


def canonical_string_sha256(value: str) -> str:
    """Hash an output string using exact escaped JSON-string bytes.

    Lone surrogate code units remain diagnostic evidence. An adjacent high/low
    pair is rejected because JSON parsers may collapse it on readback.
    """

    if not isinstance(value, str):
        _fail("canonical string digest requires a string")
    if _has_adjacent_surrogate_pair(value):
        _fail("canonical string contains an adjacent surrogate pair")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise KaggleCaptureError("string cannot be canonically encoded") from None
    return hashlib.sha256(encoded).hexdigest()


def prompt_utf8_sha256(value: str) -> str:
    """Hash prompt identity as strict UTF-8, rejecting every surrogate code unit."""

    if not isinstance(value, str):
        _fail("prompt digest requires a string")
    if _has_any_surrogate(value):
        _fail("prompt text contains a surrogate code unit")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise KaggleCaptureError("prompt text is not strict UTF-8") from None
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def expected_row_id(item_id: str, prompt_id: str, rerun_index: int) -> str:
    return f"{item_id}:{prompt_id}:rerun-{rerun_index}"


def coverage_sha256(shard_id: str, planned_calls: list[dict[str, Any]]) -> str:
    """Bind the schema, protocol, shard, and exact ordered call coverage."""

    return canonical_json_sha256(
        {
            "captureSchemaVersion": CAPTURE_SCHEMA_VERSION,
            "targetProtocolVersion": TARGET_PROTOCOL_VERSION,
            "shardId": shard_id,
            "plannedCalls": planned_calls,
        }
    )


def artifact_id_for(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "id"}
    return ARTIFACT_ID_PREFIX + canonical_json_sha256(body)


def _reject_json_constant(_value: str) -> None:
    raise KaggleCaptureError("non-finite JSON constant is forbidden")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise KaggleCaptureError("non-finite JSON number is forbidden")
    return parsed


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise KaggleCaptureError("duplicate JSON object key is forbidden")
        value[key] = child
    return value


def _parse_timestamp(value: str, *, role: str) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        _fail(f"{role} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise KaggleCaptureError(
            f"{role} must be a valid RFC 3339 UTC timestamp"
        ) from None
    if parsed.utcoffset() is None:
        _fail(f"{role} must include a UTC offset")
    return parsed


def _verify_logical_path(value: str, *, role: str) -> None:
    if (
        not value
        or "\\" in value
        or value.startswith("/")
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail(f"{role} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{role} must be a canonical relative POSIX path")


def _verify_call_identity(call: dict[str, Any], *, role: str) -> None:
    expected = expected_row_id(call["itemId"], call["promptId"], call["rerunIndex"])
    if call["rowId"] != expected:
        _fail(f"{role}.rowId does not match its coordinate")
    prompt = call["promptText"]
    if _has_any_surrogate(prompt):
        _fail(f"{role}.promptText contains a surrogate code unit")
    if call["promptCodePointCount"] != len(prompt):
        _fail(f"{role}.promptCodePointCount disagrees with promptText")
    if call["promptUtf8Sha256"] != prompt_utf8_sha256(prompt):
        _fail(f"{role}.promptUtf8Sha256 disagrees with promptText")


def _call_projection(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "rowId",
        "itemId",
        "promptId",
        "rerunIndex",
        "conversationName",
        "promptText",
        "promptUtf8Sha256",
        "promptCodePointCount",
    )
    return {key: value[key] for key in keys}


def _verify_planned_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    shard = payload["shard"]
    plan = payload["callPlanIdentity"]
    planned_calls = shard["plannedCalls"]

    if shard["fullShardPlanSha256"] != plan["fullShardPlanSha256"]:
        _fail("shard full-plan digest disagrees with callPlanIdentity")
    if len(planned_calls) > plan["totalPlannedCallCount"]:
        _fail("shard coverage exceeds the declared full call plan")
    if shard["coverageSha256"] != coverage_sha256(shard["id"], planned_calls):
        _fail("shard coverage digest mismatch")

    row_ids: set[str] = set()
    coordinates: set[tuple[str, str, int]] = set()
    conversations: set[str] = set()
    for index, call in enumerate(planned_calls):
        role = f"shard.plannedCalls[{index}]"
        _verify_call_identity(call, role=role)
        row_id = call["rowId"]
        coordinate = (call["itemId"], call["promptId"], call["rerunIndex"])
        conversation = call["conversationName"]
        if row_id in row_ids:
            _fail("planned calls contain a duplicate row id")
        if coordinate in coordinates:
            _fail("planned calls contain a duplicate prompt/rerun coordinate")
        if conversation in conversations:
            _fail("planned calls contain a duplicate conversation name")
        row_ids.add(row_id)
        coordinates.add(coordinate)
        conversations.add(conversation)
    return planned_calls


def _verify_usage(usage: dict[str, Any], *, role: str) -> None:
    for field in ("inputCostNanodollars", "outputCostNanodollars"):
        value = usage[field]
        if value is not None and (
            not value.isascii() or not value.isdigit() or len(value) > 40
        ):
            _fail(f"{role}.{field} must be a bounded unsigned integer string")


def _verify_failure_shape(row: dict[str, Any], *, role: str) -> None:
    outcome = row["outcome"]
    dispatch_started = row["dispatchStarted"]
    terminal_failure = row["terminalFailure"]
    lifecycle_failure = row["lifecycleFailure"]

    if outcome == "failed":
        if terminal_failure is None:
            _fail(f"{role} failed outcome requires terminalFailure")
        failure_kind = terminal_failure["failureKind"]
        if failure_kind == "chatOpenFailed" and dispatch_started:
            _fail(f"{role} chatOpenFailed cannot claim dispatchStarted")
        if failure_kind == "modelCallFailed" and not dispatch_started:
            _fail(f"{role} modelCallFailed requires dispatchStarted")
        if failure_kind == "chatOpenFailed" and lifecycle_failure is not None:
            _fail(f"{role} chatOpenFailed cannot also claim chatCloseFailed")
    else:
        if not dispatch_started:
            _fail(f"{role} model output requires dispatchStarted")
        if terminal_failure is not None:
            _fail(f"{role} non-failed outcome cannot include terminalFailure")


def _verify_row_output(row: dict[str, Any], *, role: str) -> None:
    outcome = row["outcome"]
    if outcome not in _ROW_OUTCOMES:
        _fail(f"{role}.outcome is unsupported")

    has_raw = "rawOutput" in row
    raw_digest = row["rawOutputStringSha256"]
    raw_count = row["rawOutputCodePointCount"]
    observed_type = row["observedOutputType"]

    _verify_failure_shape(row, role=role)
    if outcome == "returnedString":
        if not has_raw or observed_type != "string":
            _fail(f"{role} returnedString shape is inconsistent")
        raw = row["rawOutput"]
        if _has_adjacent_surrogate_pair(raw):
            _fail(f"{role}.rawOutput contains an adjacent surrogate pair")
        if raw_count != len(raw):
            _fail(f"{role}.rawOutputCodePointCount disagrees with rawOutput")
        if raw_digest != canonical_string_sha256(raw):
            _fail(f"{role}.rawOutputStringSha256 disagrees with rawOutput")
        if raw_count > MAX_RAW_OUTPUT_CODEPOINTS:
            _fail(f"{role} returnedString exceeds the retained-output limit")
    elif outcome == "oversizedOutput":
        if has_raw or observed_type != "string":
            _fail(f"{role} oversizedOutput shape is inconsistent")
        if raw_count is None or raw_count <= MAX_RAW_OUTPUT_CODEPOINTS:
            _fail(f"{role} oversizedOutput must exceed the retained-output limit")
        if not isinstance(raw_digest, str) or _SHA256_RE.fullmatch(raw_digest) is None:
            _fail(f"{role} oversizedOutput requires a canonical string digest")
    elif outcome == "missingOutput":
        if (
            has_raw
            or observed_type != "missing"
            or raw_digest is not None
            or raw_count is not None
        ):
            _fail(f"{role} missingOutput shape is inconsistent")
    elif outcome == "nonStringOutput":
        if (
            has_raw
            or observed_type not in {"null", "boolean", "number", "array", "object"}
            or raw_digest is not None
            or raw_count is not None
        ):
            _fail(f"{role} nonStringOutput shape is inconsistent")
    elif (
        has_raw
        or observed_type is not None
        or raw_digest is not None
        or raw_count is not None
    ):
        _fail(f"{role} failed shape is inconsistent")

    _verify_usage(row["usage"], role=f"{role}.usage")


def _verify_rows(
    payload: dict[str, Any], planned_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = payload["rows"]
    if len(rows) > len(planned_calls):
        _fail("capture has more rows than planned calls")

    previous_captured_at: datetime | None = None
    for index, row in enumerate(rows):
        role = f"rows[{index}]"
        _verify_call_identity(row, role=role)
        if _call_projection(row) != planned_calls[index]:
            _fail(f"{role} does not match the ordered planned-call prefix")
        _verify_row_output(row, role=role)
        captured_at = _parse_timestamp(row["capturedAt"], role=f"{role}.capturedAt")
        if previous_captured_at is not None and captured_at < previous_captured_at:
            _fail("row capture timestamps are not monotonic")
        previous_captured_at = captured_at
    return rows


def _verify_active_call(
    payload: dict[str, Any],
    planned_calls: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    active = payload["calls"]["activeCall"]
    if active is None:
        return None

    call = active["call"]
    state = active["state"]
    _verify_call_identity(call, role="calls.activeCall.call")
    if state in {"prepared", "dispatching"}:
        if len(rows) >= len(planned_calls):
            _fail("unpersisted active call cannot follow complete coverage")
        if call != planned_calls[len(rows)]:
            _fail("prepared or dispatching active call must be the next planned call")
    elif state == "terminalPersisted":
        if not rows:
            _fail("terminalPersisted active call requires a persisted row")
        if call != planned_calls[len(rows) - 1] or _call_projection(rows[-1]) != call:
            _fail("terminalPersisted active call must identify the last persisted row")
    else:
        _fail("active call state is unsupported")
    _parse_timestamp(active["updatedAt"], role="calls.activeCall.updatedAt")
    return active


def _expected_call_counts(
    rows: list[dict[str, Any]],
    active: dict[str, Any] | None,
    planned_count: int,
) -> dict[str, int]:
    outcomes = {outcome: 0 for outcome in _ROW_OUTCOMES}
    for row in rows:
        outcomes[row["outcome"]] += 1

    unpersisted_dispatch = (
        1 if active is not None and active["state"] == "dispatching" else 0
    )
    dispatched_rows = sum(row["dispatchStarted"] for row in rows)
    return {
        "plannedCallCount": planned_count,
        "attemptedCallCount": dispatched_rows + unpersisted_dispatch,
        "terminalCallCount": dispatched_rows,
        "returnedStringCount": outcomes["returnedString"],
        "missingOutputCount": outcomes["missingOutput"],
        "nonStringOutputCount": outcomes["nonStringOutput"],
        "oversizedOutputCount": outcomes["oversizedOutput"],
        "failedCallCount": outcomes["failed"],
    }


def _expected_diagnostics(
    payload: dict[str, Any],
    planned_calls: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    active: dict[str, Any] | None,
) -> dict[str, Any]:
    max_output_tokens = payload["requestPolicy"]["maxOutputTokens"]
    near_cap_margin = payload["requestPolicy"]["nearCapMarginTokens"]
    if not 0 <= near_cap_margin < max_output_tokens:
        _fail("nearCapMarginTokens must be smaller than maxOutputTokens")
    near_cap_threshold = max_output_tokens - near_cap_margin

    missing_row_ids = [call["rowId"] for call in planned_calls[len(rows) :]]
    lone_surrogate_ids = [
        row["rowId"]
        for row in rows
        if row["outcome"] == "returnedString"
        and _has_lone_surrogate(row["rawOutput"])
    ]
    missing_output_ids = [
        row["rowId"] for row in rows if row["outcome"] == "missingOutput"
    ]
    non_string_ids = [
        row["rowId"] for row in rows if row["outcome"] == "nonStringOutput"
    ]
    oversized_ids = [
        row["rowId"] for row in rows if row["outcome"] == "oversizedOutput"
    ]
    terminal_failure_ids = [
        row["rowId"] for row in rows if row["terminalFailure"] is not None
    ]
    lifecycle_failure_ids = [
        row["rowId"] for row in rows if row["lifecycleFailure"] is not None
    ]
    scoring_too_long_ids = [
        row["rowId"]
        for row in rows
        if row["outcome"] == "returnedString"
        and row["rawOutputCodePointCount"] > MAX_SCORING_TEXT_CODEPOINTS
    ]
    missing_usage_ids = [
        row["rowId"]
        for row in rows
        if row["usage"]["inputTokens"] is None
        or row["usage"]["outputTokens"] is None
    ]
    missing_finish_reason_ids = [
        row["rowId"]
        for row in rows
        if row["usage"]["finishReason"] is None
    ]
    zero_usage_ids = [
        row["rowId"]
        for row in rows
        if row["usage"]["inputTokens"] == 0
        or row["usage"]["outputTokens"] == 0
    ]
    near_cap_ids = [
        row["rowId"]
        for row in rows
        if row["usage"]["outputTokens"] is not None
        and row["usage"]["outputTokens"] >= near_cap_threshold
    ]
    token_limit_ids = [
        row["rowId"]
        for row in rows
        if row["usage"]["finishReason"] in _TOKEN_LIMIT_FINISH_REASONS
    ]
    unsupported_finish_ids = [
        row["rowId"]
        for row in rows
        if row["usage"]["finishReason"] is not None
        and row["usage"]["finishReason"] not in _CLEAN_FINISH_REASONS
        and row["usage"]["finishReason"] not in _TOKEN_LIMIT_FINISH_REASONS
    ]
    active_is_unpersisted = active is not None and active["state"] in {
        "prepared",
        "dispatching",
    }

    blockers: list[str] = []
    for present, reason in (
        (bool(missing_row_ids), "incompleteCoverage"),
        (active_is_unpersisted, "activeCall"),
        (bool(lone_surrogate_ids), "unpairedSurrogate"),
        (bool(missing_output_ids), "missingOutput"),
        (bool(non_string_ids), "nonStringOutput"),
        (bool(oversized_ids), "oversizedOutput"),
        (bool(terminal_failure_ids), "terminalFailure"),
        (bool(lifecycle_failure_ids), "lifecycleFailure"),
        (bool(scoring_too_long_ids), "scoringTextTooLong"),
        (bool(missing_usage_ids), "missingUsage"),
        (bool(near_cap_ids), "nearCapOutput"),
        (bool(token_limit_ids), "tokenLimitTermination"),
        (bool(unsupported_finish_ids), "unsupportedFinishReason"),
    ):
        if present:
            blockers.append(reason)

    return {
        "coverageComplete": not missing_row_ids,
        "missingPlannedRowIds": missing_row_ids,
        "unpairedSurrogateRowIds": lone_surrogate_ids,
        "missingOutputRowIds": missing_output_ids,
        "nonStringOutputRowIds": non_string_ids,
        "oversizedOutputRowIds": oversized_ids,
        "terminalFailureRowIds": terminal_failure_ids,
        "lifecycleFailureRowIds": lifecycle_failure_ids,
        "scoringTextTooLongRowIds": scoring_too_long_ids,
        "missingUsageRowIds": missing_usage_ids,
        "missingFinishReasonRowIds": missing_finish_reason_ids,
        "zeroUsageRowIds": zero_usage_ids,
        "nearCapOutputRowIds": near_cap_ids,
        "tokenLimitTerminationRowIds": token_limit_ids,
        "unsupportedFinishReasonRowIds": unsupported_finish_ids,
        "activeCallRowId": active["call"]["rowId"] if active is not None else None,
        "replayBlockedReasons": blockers,
    }


def _verify_identity_and_policy(payload: dict[str, Any]) -> None:
    if payload["captureSchemaVersion"] != CAPTURE_SCHEMA_VERSION:
        _fail("unsupported capture schema version")
    if payload["artifactKind"] != ARTIFACT_KIND:
        _fail("capture artifact kind mismatch")
    if payload["targetProtocolVersion"] != TARGET_PROTOCOL_VERSION:
        _fail("capture target protocol mismatch")
    if payload["leaderboardEligible"] is not False:
        _fail("raw capture can never be leaderboard eligible")
    if payload["publicationEligible"] is not False:
        _fail("raw capture can never be publication eligible")
    if payload["datasetIdentity"]["hashAlgorithm"] != FROZEN_DATASET_HASH_ALGORITHM:
        _fail("capture dataset hash algorithm disagrees with frozen v0.2")

    _verify_logical_path(
        payload["packageIdentity"]["manifestPath"],
        role="packageIdentity.manifestPath",
    )
    _verify_logical_path(
        payload["taskIdentity"]["sourcePath"], role="taskIdentity.sourcePath"
    )

    model = payload["modelObservation"]
    if model["providerRevision"] is not None:
        _fail("Kaggle capture cannot claim an unobserved provider revision")
    if model["mappingStatus"] == "unmapped":
        if model["canonicalModelId"] is not None:
            _fail("unmapped model observation cannot claim a canonical model id")
    elif model["canonicalModelId"] is None:
        _fail("operator-declared model mapping requires a canonical model id")

    request = payload["requestPolicy"]
    if not 0 <= request["nearCapMarginTokens"] < request["maxOutputTokens"]:
        _fail("nearCapMarginTokens must be smaller than maxOutputTokens")


def _verify_times(
    payload: dict[str, Any], rows: list[dict[str, Any]], active: dict[str, Any] | None
) -> None:
    started = _parse_timestamp(payload["startedAt"], role="startedAt")
    ended_value = payload["endedAt"]
    ended = (
        _parse_timestamp(ended_value, role="endedAt")
        if ended_value is not None
        else None
    )
    if ended is not None and ended < started:
        _fail("endedAt precedes startedAt")
    if payload["captureComplete"] and ended is None:
        _fail("captureComplete requires endedAt")

    last_captured: datetime | None = None
    for index, row in enumerate(rows):
        captured = _parse_timestamp(row["capturedAt"], role=f"rows[{index}].capturedAt")
        if captured < started or (ended is not None and captured > ended):
            _fail(f"rows[{index}].capturedAt is outside the capture interval")
        last_captured = captured
    if active is not None:
        updated = _parse_timestamp(active["updatedAt"], role="calls.activeCall.updatedAt")
        if updated < started or (ended is not None and updated > ended):
            _fail("calls.activeCall.updatedAt is outside the capture interval")
        if last_captured is not None and updated < last_captured:
            _fail("calls.activeCall.updatedAt precedes the last persisted row")


def _verify_and_encode_capture_payload(payload: dict[str, Any]) -> bytes:
    """Validate every contract gate and return the bounded persisted bytes.

    This function is pure: it performs no model, network, Kaggle, or scorer work.
    Raised messages are value-free so backend text, prompts, keys, and exception
    strings cannot leak through verifier logs.
    """

    if not isinstance(payload, dict):
        _fail("capture payload must be a JSON object")
    try:
        _validate_json_tree(payload)
        validate(payload, load_schema(CAPTURE_SCHEMA_PATH))
    except KaggleCaptureError:
        raise
    except (SchemaValidationError, KeyError, OSError, json.JSONDecodeError, RecursionError):
        raise KaggleCaptureError("capture schema validation failed") from None

    # Apply the persisted-byte boundary before digest and cross-field work. The
    # compact identity encoding is never larger than this pretty representation,
    # so subsequent hashing also stays within an already-bounded input graph.
    encoded = _encode_capture_bytes(payload)

    _verify_identity_and_policy(payload)
    planned_calls = _verify_planned_calls(payload)
    rows = _verify_rows(payload, planned_calls)
    active = _verify_active_call(payload, planned_calls, rows)

    expected_counts = _expected_call_counts(rows, active, len(planned_calls))
    observed_counts = {key: payload["calls"][key] for key in expected_counts}
    if observed_counts != expected_counts:
        _fail("call counts disagree with rows and active state")

    expected_diagnostics = _expected_diagnostics(payload, planned_calls, rows, active)
    if payload["diagnostics"] != expected_diagnostics:
        _fail("capture diagnostics disagree with persisted observations")

    expected_capture_complete = expected_diagnostics["coverageComplete"]
    if payload["captureComplete"] is not expected_capture_complete:
        _fail("captureComplete disagrees with persisted coverage")
    expected_replay_eligible = (
        expected_capture_complete and not expected_diagnostics["replayBlockedReasons"]
    )
    if payload["canonicalReplayEligible"] is not expected_replay_eligible:
        _fail("canonicalReplayEligible disagrees with replay blockers")

    _verify_times(payload, rows, active)
    aggregate_raw_count = sum(
        row["rawOutputCodePointCount"]
        for row in rows
        if row["outcome"] == "returnedString"
    )
    if aggregate_raw_count > MAX_AGGREGATE_RAW_OUTPUT_CODEPOINTS:
        _fail("aggregate retained raw output exceeds the capture limit")

    if payload["id"] != artifact_id_for(payload):
        _fail("capture artifact id mismatch")
    return encoded


def verify_capture_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a payload, including its exact persisted-size boundary."""

    _verify_and_encode_capture_payload(payload)
    return payload


def serialize_capture_payload(payload: dict[str, Any]) -> bytes:
    """Verify and serialize a payload without changing any decoded string."""

    return _verify_and_encode_capture_payload(payload)


def parse_capture_payload(data: bytes) -> dict[str, Any]:
    """Parse bounded UTF-8 JSON, rejecting duplicates and non-finite values."""

    if not isinstance(data, bytes):
        _fail("capture input must be bytes")
    if len(data) > MAX_CAPTURE_BYTES:
        _fail(f"capture exceeds the {MAX_CAPTURE_BYTES}-byte payload limit")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise KaggleCaptureError("capture is not valid UTF-8") from None
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except KaggleCaptureError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError, OverflowError):
        raise KaggleCaptureError("capture is not valid JSON") from None
    if not isinstance(payload, dict):
        _fail("capture payload must be a JSON object")
    return verify_capture_payload(payload)


def load_capture_payload(path: str | Path) -> dict[str, Any]:
    """Load one bounded regular file without following a symlink."""

    capture_path = Path(path)
    try:
        before = capture_path.lstat()
    except OSError:
        raise KaggleCaptureError("cannot inspect capture file") from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail("capture path must name a regular file, not a symlink or directory")
    if before.st_size > MAX_CAPTURE_BYTES:
        _fail(f"capture exceeds the {MAX_CAPTURE_BYTES}-byte payload limit")

    required_open_flags = ("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required_open_flags):
        _fail("capture loader requires fail-closed filesystem flags")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    descriptor: int | None = None
    try:
        descriptor = os.open(capture_path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                _fail("capture path must name a regular file")
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                _fail("capture file changed while it was opened")
            data = stream.read(MAX_CAPTURE_BYTES + 1)
            after = os.fstat(stream.fileno())
    except KaggleCaptureError:
        raise
    except OSError:
        raise KaggleCaptureError("cannot read capture file") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if len(data) > MAX_CAPTURE_BYTES:
        _fail(f"capture exceeds the {MAX_CAPTURE_BYTES}-byte payload limit")
    if (
        after.st_size != len(data)
        or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
        or after.st_mtime_ns != opened.st_mtime_ns
    ):
        _fail("capture file changed while it was read")
    return parse_capture_payload(data)
