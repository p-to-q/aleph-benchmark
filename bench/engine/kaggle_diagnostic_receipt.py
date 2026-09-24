from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema_validation import load_schema, validate


REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_SCHEMA_PATH = (
    REPO_ROOT
    / "schemas/v0.2/aleph-bench-kaggle-diagnostic-receipt.schema.json"
)
ARTIFACT_ID_PREFIX = "aleph-bench-kaggle-diagnostic-v0.2-artifact-"
MAX_OUTPUT_CHARACTERS = 65_536
MAX_RECEIPT_BYTES = 1_048_576


class KaggleDiagnosticReceiptError(ValueError):
    """Raised when a deployment-canary receipt is not semantically coherent."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise KaggleDiagnosticReceiptError(
            f"receipt contains invalid canonical JSON data: {exc}"
        ) from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fail(message: str) -> None:
    raise KaggleDiagnosticReceiptError(message)


def _reject_json_constant(value: str) -> None:
    raise KaggleDiagnosticReceiptError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise KaggleDiagnosticReceiptError(
                f"duplicate JSON object key is forbidden: {key!r}"
            )
        value[key] = child
    return value


def _parse_timestamp(value: str, *, role: str) -> datetime:
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise KaggleDiagnosticReceiptError(
            f"{role} is not a valid RFC 3339 UTC timestamp"
        ) from exc


def _expected_context() -> dict[str, Any]:
    # The generator is the durable source for the task definition, package identity,
    # policy, and exact prompt text hashes. Import lazily so schema validation always
    # runs before repository/runtime-specific semantic work.
    from bench.tasks.kaggle.generate_v0_2_diagnostic import (
        TASK_DESCRIPTION,
        TASK_NAME,
        TASK_VERSION,
        _generation_context,
    )

    context = _generation_context()
    task_identity = {
        "name": TASK_NAME,
        "version": TASK_VERSION,
        "description": TASK_DESCRIPTION,
        "definitionSha256": context["definitionSha256"],
    }
    if "implementationSha256" in context:
        task_identity["implementationSha256"] = context["implementationSha256"]
    return {**context, "taskIdentity": task_identity}


def _verify_artifact_id(receipt: dict[str, Any]) -> None:
    payload = {key: value for key, value in receipt.items() if key != "id"}
    expected = ARTIFACT_ID_PREFIX + _sha256(_canonical_json_bytes(payload))
    if receipt["id"] != expected:
        _fail(
            "diagnostic receipt artifact id mismatch: "
            f"expected {expected}, found {receipt['id']}"
        )


def _verify_definition(
    receipt: dict[str, Any], context: dict[str, Any]
) -> None:
    if receipt["taskIdentity"] != context["taskIdentity"]:
        _fail("diagnostic task identity or definition digest drifted")
    if receipt["dependency"]["expected"] != context["packageIdentity"]:
        _fail("diagnostic package identity drifted")
    if receipt["policy"] != context["policy"]:
        _fail("diagnostic call policy drifted")


def _verify_runtime(receipt: dict[str, Any]) -> None:
    runtime = receipt["runtime"]
    required = runtime["required"]
    observed = runtime["observed"]
    expected_conformance = (
        observed["pythonVersion"] == required["pythonVersion"]
        and observed["unicodeDatabaseVersion"]
        == required["unicodeDatabaseVersion"]
    )
    if runtime["conformant"] is not expected_conformance:
        _fail("runtime.conformant disagrees with the observed runtime")
    full_version_parts = observed["pythonFullVersion"].split(".")
    if len(full_version_parts) < 2:
        _fail("runtime observed pythonFullVersion has no major/minor version")
    full_major_minor = ".".join(full_version_parts[:2])
    if full_major_minor != observed["pythonVersion"]:
        _fail("runtime pythonVersion disagrees with pythonFullVersion")


def _verify_dependency(receipt: dict[str, Any]) -> None:
    dependency = receipt["dependency"]
    expected = dependency["expected"]
    observed = dependency["observed"]
    digest_fields = (
        "manifestSha256",
        "itemsSha256",
        "scoringCoreSha256",
    )
    if observed["status"] in {"unchecked", "blocked"}:
        if any(observed[field] is not None for field in digest_fields):
            _fail(
                "unchecked or blocked dependency cannot claim verified digests"
            )
        return
    expected_digests = {
        "manifestSha256": expected["manifest"]["sha256"],
        "itemsSha256": expected["items"]["sha256"],
        "scoringCoreSha256": expected["scoringCore"]["sha256"],
    }
    if {field: observed[field] for field in digest_fields} != expected_digests:
        _fail("verified dependency digests disagree with the pinned package")


def _verify_model_identity(receipt: dict[str, Any]) -> None:
    model = receipt["modelIdentity"]
    policy = receipt["policy"]
    expected_sdk = {
        "kaggle_benchmarks.actors.llms.OpenAI": "openai",
        "kaggle_benchmarks.actors.llms.GoogleGenAI": "google-genai",
    }
    if model["status"] == "observed":
        if model["slug"] is None or model["pythonType"] is None:
            _fail("observed model identity requires a slug and Python type")
    elif model["slug"] is not None:
        _fail("unavailable model identity cannot claim a model slug")

    python_type = model["pythonType"]
    expected_parameter = (
        policy["outputTokenParameterByTransport"].get(python_type)
        if python_type is not None
        else None
    )
    if model["outputTokenParameter"] != expected_parameter:
        _fail("model output-token parameter disagrees with its transport type")
    expected_sdk_name = expected_sdk.get(python_type)
    if (
        model["transportSdkName"] is not None
        and model["transportSdkName"] != expected_sdk_name
    ):
        _fail("model transport SDK name disagrees with its Python type")
    if model["transportMaxAttempts"] is not None and expected_sdk_name is None:
        _fail("unsupported model transport cannot claim an HTTP attempt bound")
    if receipt["phase"] in {"model_calls", "complete"} and (
        model["transportSdkName"] != expected_sdk_name
        or model["transportMaxAttempts"] != 1
    ):
        _fail("model-call evidence requires a verified one-attempt transport")


def _sum_usage(rows: list[dict[str, Any]], field: str) -> int | None:
    values = [row["usage"][field] for row in rows]
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def _sum_cost(rows: list[dict[str, Any]], field: str) -> str | None:
    values = [row["usage"][field] for row in rows]
    if not values or any(value is None for value in values):
        return None
    return str(sum(int(value) for value in values))


def _expected_aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "inputTokens": _sum_usage(rows, "inputTokens"),
        "outputTokens": _sum_usage(rows, "outputTokens"),
        "inputCostNanodollars": _sum_cost(rows, "inputCostNanodollars"),
        "outputCostNanodollars": _sum_cost(rows, "outputCostNanodollars"),
        "totalBackendLatencyMs": _sum_usage(
            rows, "totalBackendLatencyMs"
        ),
    }


def _verify_rows(
    receipt: dict[str, Any], context: dict[str, Any]
) -> dict[str, list[str]]:
    rows = receipt["rows"]
    expected_prompts = context["prompts"]
    prompt_ids: list[str] = []
    conversation_names: list[str] = []
    empty_prompt_ids: list[str] = []
    invalid_usage_prompt_ids: list[str] = []
    near_cap_prompt_ids: list[str] = []
    output_reason_by_prompt: dict[str, str] = {}
    policy = receipt["policy"]

    for index, row in enumerate(rows):
        expected = expected_prompts[index]
        prompt_id = row["promptId"]
        conversation_name = f"{expected['itemId']}:{expected['promptId']}"
        if prompt_id != expected["promptId"]:
            _fail(
                "diagnostic rows must be the canonical prompt prefix in order; "
                f"row {index} expected {expected['promptId']}, found {prompt_id}"
            )
        if row["itemId"] != expected["itemId"]:
            _fail(f"row {prompt_id} itemId disagrees with its canonical prompt")
        if row["conversationName"] != conversation_name:
            _fail(f"row {prompt_id} conversation name is not canonical")
        if row["promptSha256"] != expected["promptSha256"]:
            _fail(f"row {prompt_id} prompt digest drifted")
        prompt_ids.append(prompt_id)
        conversation_names.append(row["conversationName"])

        output_text = row["outputText"]
        omitted_reason = row["outputOmittedReason"]
        if omitted_reason is None:
            if row["outputType"] != "str" or not isinstance(output_text, str):
                _fail(f"row {prompt_id} retained output must be a string")
            expected_output_digest = _sha256(output_text.encode("utf-8"))
            if row["outputSha256"] != expected_output_digest:
                _fail(f"row {prompt_id} output digest disagrees with outputText")
            if row["outputCharacters"] != len(output_text):
                _fail(f"row {prompt_id} output character count drifted")
            if len(output_text) > MAX_OUTPUT_CHARACTERS:
                _fail(f"row {prompt_id} retained an oversized output")
            if not output_text.strip():
                empty_prompt_ids.append(prompt_id)
        elif omitted_reason == "non_string":
            if row["outputType"] == "str":
                _fail(f"row {prompt_id} non-string omission claims string type")
            if any(
                row[field] is not None
                for field in ("outputText", "outputSha256", "outputCharacters")
            ):
                _fail(
                    f"row {prompt_id} non-string omission retained output fields"
                )
            output_reason_by_prompt[prompt_id] = "non_string_output"
        elif omitted_reason == "character_limit":
            if row["outputType"] != "str" or output_text is not None:
                _fail(
                    f"row {prompt_id} character-limit omission has invalid type/text"
                )
            if row["outputSha256"] is None:
                _fail(
                    f"row {prompt_id} character-limit omission lost its digest"
                )
            characters = row["outputCharacters"]
            if characters is None or characters <= MAX_OUTPUT_CHARACTERS:
                _fail(
                    f"row {prompt_id} character-limit omission is below the limit"
                )
            output_reason_by_prompt[prompt_id] = "output_character_limit"

        usage = row["usage"]
        input_tokens = usage["inputTokens"]
        output_tokens = usage["outputTokens"]
        if (
            input_tokens is None
            or input_tokens <= 0
            or output_tokens is None
            or output_tokens <= 0
            or output_tokens > policy["maxOutputTokensRequested"]
        ):
            invalid_usage_prompt_ids.append(prompt_id)
        elif output_tokens >= (
            policy["maxOutputTokensRequested"]
            - policy["nearCapMarginTokens"]
        ):
            near_cap_prompt_ids.append(prompt_id)

    if len(prompt_ids) != len(set(prompt_ids)):
        _fail("diagnostic receipt contains duplicate prompt rows")
    if len(conversation_names) != len(set(conversation_names)):
        _fail("diagnostic receipt contains duplicate conversation names")
    if receipt["calls"]["aggregateUsage"] != _expected_aggregate_usage(rows):
        _fail("aggregate usage disagrees with the retained rows")
    return {
        "emptyPromptIds": empty_prompt_ids,
        "invalidUsagePromptIds": invalid_usage_prompt_ids,
        "nearCapPromptIds": near_cap_prompt_ids,
        "outputReasons": [
            output_reason_by_prompt[prompt_id]
            for prompt_id in prompt_ids
            if prompt_id in output_reason_by_prompt
        ],
    }


def _verify_call_accounting(
    receipt: dict[str, Any], context: dict[str, Any]
) -> None:
    calls = receipt["calls"]
    planned = calls["planned"]
    attempted = calls["attempted"]
    completed = calls["completed"]
    rows = receipt["rows"]
    if planned != len(context["prompts"]):
        _fail("planned call count disagrees with the canonical prompt set")
    if not 0 <= completed <= attempted <= planned:
        _fail("call counts must satisfy 0 <= completed <= attempted <= planned")
    if completed != len(rows):
        _fail("completed call count must equal the retained row count")

    active = calls.get("activeCall")
    if active is None:
        if attempted != completed:
            _fail("a call-count gap requires an explicit active call boundary")
        return
    if receipt["status"] != "blocked" or receipt["phase"] != "model_calls":
        _fail("an active call is valid only in a blocked model_calls receipt")
    state = active["state"]
    if state == "prepared":
        if attempted != completed or completed >= planned:
            _fail("prepared active call has incoherent call counts")
        expected = context["prompts"][completed]
    elif state == "dispatching":
        if attempted != completed + 1:
            _fail("dispatching active call has incoherent call counts")
        expected = context["prompts"][completed]
    elif state == "returned":
        if attempted != completed or completed == 0:
            _fail("returned active call has incoherent call counts")
        expected = context["prompts"][completed - 1]
    else:  # The schema should make this unreachable.
        _fail(f"unsupported active call state: {state}")
    expected_name = f"{expected['itemId']}:{expected['promptId']}"
    if active["promptId"] != expected["promptId"]:
        _fail("active call prompt does not match the canonical call boundary")
    if active["conversationName"] != expected_name:
        _fail("active call conversation name is not canonical")


def _verify_diagnostics(
    receipt: dict[str, Any], derived: dict[str, list[str]]
) -> None:
    diagnostics = receipt["diagnostics"]
    for field in (
        "emptyPromptIds",
        "invalidUsagePromptIds",
        "nearCapPromptIds",
    ):
        if diagnostics[field] != derived[field]:
            _fail(f"diagnostics.{field} disagrees with retained rows")

    reasons = diagnostics["blockedReasons"]
    if reasons != sorted(reasons):
        _fail("diagnostic blocked reasons must be sorted canonically")
    row_reasons = set(derived["outputReasons"])
    if derived["emptyPromptIds"]:
        row_reasons.add("empty_output")
    if derived["invalidUsagePromptIds"]:
        row_reasons.add("invalid_usage")
    if derived["nearCapPromptIds"]:
        row_reasons.add("near_output_cap")
    if not row_reasons.issubset(reasons):
        _fail("blocked reasons omit a condition derived from retained rows")
    row_reason_names = {
        "empty_output",
        "invalid_usage",
        "near_output_cap",
        "non_string_output",
        "output_character_limit",
    }
    if set(reasons) & row_reason_names != row_reasons:
        _fail("blocked reasons claim a row condition not present in retained rows")


def _verify_status_machine(receipt: dict[str, Any]) -> None:
    status = receipt["status"]
    phase = receipt["phase"]
    runtime = receipt["runtime"]
    dependency = receipt["dependency"]["observed"]
    model = receipt["modelIdentity"]
    calls = receipt["calls"]
    diagnostics = receipt["diagnostics"]
    reasons = set(diagnostics["blockedReasons"])
    failure = diagnostics["failure"]
    active = calls.get("activeCall")

    if status == "complete":
        if phase != "complete":
            _fail("complete receipt must end in the complete phase")
        if not runtime["conformant"] or dependency["status"] != "verified":
            _fail("complete receipt requires conformant runtime and package")
        if model["status"] != "observed" or model["outputTokenParameter"] is None:
            _fail("complete receipt requires a supported observed model")
        if not (
            calls["attempted"]
            == calls["completed"]
            == calls["planned"]
        ):
            _fail("complete receipt requires all planned calls to complete")
        if active is not None:
            _fail("complete receipt cannot retain an active call")
        if reasons or failure is not None:
            _fail("complete receipt cannot retain blocked diagnostics")
        return

    if phase == "complete":
        _fail("blocked receipt cannot claim the complete phase")
    if not reasons:
        _fail("blocked receipt requires at least one blocked reason")

    if phase == "runtime_preflight":
        if reasons != {"runtime_mismatch"} or runtime["conformant"]:
            _fail("runtime preflight reason disagrees with runtime observation")
        if dependency["status"] != "unchecked":
            _fail("runtime preflight must leave the package unchecked")
        if model["status"] != "unavailable":
            _fail("runtime preflight cannot claim an observed model")
        if calls["attempted"] or calls["completed"] or receipt["rows"]:
            _fail("runtime preflight cannot claim model calls")
        if active is not None or failure is not None:
            _fail("runtime preflight cannot retain a call or failure")
        return

    if not runtime["conformant"]:
        _fail(f"{phase} requires a conformant runtime")

    if phase == "package_preflight":
        if reasons != {"package_unavailable_or_drifted"}:
            _fail("package preflight has an incoherent blocked reason")
        if dependency["status"] != "blocked":
            _fail("package preflight must record a blocked dependency")
        if model["status"] != "unavailable":
            _fail("package preflight cannot claim an observed model")
        if calls["attempted"] or calls["completed"] or receipt["rows"]:
            _fail("package preflight cannot claim model calls")
        if active is not None or failure is None or failure["promptId"] is not None:
            _fail("package preflight requires a package failure without promptId")
        return

    if dependency["status"] != "verified":
        _fail(f"{phase} requires a verified package")

    if phase == "model_preflight":
        allowed = {
            "model_identity_unavailable",
            "sdk_api_incompatible",
            "transport_retry_policy_unverified",
            "unsupported_model_transport",
        }
        if len(reasons) != 1 or not reasons.issubset(allowed):
            _fail("model preflight has an incoherent blocked reason")
        if calls["attempted"] or calls["completed"] or receipt["rows"]:
            _fail("model preflight cannot claim model calls")
        if active is not None or failure is not None:
            _fail("model preflight cannot retain a call or failure")
        if reasons == {"model_identity_unavailable"}:
            if model["status"] != "unavailable":
                _fail("model identity reason disagrees with model status")
        elif model["status"] != "observed":
            _fail("transport/SDK preflight requires an observed model")
        if reasons == {"unsupported_model_transport"}:
            if model["outputTokenParameter"] is not None:
                _fail("unsupported transport cannot name a token parameter")
        elif reasons == {"sdk_api_incompatible"}:
            if model["outputTokenParameter"] is None:
                _fail("SDK API check requires a supported model transport")
        elif reasons == {"transport_retry_policy_unverified"}:
            if model["outputTokenParameter"] is None:
                _fail("retry-policy check requires a supported model transport")
            if model["transportMaxAttempts"] == 1:
                _fail("retry-policy reason disagrees with the attempt bound")
        return

    if phase != "model_calls":
        _fail(f"unsupported blocked phase: {phase}")
    if model["status"] != "observed" or model["outputTokenParameter"] is None:
        _fail("model_calls requires a supported observed model")

    failure_reasons = {
        "model_call_failed",
        "chat_open_failed",
        "chat_close_failed",
    }
    present_failure_reasons = reasons & failure_reasons
    if failure is None:
        if present_failure_reasons:
            _fail("call failure reason requires failure details")
    else:
        if active is None:
            _fail("model call failure must retain its active call boundary")
        allowed_failure_reason_sets = {
            frozenset({"chat_open_failed"}),
            frozenset({"chat_close_failed"}),
            frozenset({"model_call_failed"}),
            frozenset({"model_call_failed", "chat_close_failed"}),
        }
        if frozenset(present_failure_reasons) not in allowed_failure_reason_sets:
            _fail("model call failure has an incoherent failure-reason set")
        if failure["promptId"] is None:
            _fail("model call failure requires a promptId")
        if active is not None:
            expected_by_state = {
                "prepared": {"chat_open_failed"},
                "dispatching": {"model_call_failed"},
                "returned": {"chat_close_failed"},
            }
            if not expected_by_state[active["state"]].issubset(
                present_failure_reasons
            ):
                _fail("failure reason disagrees with the active call state")
        expected_prompt_id = active["promptId"]
        if failure["promptId"] != expected_prompt_id:
            _fail("failure promptId disagrees with the active call boundary")

    row_reason_names = {
        "empty_output",
        "invalid_usage",
        "near_output_cap",
        "non_string_output",
        "output_character_limit",
    }
    if (
        active is not None
        and active["state"] in {"prepared", "dispatching"}
        and reasons & row_reason_names
    ):
        _fail("an unreturned active call cannot follow an invalid retained row")
    base_reasons = reasons - row_reason_names - failure_reasons
    if failure is not None:
        allowed_base_reasons = (
            {"canary_incomplete"}
            if present_failure_reasons == {"chat_close_failed"}
            and not reasons & row_reason_names
            else set()
        )
        if base_reasons != allowed_base_reasons:
            _fail("failed model call cannot claim unrelated blocked reasons")
    elif reasons & row_reason_names:
        if base_reasons:
            _fail("bad model observation cannot claim unrelated blocked reasons")
    elif base_reasons != {"canary_incomplete"}:
        _fail("incomplete model_calls receipt must say canary_incomplete")


def verify_kaggle_diagnostic_receipt(receipt: dict[str, Any]) -> None:
    """Validate a v0.2 Kaggle deployment-canary receipt, failing closed.

    Structural failures raise ``SchemaValidationError``; semantic failures raise
    ``KaggleDiagnosticReceiptError``.
    """

    # Structural validation is deliberately first. Semantic code can therefore
    # index required fields without accepting Python values that are invalid JSON.
    validate(receipt, load_schema(RECEIPT_SCHEMA_PATH))
    _verify_artifact_id(receipt)
    context = _expected_context()
    _verify_definition(receipt, context)

    started_at = _parse_timestamp(receipt["startedAt"], role="startedAt")
    ended_at = _parse_timestamp(receipt["endedAt"], role="endedAt")
    if ended_at < started_at:
        _fail("diagnostic receipt ended before it started")

    _verify_runtime(receipt)
    _verify_dependency(receipt)
    _verify_model_identity(receipt)
    _verify_call_accounting(receipt, context)
    derived = _verify_rows(receipt, context)
    _verify_diagnostics(receipt, derived)
    _verify_status_machine(receipt)


def _requires_manual_review(receipt: dict[str, Any]) -> bool:
    return receipt["calls"].get("activeCall") is not None or (
        receipt["status"] == "blocked" and receipt["calls"]["attempted"] > 0
    )


def diagnostic_receipt_requires_manual_review(receipt: dict[str, Any]) -> bool:
    """Return whether rerunning a valid blocked receipt could duplicate a call."""

    verify_kaggle_diagnostic_receipt(receipt)
    return _requires_manual_review(receipt)


def parse_kaggle_diagnostic_receipt_bytes(raw: bytes) -> dict[str, Any]:
    """Parse bounded strict JSON bytes and verify one diagnostic receipt."""

    if len(raw) > MAX_RECEIPT_BYTES:
        _fail(
            f"diagnostic receipt exceeds the {MAX_RECEIPT_BYTES}-byte safety limit"
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except UnicodeDecodeError as exc:
        raise KaggleDiagnosticReceiptError(
            "diagnostic receipt is not valid UTF-8"
        ) from exc
    except json.JSONDecodeError as exc:
        raise KaggleDiagnosticReceiptError(
            f"diagnostic receipt is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail("diagnostic receipt root must be a JSON object")
    verify_kaggle_diagnostic_receipt(value)
    return value


def load_kaggle_diagnostic_receipt(path: str | Path) -> dict[str, Any]:
    """Load one bounded strict-JSON receipt and verify it semantically."""

    receipt_path = Path(path)
    try:
        with receipt_path.open("rb") as handle:
            raw = handle.read(MAX_RECEIPT_BYTES + 1)
    except OSError as exc:
        raise KaggleDiagnosticReceiptError(
            f"could not read diagnostic receipt: {exc}"
        ) from exc
    return parse_kaggle_diagnostic_receipt_bytes(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify one Aleph-Bench v0.2 Kaggle diagnostic receipt."
    )
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        receipt = load_kaggle_diagnostic_receipt(args.receipt)
    except ValueError as exc:
        print(f"invalid Kaggle diagnostic receipt: {exc}", file=sys.stderr)
        return 1
    review = "required" if _requires_manual_review(receipt) else "no"
    print(
        f"valid Kaggle diagnostic receipt: {receipt['id']} "
        f"(manual review: {review})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
