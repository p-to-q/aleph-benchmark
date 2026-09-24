#!/usr/bin/env python3
"""Generate the self-contained Aleph-Bench v0.2 Kaggle deployment canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.engine.platform_package_v0_2 import (
    CHECKSUMS_NAME,
    MANIFEST_NAME,
    _expected_package,
)
from bench.engine.scoring_core import validate_scoring_runtime

OUTPUT_PATH = ROOT / "bench/tasks/kaggle/aleph_bench_v0_2_diagnostic.py"

TASK_NAME = "aleph_bench_deployment_diagnostic_2048_none"
TASK_VERSION = 1
TASK_DESCRIPTION = (
    "Diagnose Aleph-Bench v0.2 Kaggle runtime, package, model transport, and "
    "usage capture with a fixed six-call canary."
)
PACKAGE_MOUNT = "/kaggle/input/aleph-bench-v02-scorer-conformance"
RECEIPT_FILENAME = "aleph-bench-v0.2-kaggle-diagnostic-receipt.json"
CANARY_PROMPT_IDS = (
    "s2-001-r1-p0",
    "s2-001-r1-p1",
    "s2-002-r1-p0",
    "s2-002-r1-p1",
    "s2-003-r1-p0",
    "s2-003-r1-p1",
)
MAX_OUTPUT_TOKENS = 2048
NEAR_CAP_MARGIN_TOKENS = 4
MAX_OUTPUT_CHARACTERS = 65_536
_DEFINITION_SHA256_PLACEHOLDER = "0" * 64
_IMPLEMENTATION_SHA256_PLACEHOLDER = "1" * 64


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _selected_prompts(items_bytes: bytes) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(items_bytes.decode("utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank packaged item line: {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"packaged item line {line_number} is not an object")
        items.append(value)

    selected: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item.get("id")
        for ladder_prompt in item.get("frozenLadder", []):
            prompt_id = ladder_prompt.get("id")
            if prompt_id not in CANARY_PROMPT_IDS:
                continue
            if prompt_id in selected:
                raise ValueError(f"duplicate canary prompt: {prompt_id}")
            expected_item_id = prompt_id.split("-r", 1)[0]
            if item_id != expected_item_id:
                raise ValueError(
                    f"canary item mismatch for {prompt_id}: {item_id!r}"
                )
            expected_paraphrase = int(prompt_id.rsplit("-p", 1)[1])
            expected = {
                "itemId": item_id,
                "promptId": prompt_id,
                "rung": 1,
                "paraphrase": expected_paraphrase,
                "expectedLeakage": "non_leaking_candidate",
            }
            for field, expected_value in expected.items():
                source_field = {
                    "itemId": None,
                    "promptId": "id",
                    "rung": "rung",
                    "paraphrase": "paraphrase",
                    "expectedLeakage": "expectedLeakage",
                }[field]
                observed = item_id if source_field is None else ladder_prompt.get(source_field)
                if observed != expected_value:
                    raise ValueError(
                        f"canary {prompt_id} {field} drift: "
                        f"expected {expected_value!r}, found {observed!r}"
                    )
            prompt = ladder_prompt.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"canary {prompt_id} has no prompt text")
            selected[prompt_id] = {
                **expected,
                "prompt": prompt,
                "promptSha256": _sha256(prompt.encode("utf-8")),
            }

    missing = [prompt_id for prompt_id in CANARY_PROMPT_IDS if prompt_id not in selected]
    if missing:
        raise ValueError(f"missing canonical canary prompts: {missing}")
    return [selected[prompt_id] for prompt_id in CANARY_PROMPT_IDS]


def _generation_context() -> dict[str, Any]:
    validate_scoring_runtime()
    manifest, artifacts, checksums = _expected_package()
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    items_bytes = artifacts["data/public_s2_items.jsonl"]
    scorer_bytes = artifacts["kaggle/_scoring.py"]
    prompts = _selected_prompts(items_bytes)
    package_identity = {
        "id": manifest["id"],
        "protocolVersion": manifest["protocolVersion"],
        "packageVersion": manifest["packageVersion"],
        "datasetId": manifest["datasetId"],
        "datasetItemCount": manifest["datasetItemCount"],
        "datasetSha256": manifest["datasetSha256"],
        "manifest": {
            "path": MANIFEST_NAME,
            "bytes": len(manifest_bytes),
            "sha256": _sha256(manifest_bytes),
        },
        "checksums": {
            "path": CHECKSUMS_NAME,
            "bytes": len(checksums),
            "sha256": _sha256(checksums),
        },
        "items": {
            "path": "data/public_s2_items.jsonl",
            "bytes": len(items_bytes),
            "sha256": _sha256(items_bytes),
        },
        "scoringCore": {
            "path": "kaggle/_scoring.py",
            "bytes": len(scorer_bytes),
            "sha256": _sha256(scorer_bytes),
        },
    }
    policy = {
        "sampleCount": len(prompts),
        "promptIds": list(CANARY_PROMPT_IDS),
        "reasoningRequested": "none",
        "seedRequested": 0,
        "temperatureRequested": 0,
        "maxOutputTokensRequested": MAX_OUTPUT_TOKENS,
        "maxTotalOutputTokens": MAX_OUTPUT_TOKENS * len(prompts),
        "nearCapMarginTokens": NEAR_CAP_MARGIN_TOKENS,
        "retries": 0,
        "transportMaxAttempts": 1,
        "conversationIsolation": "one_named_chat_per_prompt",
        "sdkCall": "llm.prompt",
        "outputTokenParameterByTransport": {
            "kaggle_benchmarks.actors.llms.GoogleGenAI": "max_output_tokens",
            "kaggle_benchmarks.actors.llms.OpenAI": "max_tokens",
        },
    }
    implementation_source = _render_task_source(
        definition_sha256=_DEFINITION_SHA256_PLACEHOLDER,
        implementation_sha256=_IMPLEMENTATION_SHA256_PLACEHOLDER,
        package_identity=package_identity,
        policy=policy,
        prompts=prompts,
    )
    implementation_sha256 = _sha256(implementation_source.encode("utf-8"))
    definition = {
        "task": {
            "name": TASK_NAME,
            "version": TASK_VERSION,
            "description": TASK_DESCRIPTION,
            "implementationSha256": implementation_sha256,
        },
        "requiredRuntime": {
            "pythonVersion": "3.13",
            "unicodeDatabaseVersion": "15.1.0",
        },
        "package": package_identity,
        "policy": policy,
        "prompts": prompts,
    }
    return {
        "definitionSha256": _sha256(_canonical_json_bytes(definition)),
        "implementationSha256": definition["task"]["implementationSha256"],
        "packageIdentity": package_identity,
        "policy": policy,
        "prompts": prompts,
    }


TASK_BODY = r'''

REQUIRED_RUNTIME = {
    "pythonVersion": "3.13",
    "unicodeDatabaseVersion": "15.1.0",
}
RECEIPT_VERSION = "0.2.0"
ARTIFACT_KIND = "kaggle_runtime_canary"
MAX_READ_BYTES = 1_048_576


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pretty_json_bytes(value):
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


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _runtime_observation():
    return {
        "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pythonFullVersion": platform.python_version(),
        "unicodeDatabaseVersion": unicodedata.unidata_version,
    }


def _runtime_conformant(observed):
    return (
        observed.get("pythonVersion") == REQUIRED_RUNTIME["pythonVersion"]
        and observed.get("unicodeDatabaseVersion")
        == REQUIRED_RUNTIME["unicodeDatabaseVersion"]
    )


def _read_exact_regular_file(path, expected):
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"unavailable package artifact {expected['path']}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"package artifact is not a regular file: {expected['path']}")
    if metadata.st_size != expected["bytes"]:
        raise ValueError(
            f"package artifact size mismatch for {expected['path']}: "
            f"expected {expected['bytes']}, found {metadata.st_size}"
        )
    if metadata.st_size > MAX_READ_BYTES:
        raise ValueError(f"package artifact exceeds read limit: {expected['path']}")
    data = path.read_bytes()
    if len(data) != expected["bytes"] or _sha256(data) != expected["sha256"]:
        raise ValueError(f"package artifact digest mismatch: {expected['path']}")
    return data


def _reject_duplicate_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _verify_package(package_root):
    observed = {
        "status": "blocked",
        "mountPath": str(package_root),
        "manifestSha256": None,
        "itemsSha256": None,
        "scoringCoreSha256": None,
    }
    try:
        metadata = package_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("package mount is not a regular directory")
        manifest_bytes = _read_exact_regular_file(
            package_root / PACKAGE_IDENTITY["manifest"]["path"],
            PACKAGE_IDENTITY["manifest"],
        )
        manifest = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
        for field in (
            "id",
            "protocolVersion",
            "packageVersion",
            "datasetId",
            "datasetItemCount",
            "datasetSha256",
        ):
            if manifest.get(field) != PACKAGE_IDENTITY[field]:
                raise ValueError(f"package manifest identity mismatch: {field}")
        _read_exact_regular_file(
            package_root / PACKAGE_IDENTITY["checksums"]["path"],
            PACKAGE_IDENTITY["checksums"],
        )
        items_bytes = _read_exact_regular_file(
            package_root / PACKAGE_IDENTITY["items"]["path"],
            PACKAGE_IDENTITY["items"],
        )
        _read_exact_regular_file(
            package_root / PACKAGE_IDENTITY["scoringCore"]["path"],
            PACKAGE_IDENTITY["scoringCore"],
        )
        expected_by_id = {row["promptId"]: row for row in CANARY_PROMPTS}
        selected = {}
        item_count = 0
        for line_number, line in enumerate(items_bytes.decode("utf-8").splitlines(), 1):
            if not line:
                raise ValueError(f"blank package item line: {line_number}")
            item = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
            item_count += 1
            item_id = item.get("id")
            for ladder_prompt in item.get("frozenLadder", []):
                prompt_id = ladder_prompt.get("id")
                if prompt_id not in expected_by_id:
                    continue
                if prompt_id in selected:
                    raise ValueError(f"duplicate package canary prompt: {prompt_id}")
                expected = expected_by_id[prompt_id]
                prompt = ladder_prompt.get("prompt")
                actual = {
                    "itemId": item_id,
                    "promptId": prompt_id,
                    "rung": ladder_prompt.get("rung"),
                    "paraphrase": ladder_prompt.get("paraphrase"),
                    "expectedLeakage": ladder_prompt.get("expectedLeakage"),
                    "prompt": prompt,
                    "promptSha256": (
                        _sha256(prompt.encode("utf-8")) if isinstance(prompt, str) else None
                    ),
                }
                if actual != expected:
                    raise ValueError(f"package canary prompt drift: {prompt_id}")
                selected[prompt_id] = actual
        if item_count != PACKAGE_IDENTITY["datasetItemCount"]:
            raise ValueError(
                f"package item count mismatch: expected "
                f"{PACKAGE_IDENTITY['datasetItemCount']}, found {item_count}"
            )
        missing = [
            prompt_id for prompt_id in POLICY["promptIds"] if prompt_id not in selected
        ]
        if missing:
            raise ValueError(f"missing package canary prompts: {missing}")
        observed.update(
            {
                "status": "verified",
                "manifestSha256": PACKAGE_IDENTITY["manifest"]["sha256"],
                "itemsSha256": PACKAGE_IDENTITY["items"]["sha256"],
                "scoringCoreSha256": PACKAGE_IDENTITY["scoringCore"]["sha256"],
            }
        )
        return [selected[prompt_id] for prompt_id in POLICY["promptIds"]], observed, None
    except Exception as exc:
        return None, observed, {
            "promptId": None,
            "exceptionType": type(exc).__name__[:200],
            "message": str(exc)[:2000],
        }


def _transport_retry_preflight(llm, model_type):
    sdk_name = {
        "kaggle_benchmarks.actors.llms.OpenAI": "openai",
        "kaggle_benchmarks.actors.llms.GoogleGenAI": "google-genai",
    }.get(model_type)
    try:
        sdk_version = (
            importlib_metadata.version(sdk_name) if sdk_name is not None else None
        )
    except importlib_metadata.PackageNotFoundError:
        sdk_version = None
    observation = {
        "transportSdkName": sdk_name,
        "transportSdkVersion": sdk_version,
        "transportMaxAttempts": None,
    }
    try:
        client = llm.client
        if model_type == "kaggle_benchmarks.actors.llms.OpenAI":
            configure = getattr(client, "with_options", None)
            if not callable(configure):
                return observation, "transport_retry_policy_unverified"
            no_retry_client = configure(max_retries=0)
            if getattr(no_retry_client, "max_retries", None) != 0:
                return observation, "transport_retry_policy_unverified"
            llm.client = no_retry_client
            observation["transportMaxAttempts"] = 1
            return observation, None
        if model_type == "kaggle_benchmarks.actors.llms.GoogleGenAI":
            api_client = getattr(client, "_api_client", None)
            http_options = getattr(api_client, "_http_options", None)
            retry_options = getattr(http_options, "retry_options", object())
            retry_controller = getattr(api_client, "_retry", None)
            stop_strategy = getattr(retry_controller, "stop", None)
            max_attempts = getattr(stop_strategy, "max_attempt_number", None)
            if retry_options is not None or max_attempts != 1:
                return observation, "transport_retry_policy_unverified"
            observation["transportMaxAttempts"] = 1
            return observation, None
    except Exception:
        return observation, "transport_retry_policy_unverified"
    return observation, "unsupported_model_transport"


def _model_preflight(llm):
    model_type = f"{type(llm).__module__}.{type(llm).__name__}"
    token_parameter = POLICY["outputTokenParameterByTransport"].get(model_type)
    slug = getattr(llm, "model", None)
    if not isinstance(slug, str) or not slug:
        slug = getattr(llm, "name", None)
    identity = {
        "status": "observed" if isinstance(slug, str) and slug else "unavailable",
        "slug": slug if isinstance(slug, str) and slug else None,
        "pythonType": model_type,
        "sdkVersion": getattr(kbench, "__version__", None),
        "outputTokenParameter": token_parameter,
        "transportSdkName": None,
        "transportSdkVersion": None,
        "transportMaxAttempts": None,
    }
    if identity["status"] != "observed":
        return identity, None, "model_identity_unavailable"
    if token_parameter is None:
        return identity, None, "unsupported_model_transport"
    try:
        parameters = inspect.signature(llm.prompt).parameters
    except (TypeError, ValueError):
        return identity, None, "sdk_api_incompatible"
    required = {"reasoning", "seed", "temperature", "extra_api_params"}
    if not required.issubset(parameters):
        return identity, None, "sdk_api_incompatible"
    retry_observation, retry_error = _transport_retry_preflight(llm, model_type)
    identity.update(retry_observation)
    if retry_error is not None:
        return identity, None, retry_error
    return identity, token_parameter, None


def _usage_value(usage, field):
    value = getattr(usage, field, None) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _cost_string(value):
    return str(value) if value is not None else None


def _usage_record(chat):
    try:
        usage = chat.usage
    except Exception:
        usage = None
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    input_cost = _usage_value(usage, "input_tokens_cost_nanodollars")
    output_cost = _usage_value(usage, "output_tokens_cost_nanodollars")
    latency = _usage_value(usage, "total_backend_latency_ms")
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "inputCostNanodollars": _cost_string(input_cost),
        "outputCostNanodollars": _cost_string(output_cost),
        "totalBackendLatencyMs": latency,
    }


def _aggregate_usage(rows):
    def total(field):
        values = [row["usage"][field] for row in rows]
        if not values or any(value is None for value in values):
            return None
        return sum(values)

    def total_cost(field):
        values = [row["usage"][field] for row in rows]
        if not values or any(value is None for value in values):
            return None
        return str(sum(int(value) for value in values))

    return {
        "inputTokens": total("inputTokens"),
        "outputTokens": total("outputTokens"),
        "inputCostNanodollars": total_cost("inputCostNanodollars"),
        "outputCostNanodollars": total_cost("outputCostNanodollars"),
        "totalBackendLatencyMs": total("totalBackendLatencyMs"),
    }


def _diagnostics(rows, reasons, failure):
    empty = []
    invalid_usage = []
    near_cap = []
    for row in rows:
        prompt_id = row["promptId"]
        output = row["outputText"]
        if isinstance(output, str) and not output.strip():
            empty.append(prompt_id)
        usage = row["usage"]
        input_tokens = usage["inputTokens"]
        output_tokens = usage["outputTokens"]
        if (
            input_tokens is None
            or input_tokens <= 0
            or output_tokens is None
            or output_tokens <= 0
            or output_tokens > POLICY["maxOutputTokensRequested"]
        ):
            invalid_usage.append(prompt_id)
        elif output_tokens >= (
            POLICY["maxOutputTokensRequested"] - POLICY["nearCapMarginTokens"]
        ):
            near_cap.append(prompt_id)
    derived = list(reasons)
    if empty:
        derived.append("empty_output")
    if invalid_usage:
        derived.append("invalid_usage")
    if near_cap:
        derived.append("near_output_cap")
    return {
        "blockedReasons": sorted(set(derived)),
        "emptyPromptIds": empty,
        "invalidUsagePromptIds": invalid_usage,
        "nearCapPromptIds": near_cap,
        "failure": failure,
    }


def _receipt(
    *,
    started_at,
    ended_at,
    phase,
    runtime,
    dependency,
    model,
    attempted,
    completed,
    active_call,
    rows,
    reasons,
    failure,
):
    diagnostics = _diagnostics(rows, reasons, failure)
    is_complete = (
        phase == "complete"
        and attempted == POLICY["sampleCount"]
        and completed == POLICY["sampleCount"]
        and len(rows) == POLICY["sampleCount"]
        and active_call is None
        and not diagnostics["blockedReasons"]
    )
    payload = {
        "receiptVersion": RECEIPT_VERSION,
        "artifactKind": ARTIFACT_KIND,
        "targetProtocolVersion": PACKAGE_IDENTITY["protocolVersion"],
        "evidenceMode": "none",
        "protocolConformant": False,
        "leaderboardEligible": False,
        "publicationEligible": False,
        "status": "complete" if is_complete else "blocked",
        "phase": phase,
        "startedAt": started_at,
        "endedAt": ended_at,
        "taskIdentity": {
            "name": TASK_NAME,
            "version": TASK_VERSION,
            "description": TASK_DESCRIPTION,
            "definitionSha256": DEFINITION_SHA256,
            "implementationSha256": IMPLEMENTATION_SHA256,
        },
        "runtime": {
            "required": REQUIRED_RUNTIME,
            "observed": runtime,
            "conformant": _runtime_conformant(runtime),
        },
        "dependency": {"expected": PACKAGE_IDENTITY, "observed": dependency},
        "modelIdentity": model,
        "policy": POLICY,
        "calls": {
            "planned": POLICY["sampleCount"],
            "attempted": attempted,
            "completed": completed,
            "activeCall": active_call,
            "aggregateUsage": _aggregate_usage(rows),
        },
        "rows": rows,
        "diagnostics": diagnostics,
        "diagnosticScalar": None,
        "notes": [
            "This receipt is a deployment canary, not an Aleph-Bench result.",
            "It cannot establish AURC, ECL, Elicit, model rank, or shortest-found claims.",
            "A complete canary proves only this fixed six-call transport and capture path.",
            "Never attach this task to the Aleph-Bench v0.2 leaderboard.",
        ],
    }
    receipt = dict(payload)
    receipt["id"] = (
        "aleph-bench-kaggle-diagnostic-v0.2-artifact-"
        + _sha256(_canonical_json_bytes(payload))
    )
    return receipt


def _write_receipt(receipt, receipt_path):
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.", dir=receipt_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_pretty_json_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, receipt_path)
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _row(prompt, conversation_name, output, chat):
    output_type = type(output).__name__
    output_text = output if isinstance(output, str) else None
    omitted_reason = None
    if not isinstance(output, str):
        omitted_reason = "non_string"
    elif len(output) > MAX_OUTPUT_CHARACTERS:
        output_text = None
        omitted_reason = "character_limit"
    return {
        "itemId": prompt["itemId"],
        "promptId": prompt["promptId"],
        "conversationName": conversation_name,
        "promptSha256": prompt["promptSha256"],
        "outputType": output_type,
        "outputText": output_text,
        "outputSha256": (
            _sha256(output.encode("utf-8")) if isinstance(output, str) else None
        ),
        "outputCharacters": len(output) if isinstance(output, str) else None,
        "outputOmittedReason": omitted_reason,
        "usage": _usage_record(chat),
    }


def _active_call(prompt, conversation_name, state):
    return {
        "promptId": prompt["promptId"],
        "conversationName": conversation_name,
        "state": state,
    }


def run_deployment_diagnostic(
    llm,
    *,
    chats,
    package_root=DEFAULT_PACKAGE_ROOT,
    receipt_path=DEFAULT_RECEIPT_PATH,
    observed_runtime=None,
    clock=None,
):
    clock = clock or _utc_now
    started_at = clock()
    runtime = observed_runtime if observed_runtime is not None else _runtime_observation()
    unchecked_dependency = {
        "status": "unchecked",
        "mountPath": str(package_root),
        "manifestSha256": None,
        "itemsSha256": None,
        "scoringCoreSha256": None,
    }
    unavailable_model = {
        "status": "unavailable",
        "slug": None,
        "pythonType": None,
        "sdkVersion": None,
        "outputTokenParameter": None,
        "transportSdkName": None,
        "transportSdkVersion": None,
        "transportMaxAttempts": None,
    }

    def publish(
        phase,
        dependency,
        model,
        attempted,
        completed,
        active_call,
        rows,
        reasons,
        failure,
    ):
        receipt = _receipt(
            started_at=started_at,
            ended_at=clock(),
            phase=phase,
            runtime=runtime,
            dependency=dependency,
            model=model,
            attempted=attempted,
            completed=completed,
            active_call=active_call,
            rows=rows,
            reasons=reasons,
            failure=failure,
        )
        _write_receipt(receipt, receipt_path)
        return receipt

    # This gate deliberately precedes package reads, chat creation, and any llm access.
    if not _runtime_conformant(runtime):
        return publish(
            "runtime_preflight",
            unchecked_dependency,
            unavailable_model,
            0,
            0,
            None,
            [],
            ["runtime_mismatch"],
            None,
        )

    prompts, dependency, package_failure = _verify_package(Path(package_root))
    if prompts is None:
        return publish(
            "package_preflight",
            dependency,
            unavailable_model,
            0,
            0,
            None,
            [],
            ["package_unavailable_or_drifted"],
            package_failure,
        )

    model, token_parameter, model_reason = _model_preflight(llm)
    if model_reason is not None:
        return publish(
            "model_preflight",
            dependency,
            model,
            0,
            0,
            None,
            [],
            [model_reason],
            None,
        )
    if not hasattr(chats, "new") or not callable(chats.new):
        return publish(
            "model_preflight",
            dependency,
            model,
            0,
            0,
            None,
            [],
            ["sdk_api_incompatible"],
            None,
        )

    attempted = 0
    completed = 0
    rows = []
    publish(
        "model_calls",
        dependency,
        model,
        attempted,
        completed,
        None,
        rows,
        ["canary_incomplete"],
        None,
    )
    for prompt in prompts:
        conversation_name = f"{prompt['itemId']}:{prompt['promptId']}"
        prepared_call = _active_call(prompt, conversation_name, "prepared")
        publish(
            "model_calls",
            dependency,
            model,
            attempted,
            completed,
            prepared_call,
            rows,
            ["canary_incomplete"],
            None,
        )
        try:
            chat_manager = chats.new(conversation_name)
            chat = chat_manager.__enter__()
        except Exception as exc:
            return publish(
                "model_calls",
                dependency,
                model,
                attempted,
                completed,
                prepared_call,
                rows,
                ["chat_open_failed"],
                {
                    "promptId": prompt["promptId"],
                    "exceptionType": type(exc).__name__[:200],
                    "message": str(exc)[:2000],
                },
            )

        attempted += 1
        dispatching_call = _active_call(prompt, conversation_name, "dispatching")
        try:
            publish(
                "model_calls",
                dependency,
                model,
                attempted,
                completed,
                dispatching_call,
                rows,
                ["canary_incomplete"],
                None,
            )
        except BaseException:
            exception_info = sys.exc_info()
            try:
                chat_manager.__exit__(*exception_info)
            finally:
                raise

        try:
            output = llm.prompt(
                prompt["prompt"],
                reasoning=POLICY["reasoningRequested"],
                seed=POLICY["seedRequested"],
                temperature=POLICY["temperatureRequested"],
                extra_api_params={
                    token_parameter: POLICY["maxOutputTokensRequested"]
                },
            )
        except BaseException as exc:
            exception_info = sys.exc_info()
            close_failure = None
            try:
                chat_manager.__exit__(*exception_info)
            except Exception as close_exc:
                close_failure = close_exc
            if not isinstance(exc, Exception):
                raise
            reasons = ["model_call_failed"]
            message = str(exc)[:2000]
            if close_failure is not None:
                reasons.append("chat_close_failed")
                message = (
                    f"{message}; chat close failed: "
                    f"{type(close_failure).__name__}: {close_failure}"
                )[:2000]
            return publish(
                "model_calls",
                dependency,
                model,
                attempted,
                completed,
                dispatching_call,
                rows,
                reasons,
                {
                    "promptId": prompt["promptId"],
                    "exceptionType": type(exc).__name__[:200],
                    "message": message,
                },
            )

        row = _row(prompt, conversation_name, output, chat)
        rows.append(row)
        completed += 1
        row_reasons = []
        if row["outputOmittedReason"] == "non_string":
            row_reasons.append("non_string_output")
        elif row["outputOmittedReason"] == "character_limit":
            row_reasons.append("output_character_limit")
        diagnostics = _diagnostics(rows, row_reasons, None)
        returned_call = _active_call(prompt, conversation_name, "returned")
        persisted_reasons = (
            diagnostics["blockedReasons"]
            if diagnostics["blockedReasons"]
            else ["canary_incomplete"]
        )
        try:
            publish(
                "model_calls",
                dependency,
                model,
                attempted,
                completed,
                returned_call,
                rows,
                persisted_reasons,
                None,
            )
        except BaseException:
            exception_info = sys.exc_info()
            try:
                chat_manager.__exit__(*exception_info)
            finally:
                raise

        try:
            chat_manager.__exit__(None, None, None)
        except Exception as exc:
            return publish(
                "model_calls",
                dependency,
                model,
                attempted,
                completed,
                returned_call,
                rows,
                [*persisted_reasons, "chat_close_failed"],
                {
                    "promptId": prompt["promptId"],
                    "exceptionType": type(exc).__name__[:200],
                    "message": str(exc)[:2000],
                },
            )

        if diagnostics["blockedReasons"]:
            return publish(
                "model_calls",
                dependency,
                model,
                attempted,
                completed,
                None,
                rows,
                diagnostics["blockedReasons"],
                None,
            )
        publish(
            "model_calls",
            dependency,
            model,
            attempted,
            completed,
            None,
            rows,
            ["canary_incomplete"],
            None,
        )

    return publish(
        "complete",
        dependency,
        model,
        attempted,
        completed,
        None,
        rows,
        [],
        None,
    )


# %%
@kbench.task(
    name=TASK_NAME,
    description=TASK_DESCRIPTION,
    version=TASK_VERSION,
)
def aleph_bench_deployment_diagnostic_2048_none(llm) -> dict:
    receipt = run_deployment_diagnostic(llm, chats=kbench.chats)
    kbench.assertions.assert_true(
        receipt["status"] == "complete",
        expectation=(
            "The deployment diagnostic must complete without blocked reasons; "
            "inspect the returned receipt when this assertion fails."
        ),
    )
    return receipt


# %%
aleph_bench_deployment_diagnostic_2048_none.run(kbench.llm)
'''


def render_task_source() -> str:
    context = _generation_context()
    return _render_task_source(
        definition_sha256=context["definitionSha256"],
        implementation_sha256=context["implementationSha256"],
        package_identity=context["packageIdentity"],
        policy=context["policy"],
        prompts=context["prompts"],
    )


def _render_task_source(
    *,
    definition_sha256: str,
    implementation_sha256: str,
    package_identity: dict[str, Any],
    policy: dict[str, Any],
    prompts: list[dict[str, Any]],
) -> str:
    prompts_json = json.dumps(
        prompts, ensure_ascii=True, indent=2, sort_keys=True
    )
    package_json = json.dumps(
        package_identity, ensure_ascii=True, indent=2, sort_keys=True
    )
    policy_json = json.dumps(
        policy, ensure_ascii=True, indent=2, sort_keys=True
    )
    header = f'''# Generated by bench/tasks/kaggle/generate_v0_2_diagnostic.py.
# Do not edit this file directly; regenerate it on Python 3.13 / UCD 15.1.0.

import hashlib
import inspect
import json
import os
import platform
import stat
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import kaggle_benchmarks as kbench


TASK_NAME = {TASK_NAME!r}
TASK_VERSION = {TASK_VERSION}
TASK_DESCRIPTION = {TASK_DESCRIPTION!r}
DEFINITION_SHA256 = {definition_sha256!r}
IMPLEMENTATION_SHA256 = {implementation_sha256!r}
DEFAULT_PACKAGE_ROOT = Path({PACKAGE_MOUNT!r})
DEFAULT_RECEIPT_PATH = Path.cwd() / {RECEIPT_FILENAME!r}
MAX_OUTPUT_CHARACTERS = {MAX_OUTPUT_CHARACTERS}
PACKAGE_IDENTITY = json.loads(r\'''{package_json}\''')
POLICY = json.loads(r\'''{policy_json}\''')
CANARY_PROMPTS = tuple(json.loads(r\'''{prompts_json}\'''))
'''
    return header + TASK_BODY.lstrip("\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the checked-in generated task differs from canonical output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    expected = render_task_source().encode("utf-8")
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_bytes() != expected:
            print(
                f"generated Kaggle diagnostic task is stale: {OUTPUT_PATH}",
                file=sys.stderr,
            )
            return 1
        print(f"generated Kaggle diagnostic task is current: {OUTPUT_PATH}")
        return 0
    OUTPUT_PATH.write_bytes(expected)
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
