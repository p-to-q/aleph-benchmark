from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from .adapters import (
    MockAdapter,
    RetainedOutputReplayAdapter,
    normalize_model_id,
    validate_deployment_id,
)
from .adapters.hosted_black_box import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    HOSTED_ADAPTER_ID,
    HOSTED_ADAPTER_VERSION,
    HOSTED_REQUEST_PAYLOAD_VERSION,
    HOSTED_WIRE_PROTOCOL,
    MAX_HOSTED_RETRIES,
    MAX_RETRY_DELAY_SECONDS,
)
from .frozen_ladder import (
    FIXED_CREATED_AT as RESULT_FIXED_CREATED_AT,
    evaluate_item,
    result_notes,
    stable_dataset_path,
    summarize_model_runs,
)
from .manifest import (
    FIXED_CREATED_AT as MANIFEST_FIXED_CREATED_AT,
    build_prompt_receipts,
    manifest_notes,
)
from .protocol import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_RERUNS,
    FROZEN_DATASET_HASH_ALGORITHM,
    FROZEN_DATASET_ID,
    FROZEN_DATASET_ITEM_COUNT,
    FROZEN_DATASET_SHA256,
    aleph_run_artifact_id,
    decoding_for_evidence_mode,
    manifest_artifact_id,
    result_artifact_id,
)
from .schema_validation import SchemaValidationError, load_schema, validate
from .scoring_core import PROTOCOL_VERSION, validate_scoring_runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_DATASET_ITEM_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_DIRECTORY", 0)
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        value[key] = item
    return value


def _reject_json_surrogates(value: Any, *, role: str) -> None:
    """Reject strings that cannot be encoded as portable UTF-8.

    JSON permits escaped UTF-16 surrogate code points, but Aleph artifacts are
    hashed and replayed as UTF-8.  Walking iteratively also keeps this boundary
    check independent of Python's recursion limit.
    """

    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in current):
                raise ValueError(
                    f"invalid {role} JSON: Unicode surrogate code points are forbidden"
                )
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


def _decode_json(raw: bytes, *, role: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"invalid {role} JSON: {exc}") from exc
    _reject_json_surrogates(value, role=role)
    return value


def _stat_snapshot(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_fd(
    fd: int, *, role: str, max_bytes: int
) -> tuple[bytes, tuple[int, ...]]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{role} must be a regular file")
    if before.st_nlink != 1:
        raise ValueError(f"{role} must have exactly one hard link")
    if before.st_size > max_bytes:
        raise ValueError(
            f"{role} exceeds the {max_bytes}-byte verification limit"
        )
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > max_bytes:
        raise ValueError(f"{role} exceeds the {max_bytes}-byte verification limit")
    after = os.fstat(fd)
    if _stat_snapshot(before) != _stat_snapshot(after) or len(raw) != after.st_size:
        raise ValueError(f"{role} changed while it was being read")
    return raw, _stat_snapshot(after)


def _read_regular_path_snapshot(
    path: Path, *, role: str, max_bytes: int
) -> tuple[bytes, dict[str, Any]]:
    try:
        fd = os.open(path, _READ_FLAGS)
    except OSError as exc:
        raise ValueError(f"could not open {role}: {exc}") from exc
    try:
        raw, snapshot = _read_regular_fd(fd, role=role, max_bytes=max_bytes)
        return raw, {
            "path": Path(path),
            "role": role,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "stat": snapshot,
        }
    except OSError as exc:
        raise ValueError(f"could not read {role}: {exc}") from exc
    finally:
        os.close(fd)


def _read_regular_path(path: Path, *, role: str, max_bytes: int) -> bytes:
    raw, _ = _read_regular_path_snapshot(path, role=role, max_bytes=max_bytes)
    return raw


def _recheck_file_binding(binding: dict[str, Any], errors: list[str]) -> None:
    path = binding["path"]
    try:
        fd = os.open(path, _READ_FLAGS)
    except OSError as exc:
        errors.append(f"{path}: {binding['role']} path changed after reading: {exc}")
        return
    try:
        current = os.fstat(fd)
        if _stat_snapshot(current) != binding["stat"]:
            errors.append(f"{path}: {binding['role']} path changed after reading")
    except OSError as exc:
        errors.append(f"{path}: could not recheck {binding['role']}: {exc}")
    finally:
        os.close(fd)


def _dataset_sha256(snapshots: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for filename, content in snapshots:
        encoded = filename.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _expected_model_receipt(model: str) -> tuple[str, int]:
    canonical, evidence_mode, uses_configured_reruns = normalize_model_id(model)
    if canonical != model:
        raise ValueError(f"non-canonical v0.2 result model id {model!r}")
    return evidence_mode, DEFAULT_RERUNS if uses_configured_reruns else 1


def _verify_adapter_identity(
    *,
    identity: Any,
    model: str,
    evidence_mode: str,
    context: str,
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(identity, dict):
        errors.append(f"{context} lacks adapter identity")
        return {}
    if identity.get("model") != model:
        errors.append(f"{context} adapter identity model drifted")
    if identity.get("observationMode") != evidence_mode:
        errors.append(f"{context} adapter identity evidence mode drifted")
    if evidence_mode == "mock":
        expected = MockAdapter(model).evidence_identity()
        if identity != expected:
            errors.append(f"{context} mock adapter identity is not canonical")
        return identity
    if evidence_mode != "black_box":
        errors.append(f"{context} has unsupported adapter evidence mode")
        return identity

    expected_fields = {
        "adapterId": HOSTED_ADAPTER_ID,
        "adapterVersion": HOSTED_ADAPTER_VERSION,
        "temperature": 0.0,
        "wireProtocol": HOSTED_WIRE_PROTOCOL,
        "requestPayloadVersion": HOSTED_REQUEST_PAYLOAD_VERSION,
        "maxTokens": DEFAULT_MAX_TOKENS,
        "providerModel": model.removeprefix("hosted:"),
        "timeoutSeconds": DEFAULT_TIMEOUT_SECONDS,
    }
    for field, expected in expected_fields.items():
        if identity.get(field) != expected:
            errors.append(f"{context} adapter identity {field} is not canonical")
    endpoint_digest = identity.get("endpointSha256")
    if (
        not isinstance(endpoint_digest, str)
        or len(endpoint_digest) != 64
        or any(character not in "0123456789abcdef" for character in endpoint_digest)
    ):
        errors.append(f"{context} adapter identity endpointSha256 is invalid")
    try:
        validate_deployment_id(identity.get("deploymentId"))
    except (TypeError, ValueError):
        errors.append(f"{context} adapter identity deploymentId is invalid")
    return identity


def _load_dataset(
    data_dir: Path, errors: list[str]
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any] | None]:
    try:
        directory_fd = os.open(data_dir, _DIRECTORY_FLAGS)
    except OSError as exc:
        errors.append(f"{data_dir}: could not open dataset directory: {exc}")
        return [], None, None
    snapshots: list[tuple[str, bytes]] = []
    entry_bindings: dict[str, tuple[int, ...]] = {}
    try:
        directory_snapshot = _stat_snapshot(os.fstat(directory_fd))
    except OSError as exc:
        os.close(directory_fd)
        errors.append(f"{data_dir}: could not inspect dataset directory: {exc}")
        return [], None, None
    try:
        try:
            with os.scandir(directory_fd) as iterator:
                entries = []
                for entry in iterator:
                    entries.append(entry)
                    if len(entries) > FROZEN_DATASET_ITEM_COUNT:
                        break
                entries.sort(key=lambda entry: entry.name)
        except OSError as exc:
            errors.append(f"{data_dir}: could not enumerate dataset directory: {exc}")
            return [], None, None
        if len(entries) != FROZEN_DATASET_ITEM_COUNT:
            relation = "more than" if len(entries) > FROZEN_DATASET_ITEM_COUNT else str(len(entries))
            errors.append(
                f"{data_dir}: expected exactly {FROZEN_DATASET_ITEM_COUNT} dataset entries, "
                f"found {relation}"
            )
        if len(entries) > FROZEN_DATASET_ITEM_COUNT:
            return [], None, None
        for entry in entries:
            name = entry.name
            if not isinstance(name, str) or not name.endswith(".json"):
                errors.append(
                    f"{data_dir}: dataset contains an unexpected entry {name!r}"
                )
                continue
            try:
                name.encode("utf-8")
            except UnicodeError:
                errors.append(
                    f"{data_dir}: dataset entry name is not valid UTF-8: {name!r}"
                )
                continue
            try:
                fd = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
            except OSError as exc:
                errors.append(f"{data_dir / name}: could not open dataset item: {exc}")
                continue
            try:
                raw, snapshot = _read_regular_fd(
                    fd,
                    role=f"dataset item {name}",
                    max_bytes=MAX_DATASET_ITEM_BYTES,
                )
                snapshots.append((name, raw))
                entry_bindings[name] = snapshot
            except (OSError, ValueError) as exc:
                errors.append(f"{data_dir / name}: {exc}")
            finally:
                os.close(fd)
        try:
            directory_after = _stat_snapshot(os.fstat(directory_fd))
        except OSError as exc:
            errors.append(f"{data_dir}: could not recheck dataset directory: {exc}")
        else:
            if directory_after != directory_snapshot:
                errors.append(f"{data_dir}: dataset directory changed while it was read")
    finally:
        os.close(directory_fd)

    items: list[dict[str, Any]] = []
    for name, raw in snapshots:
        try:
            item = _decode_json(raw, role=f"dataset item {name}")
            if not isinstance(item, dict):
                raise TypeError("BenchItem must be a JSON object")
            items.append(item)
        except (TypeError, ValueError) as exc:
            errors.append(f"{data_dir / name}: {exc}")
    if not snapshots:
        errors.append(f"{data_dir}: no BenchItems found")
        return items, None, None
    binding = {
        "entries": entry_bindings,
        "path": Path(data_dir),
        "stat": directory_snapshot,
    }
    return items, _dataset_sha256(snapshots), binding


def _recheck_dataset_binding(binding: dict[str, Any], errors: list[str]) -> None:
    data_dir = binding["path"]
    try:
        directory_fd = os.open(data_dir, _DIRECTORY_FLAGS)
    except OSError as exc:
        errors.append(f"{data_dir}: dataset path changed after reading: {exc}")
        return
    try:
        try:
            current_directory = _stat_snapshot(os.fstat(directory_fd))
        except OSError as exc:
            errors.append(f"{data_dir}: could not recheck dataset directory: {exc}")
            return
        if current_directory != binding["stat"]:
            errors.append(f"{data_dir}: dataset directory changed after reading")
            return
        try:
            with os.scandir(directory_fd) as iterator:
                names = []
                for entry in iterator:
                    names.append(entry.name)
                    if len(names) > FROZEN_DATASET_ITEM_COUNT:
                        break
        except OSError as exc:
            errors.append(f"{data_dir}: could not recheck dataset directory: {exc}")
            return
        if set(names) != set(binding["entries"]):
            errors.append(f"{data_dir}: dataset entry set changed after reading")
            return
        for name, expected in binding["entries"].items():
            try:
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                errors.append(f"{data_dir / name}: dataset item changed after reading: {exc}")
                continue
            if _stat_snapshot(current) != expected:
                errors.append(f"{data_dir / name}: dataset item changed after reading")
    finally:
        os.close(directory_fd)


def _load_object(
    path: Path,
    role: str,
    errors: list[str],
    *,
    max_bytes: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        raw, binding = _read_regular_path_snapshot(
            path, role=role, max_bytes=max_bytes
        )
        value = _decode_json(raw, role=role)
    except ValueError as exc:
        errors.append(f"{path}: {exc}")
        return None, None
    if not isinstance(value, dict):
        errors.append(f"{path}: {role} must be a JSON object")
        return None, binding
    return value, binding


def _protocol_label(value: dict[str, Any]) -> str:
    version = value.get("protocolVersion")
    if version == PROTOCOL_VERSION:
        return PROTOCOL_VERSION
    return f"unsupported:{version!r}"


def _is_rfc3339_utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def _verify_v0_2_manifest_receipts(
    *,
    items: list[dict[str, Any]],
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        errors.append("manifest seed must be an integer")
    else:
        manifest_content = dict(manifest)
        manifest_content.pop("id", None)
        expected_id = manifest_artifact_id(manifest_content)
        if manifest.get("id") != expected_id:
            errors.append("manifest id does not match its seed and protocol")
    if manifest.get("createdAt") != MANIFEST_FIXED_CREATED_AT:
        errors.append("manifest createdAt does not match the frozen manifest timestamp")
    if manifest.get("notes") != manifest_notes():
        errors.append("manifest notes do not match the canonical honesty notes")
    expected_prompts, expected_gated = build_prompt_receipts(items)
    if manifest.get("prompts") != expected_prompts:
        errors.append(
            "manifest sendable prompt receipts do not match the canonical dataset and leakage gate"
        )
    if manifest.get("gatedPrompts") != expected_gated:
        errors.append(
            "manifest gated prompt receipts do not match the canonical dataset and leakage gate"
        )
    expected_counts = {
        "itemCount": len(items),
        "promptCount": len(expected_prompts) + len(expected_gated),
        "nonLeakingPromptCount": len(expected_prompts),
        "gatedPromptCount": len(expected_gated),
    }
    for field, expected in expected_counts.items():
        if manifest.get(field) != expected:
            errors.append(f"manifest {field} does not match canonical prompt receipts")

    hosted_max_retries = manifest.get("hostedMaxRetries")
    hosted_retry_delay_seconds = manifest.get("hostedRetryDelaySeconds")
    if (
        isinstance(hosted_retry_delay_seconds, bool)
        or not isinstance(hosted_retry_delay_seconds, (int, float))
        or not 0.0 <= hosted_retry_delay_seconds <= MAX_RETRY_DELAY_SECONDS
    ):
        errors.append(
            "manifest hostedRetryDelaySeconds is outside the supported retry bound"
        )
    if (
        isinstance(hosted_max_retries, bool)
        or not isinstance(hosted_max_retries, int)
        or not 0 <= hosted_max_retries <= MAX_HOSTED_RETRIES
    ):
        errors.append("manifest hostedMaxRetries is outside the supported retry bound")
    else:
        hosted_generations = len(expected_prompts) * sum(
            row.get("effectiveReruns", 0)
            for row in manifest.get("models", [])
            if row.get("evidenceMode") == "black_box"
        )
        if manifest.get("estimatedMaxHttpAttempts") != hosted_generations * (
            hosted_max_retries + 1
        ):
            errors.append(
                "manifest estimatedMaxHttpAttempts does not match its hosted retry policy"
            )

    for row in manifest.get("models", []):
        model = row.get("model")
        try:
            evidence_mode, effective_reruns = _expected_model_receipt(model)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if row.get("evidenceMode") != evidence_mode:
            errors.append(f"manifest model {model} has an invalid evidenceMode")
        if row.get("effectiveReruns") != effective_reruns:
            errors.append(f"manifest model {model} has an invalid effectiveReruns")
        adapter_identity = _verify_adapter_identity(
            identity=row.get("adapterIdentity"),
            model=model,
            evidence_mode=evidence_mode,
            context=f"manifest model {model}",
            errors=errors,
        )
        if evidence_mode == "black_box":
            if adapter_identity.get("maxRetries") != hosted_max_retries:
                errors.append(
                    f"manifest model {model} retry policy does not match hostedMaxRetries"
                )
            if (
                adapter_identity.get("retryDelaySeconds")
                != hosted_retry_delay_seconds
            ):
                errors.append(
                    f"manifest model {model} retry policy does not match hostedRetryDelaySeconds"
                )
        if row.get("status") != "ready":
            errors.append(
                f"manifest model {model} was not ready for the verified result run"
            )
        if row.get("missingEnv") != []:
            errors.append(
                f"manifest model {model} has unresolved environment requirements"
            )
        if "error" in row:
            errors.append(f"manifest model {model} has a planning error")


def _verify_v0_2_result_receipts(
    *,
    items: list[dict[str, Any]],
    result: dict[str, Any],
    errors: list[str],
) -> None:
    """Recompute item runs and aggregates from retained raw adapter strings."""

    try:
        validate_scoring_runtime()
    except RuntimeError as exc:
        errors.append(f"result receipt replay runtime incompatible: {exc}")
        return

    if result.get("evaluationScope") != "canonical":
        errors.append("verified v0.2 releases must have evaluationScope=canonical")
        return
    if result.get("requestedItemLimit") is not None:
        errors.append("canonical v0.2 releases must have requestedItemLimit=null")
    if result.get("evaluatedItemCount") != len(items):
        errors.append("result evaluatedItemCount does not match the canonical dataset")
    expected_metric_classes = sorted({item["metricClass"] for item in items})
    if result.get("metricClasses") != expected_metric_classes:
        errors.append("result metricClasses do not match the canonical dataset")
    runs_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for run in result.get("itemRuns", []):
        key = (run.get("model"), run.get("itemId"))
        if not all(isinstance(part, str) for part in key):
            continue
        if key in runs_by_pair:
            continue
        runs_by_pair[key] = run

    summaries_by_model = {
        summary.get("model"): summary
        for summary in result.get("models", [])
        if isinstance(summary.get("model"), str)
    }
    expected_modes: set[str] = set()
    for model in summaries_by_model:
        try:
            mode, _ = _expected_model_receipt(model)
        except ValueError:
            continue
        expected_modes.add(mode)
    if result.get("config", {}).get("evidenceModes") != sorted(expected_modes):
        errors.append("result config evidenceModes do not match model summaries")
    if result.get("notes") != result_notes(sorted(expected_modes)):
        errors.append("result notes do not match the declared evidence modes and protocol")
    seed = result.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        errors.append("result seed must be an integer for receipt replay")
        return
    result_content = dict(result)
    result_content.pop("id", None)
    expected_result_id = result_artifact_id(result_content)
    if result.get("id") != expected_result_id:
        errors.append("result id does not match its seed, scope, and protocol")
    created_at = result.get("createdAt")
    if not _is_rfc3339_utc(created_at):
        errors.append("result createdAt must be an RFC 3339 UTC timestamp")
        return
    if expected_modes == {"mock"} and created_at != RESULT_FIXED_CREATED_AT:
        errors.append("mock result createdAt does not match the frozen fixture timestamp")

    for model, observed_summary in summaries_by_model.items():
        try:
            evidence_mode, effective_reruns = _expected_model_receipt(model)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if observed_summary.get("evidenceMode") != evidence_mode:
            errors.append(f"result model {model} has an invalid evidenceMode")
            continue

        outputs: dict[tuple[str, str], list[str]] = {}
        receipts: dict[tuple[str, str], list[dict[str, str]]] = {}
        model_runs: list[dict[str, Any]] = []
        replayable = True
        for item in items:
            observed = runs_by_pair.get((model, item["id"]))
            if observed is None:
                replayable = False
                continue
            aleph_run = observed.get("alephRun", {})
            run_content = dict(aleph_run)
            run_content.pop("id", None)
            expected_run_id = aleph_run_artifact_id(
                item_id=item["id"], seed=seed, artifact=run_content
            )
            if aleph_run.get("id") != expected_run_id:
                errors.append(
                    f"item run {item['id']}/{model} AlephRun id does not match its identity"
                )
            run_config = aleph_run.get("config", {})
            budget = run_config.get("budget", {})
            candidates = aleph_run.get("candidates", [])
            candidate_tokens = [
                candidate.get("tokens")
                for candidate in candidates
                if isinstance(candidate, dict)
                and isinstance(candidate.get("tokens"), int)
                and not isinstance(candidate.get("tokens"), bool)
            ]
            if run_config.get("model") != model:
                errors.append(f"result {model}:{item['id']} AlephRun model drifted")
            if run_config.get("metric") != item["metricClass"]:
                errors.append(f"result {model}:{item['id']} AlephRun metric drifted")
            if run_config.get("decoding") != decoding_for_evidence_mode(evidence_mode):
                errors.append(f"result {model}:{item['id']} AlephRun decoding drifted")
            if budget.get("candidates") != len(candidates):
                errors.append(
                    f"result {model}:{item['id']} AlephRun candidate budget drifted"
                )
            expected_max_tokens = max(candidate_tokens) if candidate_tokens else 0
            if (
                len(candidate_tokens) != len(candidates)
                or budget.get("maxPromptTokens") != expected_max_tokens
            ):
                errors.append(
                    f"result {model}:{item['id']} AlephRun maxPromptTokens does not "
                    "bound its candidates"
                )
            if budget.get("repeatedSamples") != effective_reruns:
                errors.append(
                    f"result {model}:{item['id']} AlephRun repeatedSamples drifted"
                )
            measurements = {
                row.get("promptId"): row
                for row in observed.get("measurements", [])
                if isinstance(row, dict) and isinstance(row.get("promptId"), str)
            }
            for ladder in item["frozenLadder"]:
                measurement = measurements.get(ladder["id"])
                if measurement is None or not isinstance(measurement.get("outputs"), list):
                    errors.append(
                        f"result {model}:{item['id']} lacks raw outputs for {ladder['id']}"
                    )
                    replayable = False
                    continue
                raw_outputs = measurement["outputs"]
                raw_receipts = measurement.get("responseReceipts")
                expected_count = (
                    0
                    if ladder["expectedLeakage"] == "leaky_anchor"
                    else effective_reruns
                )
                if len(raw_outputs) != expected_count or any(
                    not isinstance(output, str) for output in raw_outputs
                ):
                    errors.append(
                        f"result {model}:{item['id']}:{ladder['id']} has an invalid raw-output receipt count"
                    )
                    replayable = False
                    continue
                expected_receipt_count = (
                    expected_count if evidence_mode == "black_box" else 0
                )
                if (
                    not isinstance(raw_receipts, list)
                    or len(raw_receipts) != expected_receipt_count
                    or any(
                        not isinstance(receipt, dict)
                        or not _is_rfc3339_utc(receipt.get("capturedAt"))
                        or receipt.get("source") not in {"provider", "cache"}
                        for receipt in raw_receipts
                    )
                ):
                    errors.append(
                        f"result {model}:{item['id']}:{ladder['id']} has an invalid response-capture receipt"
                    )
                    replayable = False
                    continue
                outputs[(item["id"], ladder["id"])] = raw_outputs
                receipts[(item["id"], ladder["id"])] = raw_receipts
        if not replayable:
            continue

        adapter = RetainedOutputReplayAdapter(
            model_id=model,
            observation_mode=evidence_mode,
            effective_reruns=effective_reruns,
            outputs=outputs,
            receipts=receipts,
        )
        try:
            for item in items:
                expected_run = evaluate_item(
                    item,
                    adapter,
                    seed=seed,
                    created_at=created_at,
                )
                observed_run = runs_by_pair[(model, item["id"])]
                if observed_run != expected_run:
                    errors.append(
                        f"result {model}:{item['id']} does not match canonical rescoring of raw outputs"
                    )
                model_runs.append(expected_run)
            expected_summary = summarize_model_runs(
                model=model,
                evidence_mode=evidence_mode,
                runs=model_runs,
                seed=seed,
                bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
            )
            adapter_identity = _verify_adapter_identity(
                identity=observed_summary.get("adapterIdentity"),
                model=model,
                evidence_mode=evidence_mode,
                context=f"result model {model}",
                errors=errors,
            )
            expected_summary["adapterIdentity"] = adapter_identity
            all_receipts = [
                receipt
                for values in receipts.values()
                for receipt in values
            ]
            response_capture: dict[str, Any] | None = None
            if evidence_mode == "black_box":
                captured_at = [receipt["capturedAt"] for receipt in all_receipts]
                response_capture = {
                    "responseCount": len(all_receipts),
                    "providerResponseCount": sum(
                        receipt["source"] == "provider" for receipt in all_receipts
                    ),
                    "cacheHitCount": sum(
                        receipt["source"] == "cache" for receipt in all_receipts
                    ),
                    "capturedAtMin": min(captured_at),
                    "capturedAtMax": max(captured_at),
                }
            expected_summary["responseCapture"] = response_capture
            if observed_summary != expected_summary:
                errors.append(
                    f"result model summary {model} does not match recomputed item runs"
                )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"result receipt replay failed for {model}: {exc}")


def verify_artifacts(
    *,
    data_dir: Path,
    result_path: Path,
    manifest_path: Path,
    audit_path: Path | None = None,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    items, dataset_digest, dataset_binding = _load_dataset(data_dir, errors)
    result, result_binding = _load_object(
        result_path,
        "BenchResult",
        errors,
        max_bytes=MAX_RESULT_BYTES,
    )
    manifest, manifest_binding = _load_object(
        manifest_path,
        "BenchManifest",
        errors,
        max_bytes=MAX_MANIFEST_BYTES,
    )

    labels = {
        _protocol_label(value)
        for value in [*items, result, manifest]
        if value is not None
    }
    unsupported = sorted(label for label in labels if label.startswith("unsupported:"))
    if unsupported:
        errors.append(f"unsupported protocol versions: {unsupported}")
    supported_labels = labels - set(unsupported)
    if len(supported_labels) > 1:
        errors.append(
            "dataset, result, and manifest mix incompatible protocol versions: "
            f"{sorted(supported_labels)}"
        )
    protocol_version = (
        next(iter(supported_labels)) if len(supported_labels) == 1 else "unknown"
    )
    schema_dir = REPO_ROOT / "schemas/v0.2"
    item_schema = load_schema(schema_dir / "aleph-bench-item.schema.json")
    result_schema = load_schema(schema_dir / "aleph-bench-result.schema.json")
    aleph_run_schema = result_schema["$defs"]["alephRun"]
    manifest_schema = load_schema(schema_dir / "aleph-bench-manifest.schema.json")

    if audit_path is not None:
        errors.append("v0.2 verification does not accept a legacy audit artifact")
    if bundle_path is not None:
        errors.append("v0.2 verification does not accept a legacy evidence bundle")

    items_valid = True
    if protocol_version == PROTOCOL_VERSION:
        if (
            len(items) != FROZEN_DATASET_ITEM_COUNT
            or dataset_digest != FROZEN_DATASET_SHA256
        ):
            items_valid = False
            errors.append(
                f"{data_dir}: canonical v0.2 dataset check failed: expected "
                f"{FROZEN_DATASET_ITEM_COUNT} items/{FROZEN_DATASET_SHA256}, "
                f"found {len(items)} items/{dataset_digest or '<unavailable>'}"
            )
    else:
        items_valid = False
    for index, item in enumerate(items):
        try:
            validate(item, item_schema)
        except SchemaValidationError as exc:
            items_valid = False
            errors.append(f"{data_dir} item {index}: {exc}")
    relational_items = items if items_valid else []
    result_valid = result is not None
    if result is not None:
        try:
            validate(result, result_schema)
        except SchemaValidationError as exc:
            result_valid = False
            errors.append(f"{result_path}: {exc}")
    manifest_valid = manifest is not None
    if manifest is not None:
        try:
            validate(manifest, manifest_schema)
        except SchemaValidationError as exc:
            manifest_valid = False
            errors.append(f"{manifest_path}: {exc}")

    expected_dataset_path = stable_dataset_path(data_dir)
    item_ids = {item["id"] for item in relational_items}
    canaries = {item["canaryGuid"] for item in relational_items}

    if result_valid:
        aleph_run_contract_count = 0
        for run in result.get("itemRuns", []):
            if not isinstance(run, dict):
                errors.append("unknown:unknown alephRun: item run is not an object")
                continue
            try:
                validate(run["alephRun"], aleph_run_schema, root=result_schema)
                aleph_run_contract_count += 1
            except (KeyError, TypeError, SchemaValidationError) as exc:
                errors.append(f"{run.get('itemId', '<unknown>')}:{run.get('model', '<unknown>')} alephRun: {exc}")
        if result.get("config", {}).get("datasetPath") != expected_dataset_path:
            errors.append("result config datasetPath does not match data directory")
        if len(canaries) == 1 and result.get("canaryGuid") != next(iter(canaries)):
            errors.append("result canaryGuid does not match dataset canary")
        result_item_ids = {run["itemId"] for run in result.get("itemRuns", []) if "itemId" in run}
        unknown = sorted(result_item_ids - item_ids)
        if unknown:
            errors.append(f"result references unknown item ids: {unknown}")
        model_item_total = sum(model["itemCount"] for model in result.get("models", []) if "itemCount" in model)
        if len(result.get("itemRuns", [])) != model_item_total:
            errors.append("result itemRuns length does not match model itemCount total")
        result_models = [model["model"] for model in result["models"]]
        if len(set(result_models)) != len(result_models):
            errors.append("result model ids must be unique")
        expected_pairs = {(model, item_id) for model in result_models for item_id in item_ids}
        observed_pairs = [(run["model"], run["itemId"]) for run in result["itemRuns"]]
        if len(set(observed_pairs)) != len(observed_pairs):
            errors.append("result contains duplicate model/item runs")
        if set(observed_pairs) != expected_pairs:
            errors.append("result does not cover the complete model/item Cartesian product")
        for model in result["models"]:
            if model["itemCount"] != len(item_ids):
                errors.append(
                    f"result model {model['model']} itemCount does not match dataset"
                )
        prompts_by_item = {
            item["id"]: {prompt["id"] for prompt in item["frozenLadder"]}
            for item in relational_items
        }
        for run in result["itemRuns"]:
            prompt_ids = [row["promptId"] for row in run["measurements"]]
            if len(set(prompt_ids)) != len(prompt_ids):
                errors.append(
                    f"result {run['model']}:{run['itemId']} has duplicate measurements"
                )
            if set(prompt_ids) != prompts_by_item.get(run["itemId"], set()):
                errors.append(
                    f"result {run['model']}:{run['itemId']} does not cover its frozen ladder"
                )
        if protocol_version == PROTOCOL_VERSION and items_valid:
            expected_identity = {
                "datasetId": FROZEN_DATASET_ID,
                "datasetItemCount": FROZEN_DATASET_ITEM_COUNT,
                "datasetSha256": FROZEN_DATASET_SHA256,
                "datasetHashAlgorithm": FROZEN_DATASET_HASH_ALGORITHM,
            }
            for field, expected in expected_identity.items():
                if result.get("config", {}).get(field) != expected:
                    errors.append(f"result config {field} does not match canonical dataset")
            _verify_v0_2_result_receipts(
                items=items,
                result=result,
                errors=errors,
            )
    else:
        aleph_run_contract_count = 0

    if manifest_valid:
        if manifest["itemCount"] != len(item_ids):
            errors.append("manifest itemCount does not match dataset")
        if manifest["modelCount"] != len(manifest["models"]):
            errors.append("manifest modelCount does not match models length")
        manifest_model_ids = [model["model"] for model in manifest["models"]]
        if len(set(manifest_model_ids)) != len(manifest_model_ids):
            errors.append("manifest model ids must be unique")
        if manifest["datasetPath"] != expected_dataset_path:
            errors.append("manifest datasetPath does not match data directory")
        if manifest["promptCount"] != manifest["nonLeakingPromptCount"] + manifest["gatedPromptCount"]:
            errors.append("manifest prompt counts do not add up")
        effective_reruns = sum(
            model["effectiveReruns"] for model in manifest["models"]
        )
        expected_generations = manifest["nonLeakingPromptCount"] * effective_reruns
        if manifest["estimatedGenerations"] != expected_generations:
            errors.append("manifest estimatedGenerations does not match prompt/model/rerun counts")
        if any(prompt["leakageGate"]["disqualified"] for prompt in manifest["prompts"]):
            errors.append("manifest sendable prompts include a disqualified prompt")
        if any(not prompt["leakageGate"]["disqualified"] for prompt in manifest["gatedPrompts"]):
            errors.append("manifest gated prompts include a non-disqualified prompt")
        manifest_item_ids = {prompt["itemId"] for prompt in manifest["prompts"] + manifest["gatedPrompts"]}
        unknown = sorted(manifest_item_ids - item_ids)
        if unknown:
            errors.append(f"manifest references unknown item ids: {unknown}")
        expected_prompt_pairs = {
            (item["id"], prompt["id"])
            for item in relational_items
            for prompt in item["frozenLadder"]
        }
        manifest_prompt_pairs = [
            (prompt["itemId"], prompt["promptId"])
            for prompt in manifest["prompts"] + manifest["gatedPrompts"]
        ]
        if len(set(manifest_prompt_pairs)) != len(manifest_prompt_pairs):
            errors.append("manifest contains duplicate item/prompt entries")
        if set(manifest_prompt_pairs) != expected_prompt_pairs:
            errors.append("manifest does not cover every frozen-ladder prompt exactly once")
        if manifest["promptCount"] != len(expected_prompt_pairs):
            errors.append("manifest promptCount does not match dataset ladders")
        if protocol_version == PROTOCOL_VERSION and items_valid:
            expected_identity = {
                "datasetId": FROZEN_DATASET_ID,
                "datasetItemCount": FROZEN_DATASET_ITEM_COUNT,
                "datasetSha256": FROZEN_DATASET_SHA256,
                "datasetHashAlgorithm": FROZEN_DATASET_HASH_ALGORITHM,
            }
            for field, expected in expected_identity.items():
                if manifest.get(field) != expected:
                    errors.append(f"manifest {field} does not match canonical dataset")
            _verify_v0_2_manifest_receipts(
                items=items,
                manifest=manifest,
                errors=errors,
            )

    if result_valid and manifest_valid:
        result_models = sorted(model["model"] for model in result["models"])
        manifest_models = sorted(model["model"] for model in manifest["models"])
        if result_models != manifest_models:
            errors.append("result models do not match manifest models")
        if result["track"] != manifest["track"] or result["split"] != manifest["split"] or result["seed"] != manifest["seed"]:
            errors.append("result track/split/seed do not match manifest")
        if result["config"].get("scoring") != manifest.get("scoring"):
            errors.append("result scoring profile does not match manifest")
        result_identities = {
            row["model"]: row.get("adapterIdentity") for row in result["models"]
        }
        manifest_identities = {
            row["model"]: row.get("adapterIdentity") for row in manifest["models"]
        }
        if result_identities != manifest_identities:
            errors.append("result adapter identities do not match manifest")

    # Bind the report to the exact byte snapshots used above. Rechecking the
    # path/inode metadata at the end rejects scan-to-use replacement; the
    # returned digests remain the durable identity if a path changes later.
    if dataset_binding is not None:
        _recheck_dataset_binding(dataset_binding, errors)
    if result_binding is not None:
        _recheck_file_binding(result_binding, errors)
    if manifest_binding is not None:
        _recheck_file_binding(manifest_binding, errors)

    return {
        "status": "ok" if not errors else "failed",
        "protocolVersion": protocol_version,
        "dataDir": expected_dataset_path,
        "datasetSha256": dataset_digest,
        "resultPath": stable_dataset_path(result_path),
        "resultSha256": (
            result_binding["sha256"] if result_binding is not None else None
        ),
        "manifestPath": stable_dataset_path(manifest_path),
        "manifestSha256": (
            manifest_binding["sha256"] if manifest_binding is not None else None
        ),
        "auditPath": stable_dataset_path(audit_path) if audit_path is not None else None,
        "bundlePath": stable_dataset_path(bundle_path) if bundle_path is not None else None,
        "itemCount": len(items),
        "resultItemRuns": (
            len(result["itemRuns"])
            if result is not None and isinstance(result.get("itemRuns"), list)
            else 0
        ),
        "resultAlephRunContractCount": aleph_run_contract_count,
        "resultModelCount": (
            len(result["models"])
            if result is not None and isinstance(result.get("models"), list)
            else 0
        ),
        "manifestPromptCount": manifest.get("promptCount", 0) if manifest is not None else 0,
        "manifestNonLeakingPromptCount": manifest.get("nonLeakingPromptCount", 0) if manifest is not None else 0,
        "manifestGatedPromptCount": manifest.get("gatedPromptCount", 0) if manifest is not None else 0,
        "manifestEstimatedGenerations": manifest.get("estimatedGenerations", 0) if manifest is not None else 0,
        "manifestHostedMaxRetries": manifest.get("hostedMaxRetries")
        if manifest is not None
        else None,
        "manifestEstimatedMaxHttpAttempts": manifest.get(
            "estimatedMaxHttpAttempts", 0
        )
        if manifest is not None
        else 0,
        "auditCheckCount": 0,
        "bundleArtifactCount": 0,
        "errors": errors,
    }


def format_verify_report(report: dict[str, Any]) -> str:
    lines = [
        f"status: {report['status']}",
        f"protocol: {report['protocolVersion']}",
        (
            f"dataset: {report['dataDir']} ({report['itemCount']} items, "
            f"sha256={report['datasetSha256']})"
        ),
        (
            f"result: {report['resultPath']} "
            f"({report['resultModelCount']} models, {report['resultItemRuns']} item runs, "
            f"{report['resultAlephRunContractCount']} canonical AlephRun, "
            f"sha256={report['resultSha256']})"
        ),
        (
            f"manifest: {report['manifestPath']} "
            f"({report['manifestNonLeakingPromptCount']} non-leaking, "
            f"{report['manifestGatedPromptCount']} gated, "
            f"{report['manifestEstimatedGenerations']} logical generations, "
            f"{report['manifestEstimatedMaxHttpAttempts']} max HTTP attempts, "
            f"sha256={report['manifestSha256']})"
        ),
    ]
    if report["auditPath"] is not None:
        lines.append(f"audit: {report['auditPath']} ({report['auditCheckCount']} checks)")
    if report["bundlePath"] is not None:
        lines.append(f"bundle: {report['bundlePath']} ({report['bundleArtifactCount']} artifacts)")
    for error in report["errors"]:
        lines.append(f"error: {error}")
    return "\n".join(lines)
