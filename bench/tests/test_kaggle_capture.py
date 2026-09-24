from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from bench.engine.kaggle_capture import (
    CAPTURE_SCHEMA_PATH,
    FROZEN_DATASET_HASH_ALGORITHM,
    MAX_CAPTURE_BYTES,
    MAX_RAW_OUTPUT_CODEPOINTS,
    MAX_SCORING_TEXT_CODEPOINTS,
    KaggleCaptureError,
    artifact_id_for,
    canonical_string_sha256,
    coverage_sha256,
    expected_row_id,
    load_capture_payload,
    parse_capture_payload,
    prompt_utf8_sha256,
    serialize_capture_payload,
    verify_capture_payload,
    _verify_usage,
)
from bench.engine.schema_validation import SchemaValidationError, load_schema, validate


RESULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas/v0.2/aleph-bench-result.schema.json"
)

# These expectations are deliberately literal rather than computed by a helper
# that mirrors production logic. They are compact independent contract vectors.
ACTIVE_STATE_VECTORS = {
    "prepared": {"attempted": 1, "terminal": 1, "blockers": ["incompleteCoverage", "activeCall"]},
    "dispatching": {"attempted": 2, "terminal": 1, "blockers": ["incompleteCoverage", "activeCall"]},
    "terminalPersisted": {"attempted": 1, "terminal": 1, "blockers": ["incompleteCoverage"]},
}


def _call(index: int, *, prompt: str | None = None) -> dict[str, Any]:
    item_id = f"item-{index + 1}"
    prompt_id = f"prompt-{index + 1}"
    prompt_text = prompt if prompt is not None else f"Exact prompt {index + 1}\r\n"
    rerun_index = index % 5
    return {
        "rowId": expected_row_id(item_id, prompt_id, rerun_index),
        "itemId": item_id,
        "promptId": prompt_id,
        "rerunIndex": rerun_index,
        "conversationName": f"aleph-capture-item-{index + 1}-rerun-{rerun_index}",
        "promptText": prompt_text,
        "promptUtf8Sha256": prompt_utf8_sha256(prompt_text),
        "promptCodePointCount": len(prompt_text),
    }


def _usage(
    *,
    input_tokens: int | None = 10,
    output_tokens: int | None = 20,
    finish_reason: str | None = "stop",
) -> dict[str, Any]:
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "inputCostNanodollars": "100",
        "outputCostNanodollars": "200",
        "backendLatencyMs": 25,
        "finishReason": finish_reason,
    }


def _failure(kind: str) -> dict[str, Any]:
    return {
        "failureKind": kind,
        "exceptionType": "builtins.RuntimeError",
        "messageStringSha256": canonical_string_sha256(
            "sensitive backend detail is hashed, never retained"
        ),
    }


def _row_base(
    call: dict[str, Any], *, dispatch_started: bool = True, second: int = 1
) -> dict[str, Any]:
    return {
        **copy.deepcopy(call),
        "dispatchStarted": dispatch_started,
        "terminalFailure": None,
        "lifecycleFailure": None,
        "usage": _usage(),
        "capturedAt": f"2026-09-18T00:00:{second:02d}Z",
    }


def _returned_row(
    call: dict[str, Any], raw_output: str = "answer", *, second: int = 1
) -> dict[str, Any]:
    return {
        **_row_base(call, second=second),
        "outcome": "returnedString",
        "observedOutputType": "string",
        "rawOutput": raw_output,
        "rawOutputStringSha256": canonical_string_sha256(raw_output),
        "rawOutputCodePointCount": len(raw_output),
    }


def _failed_row(call: dict[str, Any], kind: str) -> dict[str, Any]:
    dispatch_started = kind == "modelCallFailed"
    row = {
        **_row_base(call, dispatch_started=dispatch_started),
        "outcome": "failed",
        "observedOutputType": None,
        "rawOutputStringSha256": None,
        "rawOutputCodePointCount": None,
        "terminalFailure": _failure(kind),
    }
    row["usage"] = _usage(
        input_tokens=None, output_tokens=None, finish_reason=None
    )
    row["usage"]["inputCostNanodollars"] = None
    row["usage"]["outputCostNanodollars"] = None
    return row


def _active(
    call: dict[str, Any], state: str, *, updated_at: str = "2026-09-18T00:00:03Z"
) -> dict[str, Any]:
    return {
        "call": copy.deepcopy(call),
        "state": state,
        "updatedAt": updated_at,
    }


def _diagnostics(
    *,
    coverage_complete: bool = True,
    missing: list[str] | None = None,
    unpaired: list[str] | None = None,
    missing_output: list[str] | None = None,
    non_string: list[str] | None = None,
    oversized: list[str] | None = None,
    terminal_failure: list[str] | None = None,
    lifecycle_failure: list[str] | None = None,
    scoring_too_long: list[str] | None = None,
    missing_usage: list[str] | None = None,
    missing_finish_reason: list[str] | None = None,
    zero_usage: list[str] | None = None,
    near_cap: list[str] | None = None,
    token_limit: list[str] | None = None,
    unsupported_finish: list[str] | None = None,
    active_row_id: str | None = None,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "coverageComplete": coverage_complete,
        "missingPlannedRowIds": missing or [],
        "unpairedSurrogateRowIds": unpaired or [],
        "missingOutputRowIds": missing_output or [],
        "nonStringOutputRowIds": non_string or [],
        "oversizedOutputRowIds": oversized or [],
        "terminalFailureRowIds": terminal_failure or [],
        "lifecycleFailureRowIds": lifecycle_failure or [],
        "scoringTextTooLongRowIds": scoring_too_long or [],
        "missingUsageRowIds": missing_usage or [],
        "missingFinishReasonRowIds": missing_finish_reason or [],
        "zeroUsageRowIds": zero_usage or [],
        "nearCapOutputRowIds": near_cap or [],
        "tokenLimitTerminationRowIds": token_limit or [],
        "unsupportedFinishReasonRowIds": unsupported_finish or [],
        "activeCallRowId": active_row_id,
        "replayBlockedReasons": blockers or [],
    }


def _payload(
    *,
    planned: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
    active: dict[str, Any] | None = None,
    attempted: int = 1,
    terminal: int = 1,
    returned: int = 1,
    missing_output_count: int = 0,
    non_string_count: int = 0,
    oversized_count: int = 0,
    failed_count: int = 0,
    diagnostics: dict[str, Any] | None = None,
    capture_complete: bool = True,
    replay_eligible: bool = True,
) -> dict[str, Any]:
    calls = copy.deepcopy(planned if planned is not None else [_call(0)])
    captured_rows = copy.deepcopy(
        rows if rows is not None else [_returned_row(calls[0])]
    )
    full_plan_sha = "b" * 64
    payload: dict[str, Any] = {
        "captureSchemaVersion": "1.1.0",
        "artifactKind": "aleph_bench_kaggle_raw_capture",
        "targetProtocolVersion": "0.2.0",
        "leaderboardEligible": False,
        "publicationEligible": False,
        "captureComplete": capture_complete,
        "canonicalReplayEligible": replay_eligible,
        "startedAt": "2026-09-18T00:00:00Z",
        "endedAt": "2026-09-18T00:10:00Z",
        "datasetIdentity": {
            "id": "aleph-bench-v0.2-public-s2",
            "itemCount": 30,
            "sha256": "6f3a03400ec16405414afb94c7c639f2df07f7f0797c3b58ad1c4229e52f2041",
            "hashAlgorithm": FROZEN_DATASET_HASH_ALGORITHM,
        },
        "packageIdentity": {
            "id": "aleph-bench-v0.2-package",
            "manifestPath": "bench/package/v0.2/manifest.json",
            "sha256": "a" * 64,
            "bytes": 1234,
        },
        "callPlanIdentity": {
            "id": "aleph-bench-v0.2-call-plan",
            "sha256": "c" * 64,
            "fullShardPlanSha256": full_plan_sha,
            "totalPlannedCallCount": len(calls),
            "rerunsPerPrompt": 5,
        },
        "taskIdentity": {
            "name": "aleph-bench-v0.2-capture",
            "version": "1.0.0",
            "sourcePath": "bench/tasks/kaggle/capture_v0_2.py",
            "definitionSha256": "d" * 64,
            "implementationSha256": "e" * 64,
        },
        "modelObservation": {
            "platform": "kaggle",
            "slug": "operator/model",
            "pythonType": "kaggle_benchmarks.actors.llms.OpenAI",
            "mappingStatus": "unmapped",
            "canonicalModelId": None,
            "providerRevision": None,
        },
        "runtimeObservation": {
            "pythonVersion": "3.11",
            "pythonFullVersion": "3.11.9",
            "unicodeDatabaseVersion": "14.0.0",
            "platform": "Linux-x86_64",
        },
        "requestPolicy": {
            "conversationIsolation": "oneNamedChatPerPromptAndRerun",
            "transportRetries": 0,
            "maxAttemptsPerCall": 1,
            "temperature": 0,
            "maxOutputTokens": 512,
            "nearCapMarginTokens": 32,
            "seed": None,
            "reasoning": None,
        },
        "shard": {
            "id": "capture-canary-01",
            "fullShardPlanSha256": full_plan_sha,
            "coverageSha256": coverage_sha256("capture-canary-01", calls),
            "plannedCalls": calls,
        },
        "calls": {
            "plannedCallCount": len(calls),
            "attemptedCallCount": attempted,
            "terminalCallCount": terminal,
            "returnedStringCount": returned,
            "missingOutputCount": missing_output_count,
            "nonStringOutputCount": non_string_count,
            "oversizedOutputCount": oversized_count,
            "failedCallCount": failed_count,
            "activeCall": copy.deepcopy(active),
        },
        "rows": captured_rows,
        "diagnostics": copy.deepcopy(diagnostics or _diagnostics()),
        "id": "",
    }
    payload["id"] = artifact_id_for(payload)
    return payload


def _resign(payload: dict[str, Any]) -> None:
    payload["id"] = artifact_id_for(payload)


class KaggleCaptureContractTests(unittest.TestCase):
    def test_hard_coded_hash_vectors_separate_prompt_and_output_encodings(self) -> None:
        prompt = "Exact prompt 1\r\n"
        self.assertEqual(
            prompt_utf8_sha256(prompt),
            "cfbd12ce07463e0ab678389c84738bd8a91bb7f5a084079c1c9d184e2476f884",
        )
        self.assertEqual(
            canonical_string_sha256("answer"),
            "b2437a6aa53010b70011f253d0ef2e75d5a15350811531d9b82c27e77a706081",
        )
        self.assertNotEqual(
            canonical_string_sha256("answer"),
            "0db52f4076c082518412afd3dd3576e2cb0c63703fd7fed5e23ade60efef31d9",
        )

    def test_valid_capture_round_trip_is_score_free_and_immutable(self) -> None:
        payload = _payload()
        encoded = serialize_capture_payload(payload)
        decoded = parse_capture_payload(encoded)
        self.assertEqual(decoded, payload)
        self.assertTrue(decoded["captureComplete"])
        self.assertTrue(decoded["canonicalReplayEligible"])
        self.assertFalse(decoded["leaderboardEligible"])
        self.assertFalse(decoded["publicationEligible"])

    def test_empty_string_is_complete_and_replay_eligible(self) -> None:
        call = _call(0)
        payload = _payload(rows=[_returned_row(call, "")])
        self.assertEqual(verify_capture_payload(payload), payload)
        self.assertTrue(payload["captureComplete"])
        self.assertTrue(payload["canonicalReplayEligible"])
        self.assertEqual(payload["diagnostics"]["replayBlockedReasons"], [])

    def test_raw_string_hash_vectors_preserve_exact_text_semantics(self) -> None:
        vectors = [
            (
                "  leading and trailing\t ",
                "84becd011eca1f03c46d57f75b41a9eaf2737c3cd63a3abb35e56851f87da03b",
            ),
            (
                "line one\nline two",
                "fe7ce060245a7cf48259f349043911673196ca36b3202195ee79d85bac454fc0",
            ),
            (
                "line one\rline two",
                "33c92b81dfe6ddc4139f3cb689e35658412e5637b33ae2e6f873c54a53390ee4",
            ),
            (
                "line one\r\nline two",
                "4f5ca12a849dfb190fd19a9d851d82014dc35b77c49ba8b7f16d7ccb9bad2a40",
            ),
            ("é", "abc3932c9ec58f042750738c49034e7ee67cf556d6be86832ac3dda38decb703"),
            ("e\u0301", "8e0c35d689e13926898b948c3be0a6ec7277b6840a3a782d487921cf052d6008"),
            (
                "Straße STRASSE",
                "4d42d108814df92391f215f8d1fa72f6b7cb59151ac8d2e09c18d149638051cb",
            ),
            (
                "👩\u200d💻",
                "1f6fb468ef2804596d1c933fc62c27c9477cd226830e01e296be3803385303ab",
            ),
            (
                "a\0b\x1f",
                "16eb98eb09a4e6da668b0e787cc8702bb6c3ecbd3736f52f1a45bed44608416f",
            ),
        ]
        call = _call(0)
        for raw, expected_digest in vectors:
            with self.subTest(raw=repr(raw)):
                self.assertEqual(canonical_string_sha256(raw), expected_digest)
                decoded = parse_capture_payload(
                    serialize_capture_payload(
                        _payload(rows=[_returned_row(call, raw)])
                    )
                )
                self.assertEqual(decoded["rows"][0]["rawOutput"], raw)
                self.assertEqual(
                    decoded["rows"][0]["rawOutputStringSha256"], expected_digest
                )

    def test_lone_surrogate_is_retained_exactly_but_blocks_replay(self) -> None:
        call = _call(0)
        raw = "prefix\ud800suffix"
        payload = _payload(
            rows=[_returned_row(call, raw)],
            diagnostics=_diagnostics(
                unpaired=[call["rowId"]], blockers=["unpairedSurrogate"]
            ),
            replay_eligible=False,
        )
        decoded = parse_capture_payload(serialize_capture_payload(payload))
        self.assertEqual(decoded["rows"][0]["rawOutput"], raw)
        self.assertTrue(decoded["captureComplete"])
        self.assertFalse(decoded["canonicalReplayEligible"])

    def test_adjacent_surrogate_pair_is_rejected_before_serialization(self) -> None:
        pair = "\ud83d\ude00"
        with self.assertRaisesRegex(KaggleCaptureError, "adjacent surrogate pair"):
            canonical_string_sha256(pair)

        payload = _payload()
        payload["rows"][0]["rawOutput"] = pair
        payload["rows"][0]["rawOutputCodePointCount"] = 2
        payload["rows"][0]["rawOutputStringSha256"] = "0" * 64
        with self.assertRaisesRegex(KaggleCaptureError, "adjacent surrogate pair"):
            verify_capture_payload(payload)

        nested = _payload()
        nested["runtimeObservation"]["platform"] = "Linux-" + pair
        with self.assertRaisesRegex(KaggleCaptureError, "adjacent surrogate pair"):
            verify_capture_payload(nested)

    def test_prompt_rejects_every_surrogate_and_uses_strict_utf8_digest(self) -> None:
        payload = _payload()
        for target in (payload["shard"]["plannedCalls"][0], payload["rows"][0]):
            target["promptText"] = "unsafe\ud800prompt"
            target["promptUtf8Sha256"] = "0" * 64
            target["promptCodePointCount"] = 13
        with self.assertRaisesRegex(KaggleCaptureError, "non-Unicode-scalar"):
            verify_capture_payload(payload)
        with self.assertRaisesRegex(KaggleCaptureError, "surrogate"):
            prompt_utf8_sha256("\ud83d\ude00")

    def test_metadata_rejects_lone_surrogates_while_raw_output_may_retain_one(self) -> None:
        payload = _payload()
        payload["runtimeObservation"]["platform"] = "Linux\ud800metadata"
        with self.assertRaisesRegex(KaggleCaptureError, "non-Unicode-scalar"):
            verify_capture_payload(payload)

        call = _call(0)
        raw = "diagnostic\ud800output"
        accepted = _payload(
            rows=[_returned_row(call, raw)],
            diagnostics=_diagnostics(
                unpaired=[call["rowId"]], blockers=["unpairedSurrogate"]
            ),
            replay_eligible=False,
        )
        self.assertEqual(verify_capture_payload(accepted), accepted)

    def test_cycles_fail_closed_but_shared_aliases_remain_valid(self) -> None:
        cyclic_dict = _payload()
        cyclic_dict["runtimeObservation"]["cycle"] = cyclic_dict["runtimeObservation"]
        with self.assertRaisesRegex(KaggleCaptureError, "cyclic object graph"):
            verify_capture_payload(cyclic_dict)

        cyclic_list = _payload()
        cyclic_list["rows"].append(cyclic_list["rows"])
        with self.assertRaisesRegex(KaggleCaptureError, "cyclic object graph"):
            verify_capture_payload(cyclic_list)

        calls = [_call(0), _call(1)]
        rows = [_returned_row(calls[0]), _returned_row(calls[1], second=2)]
        rows[1]["usage"] = rows[0]["usage"]
        shared = _payload(
            planned=calls,
            rows=rows,
            attempted=2,
            terminal=2,
            returned=2,
        )
        self.assertIs(shared["rows"][0]["usage"], shared["rows"][1]["usage"])
        self.assertEqual(verify_capture_payload(shared), shared)
        self.assertEqual(parse_capture_payload(serialize_capture_payload(shared)), shared)

    def test_active_state_vectors_are_independent_and_do_not_double_count(self) -> None:
        planned = [_call(0), _call(1)]
        rows = [_returned_row(planned[0])]
        missing = [planned[1]["rowId"]]
        for state, expected in ACTIVE_STATE_VECTORS.items():
            with self.subTest(state=state):
                marker_call = planned[0] if state == "terminalPersisted" else planned[1]
                payload = _payload(
                    planned=planned,
                    rows=rows,
                    active=_active(marker_call, state),
                    attempted=expected["attempted"],
                    terminal=expected["terminal"],
                    diagnostics=_diagnostics(
                        coverage_complete=False,
                        missing=missing,
                        active_row_id=marker_call["rowId"],
                        blockers=expected["blockers"],
                    ),
                    capture_complete=False,
                    replay_eligible=False,
                )
                self.assertEqual(verify_capture_payload(payload), payload)
                self.assertEqual(payload["calls"]["attemptedCallCount"], expected["attempted"])
                self.assertEqual(payload["calls"]["terminalCallCount"], expected["terminal"])

        persisted = _payload(
            active=_active(planned[0], "terminalPersisted"),
            diagnostics=_diagnostics(active_row_id=planned[0]["rowId"]),
        )
        self.assertEqual(verify_capture_payload(persisted), persisted)
        self.assertEqual(persisted["calls"]["attemptedCallCount"], 1)
        self.assertEqual(persisted["calls"]["terminalCallCount"], 1)
        self.assertTrue(persisted["canonicalReplayEligible"])

    def test_prepared_and_dispatching_must_point_to_next_call(self) -> None:
        planned = [_call(0), _call(1)]
        row = _returned_row(planned[0])
        payload = _payload(
            planned=planned,
            rows=[row],
            active=_active(planned[0], "dispatching"),
            attempted=2,
            terminal=1,
            diagnostics=_diagnostics(
                coverage_complete=False,
                missing=[planned[1]["rowId"]],
                active_row_id=planned[0]["rowId"],
                blockers=["incompleteCoverage", "activeCall"],
            ),
            capture_complete=False,
            replay_eligible=False,
        )
        with self.assertRaisesRegex(KaggleCaptureError, "next planned call"):
            verify_capture_payload(payload)

    def test_terminal_persisted_requires_last_persisted_row(self) -> None:
        planned = [_call(0), _call(1)]
        rows = [_returned_row(planned[0]), _returned_row(planned[1], second=2)]
        payload = _payload(
            planned=planned,
            rows=rows,
            active=_active(planned[0], "terminalPersisted"),
            attempted=2,
            terminal=2,
            returned=2,
            diagnostics=_diagnostics(active_row_id=planned[0]["rowId"]),
        )
        with self.assertRaisesRegex(KaggleCaptureError, "last persisted row"):
            verify_capture_payload(payload)

    def test_active_timestamp_cannot_precede_last_persisted_row(self) -> None:
        call = _call(0)
        payload = _payload(
            rows=[_returned_row(call, second=5)],
            active=_active(
                call, "terminalPersisted", updated_at="2026-09-18T00:00:04Z"
            ),
            diagnostics=_diagnostics(active_row_id=call["rowId"]),
        )
        with self.assertRaisesRegex(KaggleCaptureError, "precedes the last persisted"):
            verify_capture_payload(payload)

    def test_chat_open_failure_is_complete_without_a_dispatched_call(self) -> None:
        call = _call(0)
        row = _failed_row(call, "chatOpenFailed")
        payload = _payload(
            rows=[row],
            attempted=0,
            terminal=0,
            returned=0,
            failed_count=1,
            diagnostics=_diagnostics(
                terminal_failure=[call["rowId"]],
                missing_usage=[call["rowId"]],
                missing_finish_reason=[call["rowId"]],
                blockers=["terminalFailure", "missingUsage"],
            ),
            replay_eligible=False,
        )
        self.assertEqual(verify_capture_payload(payload), payload)
        self.assertTrue(payload["captureComplete"])
        self.assertFalse(payload["canonicalReplayEligible"])
        self.assertFalse(row["dispatchStarted"])

    def test_model_call_failure_is_dispatched_and_terminal(self) -> None:
        call = _call(0)
        row = _failed_row(call, "modelCallFailed")
        payload = _payload(
            rows=[row],
            returned=0,
            failed_count=1,
            diagnostics=_diagnostics(
                terminal_failure=[call["rowId"]],
                missing_usage=[call["rowId"]],
                missing_finish_reason=[call["rowId"]],
                blockers=["terminalFailure", "missingUsage"],
            ),
            replay_eligible=False,
        )
        self.assertEqual(verify_capture_payload(payload), payload)
        self.assertEqual(payload["calls"]["attemptedCallCount"], 1)
        self.assertEqual(payload["calls"]["terminalCallCount"], 1)

    def test_chat_close_failure_can_coexist_with_retained_raw_output(self) -> None:
        call = _call(0)
        row = _returned_row(call, "raw answer survives close failure")
        row["lifecycleFailure"] = _failure("chatCloseFailed")
        payload = _payload(
            rows=[row],
            diagnostics=_diagnostics(
                lifecycle_failure=[call["rowId"]], blockers=["lifecycleFailure"]
            ),
            replay_eligible=False,
        )
        self.assertEqual(verify_capture_payload(payload), payload)
        self.assertEqual(payload["rows"][0]["rawOutput"], "raw answer survives close failure")
        self.assertTrue(payload["captureComplete"])

    def test_generic_capture_failure_and_raw_error_message_are_forbidden(self) -> None:
        call = _call(0)
        payload = _payload(
            rows=[_failed_row(call, "modelCallFailed")],
            returned=0,
            failed_count=1,
            diagnostics=_diagnostics(
                terminal_failure=[call["rowId"]],
                missing_usage=[call["rowId"]],
                missing_finish_reason=[call["rowId"]],
                blockers=["terminalFailure", "missingUsage"],
            ),
            replay_eligible=False,
        )
        payload["rows"][0]["terminalFailure"]["failureKind"] = "captureFailed"
        _resign(payload)
        with self.assertRaisesRegex(KaggleCaptureError, "schema validation"):
            verify_capture_payload(payload)

        leaked = _payload()
        leaked["rows"][0]["terminalFailure"] = {
            **_failure("modelCallFailed"),
            "message": "backend secret",
        }
        _resign(leaked)
        with self.assertRaisesRegex(KaggleCaptureError, "schema validation"):
            verify_capture_payload(leaked)

    def test_16385_codepoints_are_retained_but_block_scoring_replay(self) -> None:
        call = _call(0)
        raw = "x" * (MAX_SCORING_TEXT_CODEPOINTS + 1)
        payload = _payload(
            rows=[_returned_row(call, raw)],
            diagnostics=_diagnostics(
                scoring_too_long=[call["rowId"]], blockers=["scoringTextTooLong"]
            ),
            replay_eligible=False,
        )
        self.assertEqual(verify_capture_payload(payload), payload)
        self.assertEqual(payload["rows"][0]["rawOutput"], raw)
        self.assertEqual(payload["rows"][0]["outcome"], "returnedString")

    def test_retention_boundary_and_oversized_observation_are_distinct(self) -> None:
        call = _call(0)
        retained = "x" * MAX_RAW_OUTPUT_CODEPOINTS
        payload = _payload(
            rows=[_returned_row(call, retained)],
            diagnostics=_diagnostics(
                scoring_too_long=[call["rowId"]], blockers=["scoringTextTooLong"]
            ),
            replay_eligible=False,
        )
        self.assertEqual(verify_capture_payload(payload), payload)

        oversized_raw = retained + "x"
        oversized_row = {
            **_row_base(call),
            "outcome": "oversizedOutput",
            "observedOutputType": "string",
            "rawOutputStringSha256": canonical_string_sha256(oversized_raw),
            "rawOutputCodePointCount": len(oversized_raw),
        }
        oversized = _payload(
            rows=[oversized_row],
            returned=0,
            oversized_count=1,
            diagnostics=_diagnostics(
                oversized=[call["rowId"]], blockers=["oversizedOutput"]
            ),
            replay_eligible=False,
        )
        self.assertNotIn("rawOutput", oversized_row)
        self.assertEqual(verify_capture_payload(oversized), oversized)

    def test_astral_text_obeys_the_shared_verifier_serializer_size_boundary(self) -> None:
        raw = "😀" * MAX_RAW_OUTPUT_CODEPOINTS

        def boundary_payload(row_count: int) -> dict[str, Any]:
            calls = [_call(index) for index in range(row_count)]
            rows = [
                _returned_row(call, raw, second=index + 1)
                for index, call in enumerate(calls)
            ]
            row_ids = [call["rowId"] for call in calls]
            return _payload(
                planned=calls,
                rows=rows,
                attempted=row_count,
                terminal=row_count,
                returned=row_count,
                diagnostics=_diagnostics(
                    scoring_too_long=row_ids, blockers=["scoringTextTooLong"]
                ),
                replay_eligible=False,
            )

        accepted = boundary_payload(21)
        encoded = serialize_capture_payload(accepted)
        self.assertGreater(len(encoded), MAX_CAPTURE_BYTES - 250_000)
        self.assertLessEqual(len(encoded), MAX_CAPTURE_BYTES)
        self.assertEqual(verify_capture_payload(accepted), accepted)

        too_large = boundary_payload(22)
        for operation in (verify_capture_payload, serialize_capture_payload):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(KaggleCaptureError, "payload limit"):
                    operation(too_large)

    def test_unknown_content_filter_and_tool_finish_reasons_are_retained_and_block(self) -> None:
        call = _call(0)
        for reason in ("content_filter", "tool_calls", "new_provider_reason"):
            with self.subTest(reason=reason):
                row = _returned_row(call)
                row["usage"]["finishReason"] = reason
                payload = _payload(
                    rows=[row],
                    diagnostics=_diagnostics(
                        unsupported_finish=[call["rowId"]],
                        blockers=["unsupportedFinishReason"],
                    ),
                    replay_eligible=False,
                )
                self.assertEqual(verify_capture_payload(payload), payload)
                self.assertEqual(payload["rows"][0]["usage"]["finishReason"], reason)

    def test_known_clean_and_limit_finish_reasons_are_closed_sets(self) -> None:
        call = _call(0)
        for reason in ("stop", "end_turn", "eos", "completed"):
            with self.subTest(clean=reason):
                row = _returned_row(call)
                row["usage"]["finishReason"] = reason
                payload = _payload(rows=[row])
                self.assertEqual(verify_capture_payload(payload), payload)

        for reason in ("length", "max_tokens", "max_output_tokens", "token_limit", "MAX_TOKENS"):
            with self.subTest(limit=reason):
                row = _returned_row(call)
                row["usage"]["finishReason"] = reason
                payload = _payload(
                    rows=[row],
                    diagnostics=_diagnostics(
                        token_limit=[call["rowId"]],
                        blockers=["tokenLimitTermination"],
                    ),
                    replay_eligible=False,
                )
                self.assertEqual(verify_capture_payload(payload), payload)

    def test_missing_zero_and_optional_transport_usage_have_distinct_meanings(self) -> None:
        call = _call(0)
        missing_row = _returned_row(call)
        missing_row["usage"]["outputTokens"] = None
        missing = _payload(
            rows=[missing_row],
            diagnostics=_diagnostics(
                missing_usage=[call["rowId"]], blockers=["missingUsage"]
            ),
            replay_eligible=False,
        )
        self.assertEqual(verify_capture_payload(missing), missing)

        zero_row = _returned_row(call)
        zero_row["usage"].update({"inputTokens": 0, "outputTokens": 0})
        zero = _payload(
            rows=[zero_row], diagnostics=_diagnostics(zero_usage=[call["rowId"]])
        )
        self.assertEqual(verify_capture_payload(zero), zero)
        self.assertTrue(zero["canonicalReplayEligible"])

        optional_row = _returned_row(call)
        optional_row["usage"].update(
            {
                "inputCostNanodollars": None,
                "outputCostNanodollars": None,
                "backendLatencyMs": None,
            }
        )
        optional = _payload(rows=[optional_row])
        self.assertEqual(verify_capture_payload(optional), optional)

    def test_usage_cost_strings_reject_non_ascii_digits(self) -> None:
        for field in ("inputCostNanodollars", "outputCostNanodollars"):
            for value in ("\u0661\u0662\u0663", "\uff11\uff12\uff13"):
                with self.subTest(field=field, value=value):
                    self.assertTrue(value.isdigit())
                    usage = _usage()
                    usage[field] = value
                    with self.assertRaisesRegex(
                        KaggleCaptureError, "bounded unsigned integer string"
                    ):
                        _verify_usage(usage, role="rows[0].usage")

        for value in ("0", "9" * 40, None):
            with self.subTest(accepted=value):
                usage = _usage()
                usage["inputCostNanodollars"] = value
                usage["outputCostNanodollars"] = value
                _verify_usage(usage, role="rows[0].usage")

    def test_missing_finish_reason_is_explicit_but_not_a_replay_blocker(self) -> None:
        call = _call(0)
        row = _returned_row(call)
        row["usage"]["finishReason"] = None
        payload = _payload(
            rows=[row],
            diagnostics=_diagnostics(missing_finish_reason=[call["rowId"]]),
        )
        self.assertEqual(verify_capture_payload(payload), payload)
        self.assertTrue(payload["canonicalReplayEligible"])

    def test_rows_are_an_ordered_prefix(self) -> None:
        planned = [_call(0), _call(1)]
        payload = _payload(
            planned=planned,
            rows=[_returned_row(planned[1])],
            diagnostics=_diagnostics(
                coverage_complete=False,
                missing=[planned[1]["rowId"]],
                blockers=["incompleteCoverage"],
            ),
            capture_complete=False,
            replay_eligible=False,
        )
        with self.assertRaisesRegex(KaggleCaptureError, "ordered planned-call prefix"):
            verify_capture_payload(payload)

    def test_counts_diagnostics_completion_and_artifact_id_are_recomputed(self) -> None:
        cases = []
        wrong_count = _payload()
        wrong_count["calls"]["terminalCallCount"] = 0
        _resign(wrong_count)
        cases.append((wrong_count, "call counts"))

        wrong_diagnostics = _payload()
        wrong_diagnostics["diagnostics"]["coverageComplete"] = False
        _resign(wrong_diagnostics)
        cases.append((wrong_diagnostics, "diagnostics"))

        wrong_complete = _payload()
        wrong_complete["captureComplete"] = False
        _resign(wrong_complete)
        cases.append((wrong_complete, "captureComplete"))

        wrong_replay = _payload()
        wrong_replay["canonicalReplayEligible"] = False
        _resign(wrong_replay)
        cases.append((wrong_replay, "canonicalReplayEligible"))

        for payload, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(KaggleCaptureError, pattern):
                    verify_capture_payload(payload)

        wrong_id = _payload()
        wrong_id["id"] = wrong_id["id"][:-1] + (
            "0" if wrong_id["id"][-1] != "0" else "1"
        )
        with self.assertRaisesRegex(KaggleCaptureError, "artifact id"):
            verify_capture_payload(wrong_id)

    def test_schema_errors_duplicate_keys_and_exception_values_never_leak(self) -> None:
        sentinel = "TOP_SECRET_SENTINEL_74C9"

        encoded = json.dumps(_payload(), ensure_ascii=True).encode("ascii")
        duplicate = encoded.replace(b"{", ("{\"" + sentinel + "\":1,\"" + sentinel + "\":2,").encode(), 1)
        with self.assertRaises(KaggleCaptureError) as duplicate_error:
            parse_capture_payload(duplicate)
        self.assertNotIn(sentinel, str(duplicate_error.exception))

        schema_value = _payload()
        schema_value["rows"][0]["terminalFailure"] = {
            "failureKind": "modelCallFailed",
            "exceptionType": "RuntimeError." + sentinel + "-invalid",
            "messageStringSha256": "0" * 64,
        }
        _resign(schema_value)
        with self.assertRaises(KaggleCaptureError) as schema_error:
            verify_capture_payload(schema_value)
        self.assertNotIn(sentinel, str(schema_error.exception))

        extra_key = _payload()
        extra_key["rows"][0][sentinel] = sentinel
        _resign(extra_key)
        with self.assertRaises(KaggleCaptureError) as extra_error:
            verify_capture_payload(extra_key)
        self.assertNotIn(sentinel, str(extra_error.exception))

        malformed = b'{"value":"TOP_SECRET_SENTINEL_74C9" trailing}'
        with self.assertRaises(KaggleCaptureError) as malformed_error:
            parse_capture_payload(malformed)
        self.assertNotIn(sentinel, str(malformed_error.exception))

    def test_parser_rejects_nonfinite_malformed_and_bad_utf8(self) -> None:
        for encoded in (b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}', b'{"value":1e999}'):
            with self.subTest(encoded=encoded):
                with self.assertRaisesRegex(KaggleCaptureError, "non-finite"):
                    parse_capture_payload(encoded)
        with self.assertRaisesRegex(KaggleCaptureError, "not valid JSON"):
            parse_capture_payload(b"{")
        with self.assertRaisesRegex(KaggleCaptureError, "not valid UTF-8"):
            parse_capture_payload(b"\xff")

    def test_timestamps_paths_and_model_mapping_are_cross_checked(self) -> None:
        ended_before_start = _payload()
        ended_before_start["endedAt"] = "2026-09-17T23:59:59Z"
        _resign(ended_before_start)
        with self.assertRaisesRegex(KaggleCaptureError, "precedes"):
            verify_capture_payload(ended_before_start)

        for field_path, value in (
            (("packageIdentity", "manifestPath"), "/absolute/file.json"),
            (("packageIdentity", "manifestPath"), "a/../escape.json"),
            (("taskIdentity", "sourcePath"), "a\\file.py"),
        ):
            with self.subTest(value=value):
                payload = _payload()
                payload[field_path[0]][field_path[1]] = value
                _resign(payload)
                with self.assertRaisesRegex(KaggleCaptureError, "POSIX path"):
                    verify_capture_payload(payload)

        unmapped_claim = _payload()
        unmapped_claim["modelObservation"]["canonicalModelId"] = "provider/model"
        _resign(unmapped_claim)
        with self.assertRaisesRegex(KaggleCaptureError, "unmapped"):
            verify_capture_payload(unmapped_claim)

    def test_nonfinite_direct_values_and_oversized_bytes_fail_closed(self) -> None:
        payload = _payload()
        payload["requestPolicy"]["temperature"] = math.nan
        with self.assertRaisesRegex(KaggleCaptureError, "non-finite"):
            verify_capture_payload(payload)
        with self.assertRaisesRegex(KaggleCaptureError, "payload limit"):
            parse_capture_payload(b" " * (MAX_CAPTURE_BYTES + 1))

    def test_schema_is_closed_world_has_no_placeholder_id_and_is_score_free(self) -> None:
        schema = load_schema(CAPTURE_SCHEMA_PATH)
        self.assertNotIn("$id", schema)
        missing: list[str] = []
        property_names: list[str] = []

        def walk(value: Any, path: str) -> None:
            if isinstance(value, dict):
                if value.get("type") == "object" and value.get("additionalProperties") is not False:
                    missing.append(path)
                properties = value.get("properties")
                if isinstance(properties, dict):
                    property_names.extend(properties)
                for key, child in value.items():
                    walk(child, f"{path}/{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, f"{path}/{index}")

        walk(schema, "$schema")
        self.assertEqual(missing, [])
        self.assertFalse([name for name in property_names if name.lower() in {"score", "metrics"}])
        with self.assertRaises(SchemaValidationError):
            validate(_payload(), load_schema(RESULT_SCHEMA_PATH))

    def test_bounded_loader_accepts_regular_file_and_rejects_symlink_and_directory(self) -> None:
        payload = _payload()
        encoded = serialize_capture_payload(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture.json"
            capture.write_bytes(encoded)
            self.assertEqual(load_capture_payload(capture), payload)

            link = root / "capture-link.json"
            try:
                os.symlink(capture, link)
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaisesRegex(KaggleCaptureError, "symlink"):
                    load_capture_payload(link)

            with self.assertRaisesRegex(KaggleCaptureError, "regular file"):
                load_capture_payload(root)


if __name__ == "__main__":
    unittest.main()
