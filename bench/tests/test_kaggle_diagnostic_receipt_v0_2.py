from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from bench.engine.kaggle_diagnostic_receipt import (
    ARTIFACT_ID_PREFIX,
    KaggleDiagnosticReceiptError,
    diagnostic_receipt_requires_manual_review,
    load_kaggle_diagnostic_receipt,
    main as receipt_main,
    parse_kaggle_diagnostic_receipt_bytes,
    verify_kaggle_diagnostic_receipt,
)
from bench.engine.platform_package_v0_2 import write_v0_2_package
from bench.engine.schema_validation import SchemaValidationError
from bench.tasks.kaggle.generate_v0_2_diagnostic import OUTPUT_PATH


CONFORMANT_RUNTIME = {
    "pythonVersion": "3.13",
    "pythonFullVersion": "3.13.2",
    "unicodeDatabaseVersion": "15.1.0",
}
FIXED_TIME = "2026-09-18T00:00:00Z"


class _TaskWrapper:
    def __init__(self, function: Any) -> None:
        self.function = function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _Assertions:
    def assert_true(self, condition: bool, *, expectation: str) -> None:
        del condition, expectation


def _fake_kaggle_module() -> types.ModuleType:
    module = types.ModuleType("kaggle_benchmarks")

    def task(**metadata: Any) -> Any:
        del metadata
        return lambda function: _TaskWrapper(function)

    module.task = task  # type: ignore[attr-defined]
    module.llm = object()  # type: ignore[attr-defined]
    module.chats = object()  # type: ignore[attr-defined]
    module.assertions = _Assertions()  # type: ignore[attr-defined]
    module.__version__ = "test-sdk"
    return module


def _load_generated_task() -> tuple[types.ModuleType, types.ModuleType | None]:
    previous = sys.modules.get("kaggle_benchmarks")
    sys.modules["kaggle_benchmarks"] = _fake_kaggle_module()
    spec = importlib.util.spec_from_file_location(
        "aleph_bench_v0_2_diagnostic_receipt_test", OUTPUT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load generated Kaggle diagnostic task")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, previous


class _Usage:
    input_tokens = 20
    output_tokens = 100
    input_tokens_cost_nanodollars = 1_000
    output_tokens_cost_nanodollars = 2_000
    total_backend_latency_ms = 50


class _Chat:
    usage = _Usage()

    def __enter__(self) -> "_Chat":
        return self

    def __exit__(self, *args: Any) -> None:
        del args


class _Chats:
    def new(self, name: str) -> _Chat:
        del name
        return _Chat()


class _HardStop(BaseException):
    pass


class _OpenAIClient:
    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    def with_options(self, *, max_retries: int) -> "_OpenAIClient":
        return _OpenAIClient(max_retries=max_retries)


class OpenAI:
    model = "google/gemini-2.5-flash"

    def __init__(self, outputs: list[Any] | None = None) -> None:
        self.outputs = outputs or [f"output-{index}" for index in range(6)]
        self.calls = 0
        self.client = _OpenAIClient()

    def prompt(
        self,
        text: str,
        schema: type = str,
        seed: int = 0,
        temperature: float = 0,
        tools: Any = None,
        image: Any = None,
        video: Any = None,
        audio: Any = None,
        reasoning: str | None = None,
        extra_api_params: dict[str, Any] | None = None,
    ) -> Any:
        del (
            text,
            schema,
            seed,
            temperature,
            tools,
            image,
            video,
            audio,
            reasoning,
            extra_api_params,
        )
        output = self.outputs[self.calls]
        self.calls += 1
        if isinstance(output, BaseException):
            raise output
        return output


OpenAI.__module__ = "kaggle_benchmarks.actors.llms"


def _resign(receipt: dict[str, Any]) -> None:
    payload = {key: value for key, value in receipt.items() if key != "id"}
    digest = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt["id"] = ARTIFACT_ID_PREFIX + digest


class KaggleDiagnosticReceiptV02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated, cls.previous_kaggle_module = _load_generated_task()
        cls.package_tmp = tempfile.TemporaryDirectory()
        cls.package_root = Path(cls.package_tmp.name) / "package"
        write_v0_2_package(cls.package_root)
        cls.complete_receipt = cls._run_generated(OpenAI())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.package_tmp.cleanup()
        if cls.previous_kaggle_module is None:
            sys.modules.pop("kaggle_benchmarks", None)
        else:
            sys.modules["kaggle_benchmarks"] = cls.previous_kaggle_module

    @classmethod
    def _run_generated(
        cls,
        llm: Any,
        *,
        runtime: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            return cls.generated.run_deployment_diagnostic(
                llm,
                chats=_Chats(),
                package_root=cls.package_root,
                receipt_path=Path(tmp) / "receipt.json",
                observed_runtime=runtime or CONFORMANT_RUNTIME,
                clock=lambda: FIXED_TIME,
            )

    def _mutated(self) -> dict[str, Any]:
        return copy.deepcopy(self.complete_receipt)

    def _failed_call_receipt(self) -> dict[str, Any]:
        return self._run_generated(OpenAI([RuntimeError("proxy unavailable")]))

    def _runtime_mismatch_receipt(self) -> dict[str, Any]:
        return self._run_generated(
            object(),
            runtime={
                "pythonVersion": "3.11",
                "pythonFullVersion": "3.11.13",
                "unicodeDatabaseVersion": "14.0.0",
            },
        )

    def test_accepts_complete_generator_receipt(self) -> None:
        verify_kaggle_diagnostic_receipt(self.complete_receipt)
        self.assertFalse(
            diagnostic_receipt_requires_manual_review(self.complete_receipt)
        )
        encoded = json.dumps(self.complete_receipt).encode("utf-8")
        self.assertEqual(
            parse_kaggle_diagnostic_receipt_bytes(encoded),
            self.complete_receipt,
        )

    def test_schema_validation_runs_before_semantic_validation(self) -> None:
        receipt = self._mutated()
        receipt["unexpected"] = True
        with self.assertRaises(SchemaValidationError):
            verify_kaggle_diagnostic_receipt(receipt)

    def test_content_address_covers_every_non_id_field(self) -> None:
        receipt = self._mutated()
        receipt["endedAt"] = "2026-09-18T00:00:01Z"
        with self.assertRaisesRegex(
            KaggleDiagnosticReceiptError, "artifact id mismatch"
        ):
            verify_kaggle_diagnostic_receipt(receipt)

    def test_task_definition_and_implementation_identity_are_pinned(self) -> None:
        for field in ("definitionSha256", "implementationSha256"):
            with self.subTest(field=field):
                receipt = self._mutated()
                receipt["taskIdentity"][field] = "0" * 64
                _resign(receipt)
                with self.assertRaises(SchemaValidationError):
                    verify_kaggle_diagnostic_receipt(receipt)

    def test_model_call_receipt_requires_one_attempt_transport(self) -> None:
        receipt = self._failed_call_receipt()
        receipt["modelIdentity"]["transportMaxAttempts"] = None
        _resign(receipt)
        with self.assertRaises(SchemaValidationError):
            verify_kaggle_diagnostic_receipt(receipt)

    def test_call_bounds_and_row_count_are_semantic(self) -> None:
        cases = {
            "completed exceeds attempted": (0, 1),
            "rows disagree with completed": (1, 1),
        }
        for name, (attempted, completed) in cases.items():
            with self.subTest(name=name):
                receipt = self._failed_call_receipt()
                receipt["calls"]["attempted"] = attempted
                receipt["calls"]["completed"] = completed
                _resign(receipt)
                with self.assertRaisesRegex(
                    KaggleDiagnosticReceiptError,
                    "call counts|retained row count",
                ):
                    verify_kaggle_diagnostic_receipt(receipt)

    def test_rows_must_be_unique_canonical_prefix_in_order(self) -> None:
        cases: dict[str, Any] = {
            "reordered": lambda rows: rows.__setitem__(
                slice(0, 2), [rows[1], rows[0]]
            ),
            "duplicated": lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                receipt = self._mutated()
                mutate(receipt["rows"])
                _resign(receipt)
                with self.assertRaisesRegex(
                    KaggleDiagnosticReceiptError, "canonical prompt prefix"
                ):
                    verify_kaggle_diagnostic_receipt(receipt)

    def test_prompt_item_conversation_and_prompt_digest_are_bound(self) -> None:
        cases = {
            "item": ("itemId", "s2-002", "itemId"),
            "conversation": ("conversationName", "wrong:name", "conversation"),
            "prompt hash": ("promptSha256", "0" * 64, "prompt digest"),
        }
        for name, (field, value, message) in cases.items():
            with self.subTest(name=name):
                receipt = self._mutated()
                receipt["rows"][0][field] = value
                _resign(receipt)
                with self.assertRaisesRegex(
                    KaggleDiagnosticReceiptError, message
                ):
                    verify_kaggle_diagnostic_receipt(receipt)

    def test_output_hash_and_character_count_are_recomputed(self) -> None:
        cases = {
            "hash": ("outputSha256", "0" * 64, "output digest"),
            "characters": ("outputCharacters", 999, "character count"),
        }
        for name, (field, value, message) in cases.items():
            with self.subTest(name=name):
                receipt = self._mutated()
                receipt["rows"][0][field] = value
                _resign(receipt)
                with self.assertRaisesRegex(
                    KaggleDiagnosticReceiptError, message
                ):
                    verify_kaggle_diagnostic_receipt(receipt)

    def test_usage_aggregate_and_diagnostics_are_recomputed(self) -> None:
        aggregate_drift = self._mutated()
        aggregate_drift["calls"]["aggregateUsage"]["inputTokens"] += 1
        _resign(aggregate_drift)
        with self.assertRaisesRegex(
            KaggleDiagnosticReceiptError, "aggregate usage"
        ):
            verify_kaggle_diagnostic_receipt(aggregate_drift)

        diagnostics_drift = self._failed_call_receipt()
        diagnostics_drift["diagnostics"]["invalidUsagePromptIds"] = [
            diagnostics_drift["policy"]["promptIds"][0]
        ]
        _resign(diagnostics_drift)
        with self.assertRaisesRegex(
            KaggleDiagnosticReceiptError, "invalidUsagePromptIds"
        ):
            verify_kaggle_diagnostic_receipt(diagnostics_drift)

    def test_status_phase_reason_and_timestamp_must_be_coherent(self) -> None:
        bad_phase = self._runtime_mismatch_receipt()
        bad_phase["phase"] = "package_preflight"
        _resign(bad_phase)
        with self.assertRaisesRegex(
            KaggleDiagnosticReceiptError, "requires a conformant runtime"
        ):
            verify_kaggle_diagnostic_receipt(bad_phase)

        backwards_time = self._mutated()
        backwards_time["startedAt"] = "2026-09-18T00:00:01Z"
        _resign(backwards_time)
        with self.assertRaisesRegex(
            KaggleDiagnosticReceiptError, "ended before it started"
        ):
            verify_kaggle_diagnostic_receipt(backwards_time)

    def test_generator_blocked_receipts_remain_verifiable(self) -> None:
        runtime_receipt = self._runtime_mismatch_receipt()
        verify_kaggle_diagnostic_receipt(runtime_receipt)
        self.assertFalse(
            diagnostic_receipt_requires_manual_review(runtime_receipt)
        )

        failed_call_receipt = self._failed_call_receipt()
        verify_kaggle_diagnostic_receipt(failed_call_receipt)
        self.assertTrue(
            diagnostic_receipt_requires_manual_review(failed_call_receipt)
        )

        bad_observation_receipt = self._run_generated(OpenAI([""]))
        verify_kaggle_diagnostic_receipt(bad_observation_receipt)
        self.assertIsNone(bad_observation_receipt["calls"]["activeCall"])
        self.assertEqual(bad_observation_receipt["calls"]["attempted"], 1)
        self.assertTrue(
            diagnostic_receipt_requires_manual_review(bad_observation_receipt)
        )

    def test_active_call_journal_requires_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = Path(tmp) / "receipt.json"
            with self.assertRaises(_HardStop):
                self.generated.run_deployment_diagnostic(
                    OpenAI([_HardStop("worker terminated")]),
                    chats=_Chats(),
                    package_root=self.package_root,
                    receipt_path=receipt_path,
                    observed_runtime=CONFORMANT_RUNTIME,
                    clock=lambda: FIXED_TIME,
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        verify_kaggle_diagnostic_receipt(receipt)
        self.assertEqual(receipt["calls"]["activeCall"]["state"], "dispatching")
        self.assertTrue(diagnostic_receipt_requires_manual_review(receipt))

    def test_strict_loader_and_command_line_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = Path(tmp) / "receipt.json"
            receipt_path.write_text(
                json.dumps(self.complete_receipt, ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual(
                load_kaggle_diagnostic_receipt(receipt_path),
                self.complete_receipt,
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(receipt_main([str(receipt_path)]), 0)
            self.assertIn("manual review: no", stdout.getvalue())

            duplicate_path = Path(tmp) / "duplicate.json"
            duplicate_path.write_text(
                '{"receiptVersion":"0.2.0",'
                + json.dumps(self.complete_receipt, ensure_ascii=False)[1:],
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KaggleDiagnosticReceiptError, "duplicate JSON object key"
            ):
                load_kaggle_diagnostic_receipt(duplicate_path)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(receipt_main([str(duplicate_path)]), 1)
            self.assertIn("invalid Kaggle diagnostic receipt", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
