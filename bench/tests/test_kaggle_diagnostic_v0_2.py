from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from bench.engine.platform_package_v0_2 import write_v0_2_package
from bench.engine.schema_validation import (
    SchemaValidationError,
    load_schema,
    validate,
)
from bench.tasks.kaggle import generate_v0_2_diagnostic as diagnostic_generator
from bench.tasks.kaggle.generate_v0_2_diagnostic import (
    CANARY_PROMPT_IDS,
    OUTPUT_PATH,
    _generation_context,
    render_task_source,
)


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_SCHEMA = (
    ROOT / "schemas/v0.2/aleph-bench-kaggle-diagnostic-receipt.schema.json"
)
RESULT_SCHEMA = ROOT / "schemas/v0.2/aleph-bench-result.schema.json"
CONFORMANT_RUNTIME = {
    "pythonVersion": "3.13",
    "pythonFullVersion": "3.13.2",
    "unicodeDatabaseVersion": "15.1.0",
}
FIXED_TIME = "2026-09-18T00:00:00Z"


class _TaskWrapper:
    def __init__(self, function: Any, metadata: dict[str, Any]) -> None:
        self.function = function
        self.metadata = metadata
        self.run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.run_calls.append((args, kwargs))


class _Assertions:
    def __init__(self) -> None:
        self.results: list[tuple[bool, str | None]] = []

    def assert_true(
        self, expression: bool, expectation: str | None = None
    ) -> types.SimpleNamespace:
        self.results.append((expression, expectation))
        return types.SimpleNamespace(passed=expression, expectation=expectation)


def _fake_kaggle_module() -> types.ModuleType:
    module = types.ModuleType("kaggle_benchmarks")

    def task(**metadata: Any) -> Any:
        return lambda function: _TaskWrapper(function, metadata)

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
        "aleph_bench_v0_2_diagnostic_generated", OUTPUT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load generated Kaggle diagnostic task")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, previous


class _Bomb:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"unexpected access: {name}")


class _Usage:
    def __init__(
        self,
        *,
        input_tokens: int | None = 20,
        output_tokens: int | None = 100,
        input_cost: int | None = 1_000,
        output_cost: int | None = 2_000,
        latency: int | None = 50,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_cost_nanodollars = input_cost
        self.output_tokens_cost_nanodollars = output_cost
        self.total_backend_latency_ms = latency


class _Chat:
    def __init__(self, name: str, usage: _Usage) -> None:
        self.name = name
        self.usage = usage

    def __enter__(self) -> "_Chat":
        return self

    def __exit__(self, *args: Any) -> None:
        del args


class _EnterFailChat(_Chat):
    def __enter__(self) -> "_Chat":
        raise RuntimeError("chat enter failed")


class _ExitFailChat(_Chat):
    def __exit__(self, *args: Any) -> None:
        del args
        raise RuntimeError("chat exit failed")


class _Chats:
    def __init__(self, usages: list[_Usage] | None = None) -> None:
        self.names: list[str] = []
        self.usages = usages or [_Usage() for _ in CANARY_PROMPT_IDS]

    def new(self, name: str) -> _Chat:
        index = len(self.names)
        self.names.append(name)
        return _Chat(name, self.usages[index])


class _CustomChats(_Chats):
    def __init__(self, chat_type: type[_Chat]) -> None:
        super().__init__()
        self.chat_type = chat_type

    def new(self, name: str) -> _Chat:
        index = len(self.names)
        self.names.append(name)
        return self.chat_type(name, self.usages[index])


class _OpenAIClient:
    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    def with_options(self, *, max_retries: int) -> "_OpenAIClient":
        return _OpenAIClient(max_retries=max_retries)


class OpenAI:
    model = "google/gemini-2.5-flash"

    def __init__(self, outputs: list[Any] | None = None) -> None:
        self.outputs = outputs or [f"output-{index}" for index in range(6)]
        self.calls: list[dict[str, Any]] = []
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
        del schema, tools, image, video, audio
        index = len(self.calls)
        self.calls.append(
            {
                "text": text,
                "seed": seed,
                "temperature": temperature,
                "reasoning": reasoning,
                "extra_api_params": extra_api_params,
            }
        )
        output = self.outputs[index]
        if isinstance(output, BaseException):
            raise output
        return output


OpenAI.__module__ = "kaggle_benchmarks.actors.llms"


class _StopAfterOne:
    max_attempt_number = 1


class _GoogleApiClient:
    _http_options = types.SimpleNamespace(retry_options=None)
    _retry = types.SimpleNamespace(stop=_StopAfterOne())


class GoogleGenAI(OpenAI):
    def __init__(self, outputs: list[Any] | None = None) -> None:
        super().__init__(outputs)
        self.client = types.SimpleNamespace(_api_client=_GoogleApiClient())


GoogleGenAI.__module__ = "kaggle_benchmarks.actors.llms"


class _HardStop(BaseException):
    pass


class KaggleDiagnosticV02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated, cls.previous_kaggle_module = _load_generated_task()
        cls.package_tmp = tempfile.TemporaryDirectory()
        cls.package_root = Path(cls.package_tmp.name) / "package"
        write_v0_2_package(cls.package_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.package_tmp.cleanup()
        if cls.previous_kaggle_module is None:
            sys.modules.pop("kaggle_benchmarks", None)
        else:
            sys.modules["kaggle_benchmarks"] = cls.previous_kaggle_module

    def _run(
        self,
        *,
        llm: Any,
        chats: Any,
        runtime: dict[str, str] | None = None,
        package_root: Path | None = None,
    ) -> tuple[dict[str, Any], Path, tempfile.TemporaryDirectory[str]]:
        output_tmp: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        receipt_path = Path(output_tmp.name) / "receipt.json"
        receipt = self.generated.run_deployment_diagnostic(
            llm,
            chats=chats,
            package_root=package_root or self.package_root,
            receipt_path=receipt_path,
            observed_runtime=runtime or CONFORMANT_RUNTIME,
            clock=lambda: FIXED_TIME,
        )
        self.assertEqual(
            json.loads(receipt_path.read_text(encoding="utf-8")), receipt
        )
        validate(receipt, load_schema(DIAGNOSTIC_SCHEMA))
        return receipt, receipt_path, output_tmp

    def test_generated_task_is_current_and_has_non_leaderboard_shape(self) -> None:
        self.assertEqual(OUTPUT_PATH.read_text(encoding="utf-8"), render_task_source())
        source = OUTPUT_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "def aleph_bench_deployment_diagnostic_2048_none(llm) -> dict:",
            source,
        )
        self.assertIn(
            "aleph_bench_deployment_diagnostic_2048_none.run(kbench.llm)",
            source,
        )
        self.assertNotIn("%choose", source)
        wrapper = self.generated.aleph_bench_deployment_diagnostic_2048_none
        self.assertEqual(wrapper.metadata["name"], self.generated.TASK_NAME)
        self.assertEqual(len(wrapper.run_calls), 1)
        self.assertEqual(
            self.generated.IMPLEMENTATION_SHA256,
            _generation_context()["implementationSha256"],
        )

    def test_implementation_digest_covers_generated_header_behavior(self) -> None:
        original_context = _generation_context()
        original_source = render_task_source()
        with mock.patch.object(
            diagnostic_generator,
            "MAX_OUTPUT_CHARACTERS",
            diagnostic_generator.MAX_OUTPUT_CHARACTERS + 1,
        ):
            drifted_context = diagnostic_generator._generation_context()
            drifted_source = diagnostic_generator.render_task_source()

        self.assertNotEqual(original_source, drifted_source)
        self.assertNotEqual(
            original_context["implementationSha256"],
            drifted_context["implementationSha256"],
        )
        self.assertNotEqual(
            original_context["definitionSha256"],
            drifted_context["definitionSha256"],
        )

    def test_task_wrapper_records_blocked_receipt_as_failed_assertion(self) -> None:
        wrapper = self.generated.aleph_bench_deployment_diagnostic_2048_none
        assertions = self.generated.kbench.assertions
        assertions.results.clear()
        blocked = {"status": "blocked"}
        with mock.patch.object(
            self.generated, "run_deployment_diagnostic", return_value=blocked
        ):
            self.assertIs(wrapper(object()), blocked)
        self.assertEqual(len(assertions.results), 1)
        self.assertFalse(assertions.results[0][0])

        assertions.results.clear()
        complete = {"status": "complete"}
        with mock.patch.object(
            self.generated, "run_deployment_diagnostic", return_value=complete
        ):
            self.assertIs(wrapper(object()), complete)
        self.assertEqual(len(assertions.results), 1)
        self.assertTrue(assertions.results[0][0])
        self.assertIn("must complete", assertions.results[0][1])

    def test_generator_selects_literal_ordered_six_prompt_set(self) -> None:
        context = _generation_context()
        prompt_ids = tuple(row["promptId"] for row in context["prompts"])
        self.assertEqual(prompt_ids, CANARY_PROMPT_IDS)
        self.assertEqual(context["policy"]["promptIds"], list(CANARY_PROMPT_IDS))
        self.assertTrue(all(row["rung"] == 1 for row in context["prompts"]))
        self.assertEqual(
            [row["paraphrase"] for row in context["prompts"]],
            [0, 1, 0, 1, 0, 1],
        )
        self.assertTrue(
            all(
                row["expectedLeakage"] == "non_leaking_candidate"
                for row in context["prompts"]
            )
        )

    def test_python_and_unicode_mismatch_each_block_before_any_access(self) -> None:
        mismatches = {
            "python": {
                "pythonVersion": "3.11",
                "pythonFullVersion": "3.11.13",
                "unicodeDatabaseVersion": "15.1.0",
            },
            "unicode": {
                "pythonVersion": "3.13",
                "pythonFullVersion": "3.13.2",
                "unicodeDatabaseVersion": "14.0.0",
            },
        }
        for name, runtime in mismatches.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    receipt_path = Path(tmp) / "receipt.json"
                    receipt = self.generated.run_deployment_diagnostic(
                        _Bomb(),
                        chats=_Bomb(),
                        package_root=Path(tmp) / "must-not-be-read",
                        receipt_path=receipt_path,
                        observed_runtime=runtime,
                        clock=lambda: FIXED_TIME,
                    )
                    self.assertEqual(receipt["phase"], "runtime_preflight")
                    self.assertEqual(receipt["calls"]["attempted"], 0)
                    self.assertEqual(receipt["calls"]["completed"], 0)
                    self.assertEqual(receipt["rows"], [])
                    self.assertEqual(
                        receipt["diagnostics"]["blockedReasons"],
                        ["runtime_mismatch"],
                    )
                    self.assertEqual(
                        receipt["dependency"]["observed"]["status"], "unchecked"
                    )
                    self.assertEqual(receipt["modelIdentity"]["status"], "unavailable")
                    validate(receipt, load_schema(DIAGNOSTIC_SCHEMA))

    def test_package_drift_blocks_before_model_or_chat_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            drifted = Path(tmp) / "package"
            shutil.copytree(self.package_root, drifted)
            manifest = drifted / "package-manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            receipt, _, output_tmp = self._run(
                llm=_Bomb(), chats=_Bomb(), package_root=drifted
            )
            try:
                self.assertEqual(receipt["phase"], "package_preflight")
                self.assertEqual(receipt["calls"]["attempted"], 0)
                self.assertEqual(
                    receipt["diagnostics"]["blockedReasons"],
                    ["package_unavailable_or_drifted"],
                )
                self.assertIsNotNone(receipt["diagnostics"]["failure"])
            finally:
                output_tmp.cleanup()

    def test_conformant_path_makes_exact_six_isolated_calls(self) -> None:
        llm = OpenAI()
        chats = _Chats()
        receipt, _, output_tmp = self._run(llm=llm, chats=chats)
        try:
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["phase"], "complete")
            self.assertFalse(receipt["protocolConformant"])
            self.assertFalse(receipt["leaderboardEligible"])
            self.assertFalse(receipt["publicationEligible"])
            self.assertIsNone(receipt["diagnosticScalar"])
            self.assertEqual(receipt["calls"]["attempted"], 6)
            self.assertEqual(receipt["calls"]["completed"], 6)
            self.assertEqual(len(receipt["rows"]), 6)
            self.assertEqual(llm.client.max_retries, 0)
            self.assertEqual(
                receipt["modelIdentity"]["transportMaxAttempts"], 1
            )
            self.assertEqual(
                receipt["modelIdentity"]["transportSdkName"], "openai"
            )
            self.assertEqual(
                chats.names,
                [
                    f"{row['itemId']}:{row['promptId']}"
                    for row in self.generated.CANARY_PROMPTS
                ],
            )
            self.assertEqual(
                [call["text"] for call in llm.calls],
                [row["prompt"] for row in self.generated.CANARY_PROMPTS],
            )
            for call in llm.calls:
                self.assertEqual(call["reasoning"], "none")
                self.assertEqual(call["seed"], 0)
                self.assertEqual(call["temperature"], 0)
                self.assertEqual(call["extra_api_params"], {"max_tokens": 2048})
            self.assertEqual(
                receipt["calls"]["aggregateUsage"],
                {
                    "inputTokens": 120,
                    "outputTokens": 600,
                    "inputCostNanodollars": "6000",
                    "outputCostNanodollars": "12000",
                    "totalBackendLatencyMs": 300,
                },
            )
        finally:
            output_tmp.cleanup()

    def test_transport_retry_policy_is_verified_before_model_calls(self) -> None:
        google = GoogleGenAI()
        identity, parameter, error = self.generated._model_preflight(google)
        self.assertIsNone(error)
        self.assertEqual(parameter, "max_output_tokens")
        self.assertEqual(identity["transportSdkName"], "google-genai")
        self.assertEqual(identity["transportMaxAttempts"], 1)

        unsafe = OpenAI()
        unsafe.client = object()
        identity, parameter, error = self.generated._model_preflight(unsafe)
        self.assertEqual(error, "transport_retry_policy_unverified")
        self.assertIsNone(parameter)
        self.assertIsNone(identity["transportMaxAttempts"])
        self.assertEqual(unsafe.calls, [])

        blocked, _, output_tmp = self._run(llm=unsafe, chats=_Bomb())
        try:
            self.assertEqual(blocked["phase"], "model_preflight")
            self.assertEqual(blocked["calls"]["attempted"], 0)
            self.assertEqual(
                blocked["diagnostics"]["blockedReasons"],
                ["transport_retry_policy_unverified"],
            )
        finally:
            output_tmp.cleanup()

    def test_early_model_preflight_failures_remain_verifiable(self) -> None:
        missing_identity = OpenAI()
        missing_identity.model = None
        missing_identity.name = None
        incompatible = OpenAI()
        incompatible.prompt = lambda text: text
        cases = {
            "identity": (missing_identity, "model_identity_unavailable"),
            "signature": (incompatible, "sdk_api_incompatible"),
        }
        for name, (llm, reason) in cases.items():
            with self.subTest(name=name):
                receipt, _, output_tmp = self._run(llm=llm, chats=_Bomb())
                try:
                    self.assertEqual(receipt["phase"], "model_preflight")
                    self.assertEqual(receipt["calls"]["attempted"], 0)
                    self.assertEqual(
                        receipt["diagnostics"]["blockedReasons"], [reason]
                    )
                    self.assertIsNone(
                        receipt["modelIdentity"]["transportSdkName"]
                    )
                finally:
                    output_tmp.cleanup()

    def test_first_bad_observation_stops_canary_and_preserves_detail(self) -> None:
        cases = {
            "empty": {
                "outputs": [""],
                "usages": [_Usage()],
                "reason": "empty_output",
                "completed": 1,
            },
            "missing-usage": {
                "outputs": ["output"],
                "usages": [_Usage(input_tokens=None)],
                "reason": "invalid_usage",
                "completed": 1,
            },
            "zero-usage": {
                "outputs": ["output"],
                "usages": [_Usage(output_tokens=0)],
                "reason": "invalid_usage",
                "completed": 1,
            },
            "near-cap": {
                "outputs": ["output"],
                "usages": [_Usage(output_tokens=2044)],
                "reason": "near_output_cap",
                "completed": 1,
            },
            "non-string": {
                "outputs": [7],
                "usages": [_Usage()],
                "reason": "non_string_output",
                "completed": 1,
            },
            "oversized": {
                "outputs": ["x" * 65_537],
                "usages": [_Usage()],
                "reason": "output_character_limit",
                "completed": 1,
            },
            "proxy-error": {
                "outputs": [RuntimeError("proxy unavailable")],
                "usages": [_Usage()],
                "reason": "model_call_failed",
                "completed": 0,
            },
        }
        for name, case in cases.items():
            with self.subTest(name=name):
                llm = OpenAI(outputs=case["outputs"])
                chats = _Chats(usages=case["usages"])
                receipt, _, output_tmp = self._run(llm=llm, chats=chats)
                try:
                    self.assertEqual(receipt["status"], "blocked")
                    self.assertEqual(receipt["calls"]["attempted"], 1)
                    self.assertEqual(
                        receipt["calls"]["completed"], case["completed"]
                    )
                    self.assertIn(
                        case["reason"], receipt["diagnostics"]["blockedReasons"]
                    )
                    self.assertEqual(len(llm.calls), 1)
                    self.assertEqual(len(chats.names), 1)
                    if case["completed"]:
                        self.assertIsNone(receipt["calls"]["activeCall"])
                    else:
                        self.assertEqual(
                            receipt["calls"]["activeCall"]["state"],
                            "dispatching",
                        )
                    self.assertIsNone(receipt["diagnosticScalar"])
                finally:
                    output_tmp.cleanup()

    def test_partial_rows_survive_a_later_model_failure(self) -> None:
        llm = OpenAI(outputs=["first", RuntimeError("second failed")])
        chats = _Chats(usages=[_Usage(), _Usage()])
        receipt, receipt_path, output_tmp = self._run(llm=llm, chats=chats)
        try:
            self.assertEqual(receipt["calls"]["attempted"], 2)
            self.assertEqual(receipt["calls"]["completed"], 1)
            self.assertEqual(len(receipt["rows"]), 1)
            self.assertEqual(receipt["rows"][0]["outputText"], "first")
            self.assertEqual(
                receipt["diagnostics"]["failure"]["promptId"],
                "s2-001-r1-p1",
            )
            self.assertEqual(json.loads(receipt_path.read_text()), receipt)
        finally:
            output_tmp.cleanup()

    def test_call_ahead_receipt_survives_hard_interruption(self) -> None:
        llm = OpenAI(outputs=[_HardStop("worker terminated")])
        chats = _Chats()
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = Path(tmp) / "receipt.json"
            with self.assertRaises(_HardStop):
                self.generated.run_deployment_diagnostic(
                    llm,
                    chats=chats,
                    package_root=self.package_root,
                    receipt_path=receipt_path,
                    observed_runtime=CONFORMANT_RUNTIME,
                    clock=lambda: FIXED_TIME,
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            validate(receipt, load_schema(DIAGNOSTIC_SCHEMA))
            self.assertEqual(receipt["calls"]["attempted"], 1)
            self.assertEqual(receipt["calls"]["completed"], 0)
            self.assertEqual(receipt["rows"], [])
            self.assertEqual(
                receipt["calls"]["activeCall"],
                {
                    "promptId": "s2-001-r1-p0",
                    "conversationName": "s2-001:s2-001-r1-p0",
                    "state": "dispatching",
                },
            )

    def test_chat_lifecycle_failures_preserve_call_boundary(self) -> None:
        enter_llm = OpenAI()
        enter_receipt, _, enter_tmp = self._run(
            llm=enter_llm, chats=_CustomChats(_EnterFailChat)
        )
        try:
            self.assertEqual(enter_receipt["calls"]["attempted"], 0)
            self.assertEqual(enter_receipt["calls"]["completed"], 0)
            self.assertEqual(enter_llm.calls, [])
            self.assertEqual(
                enter_receipt["calls"]["activeCall"]["state"], "prepared"
            )
            self.assertIn(
                "chat_open_failed",
                enter_receipt["diagnostics"]["blockedReasons"],
            )
        finally:
            enter_tmp.cleanup()

        exit_llm = OpenAI()
        exit_receipt, _, exit_tmp = self._run(
            llm=exit_llm, chats=_CustomChats(_ExitFailChat)
        )
        try:
            self.assertEqual(exit_receipt["calls"]["attempted"], 1)
            self.assertEqual(exit_receipt["calls"]["completed"], 1)
            self.assertEqual(len(exit_receipt["rows"]), 1)
            self.assertEqual(
                exit_receipt["calls"]["activeCall"]["state"], "returned"
            )
            self.assertIn(
                "chat_close_failed",
                exit_receipt["diagnostics"]["blockedReasons"],
            )
        finally:
            exit_tmp.cleanup()

    def test_receipt_id_covers_every_non_id_field_and_is_not_bench_result(self) -> None:
        receipt, _, output_tmp = self._run(llm=OpenAI(), chats=_Chats())
        try:
            payload = copy.deepcopy(receipt)
            artifact_id = payload.pop("id")
            digest = hashlib.sha256(
                json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(
                artifact_id,
                f"aleph-bench-kaggle-diagnostic-v0.2-artifact-{digest}",
            )
            with self.assertRaises(SchemaValidationError):
                validate(receipt, load_schema(RESULT_SCHEMA))
            self.assertNotIn("aggregate", receipt)
            self.assertNotIn("itemRuns", receipt)
            self.assertNotIn("models", receipt)
            self.assertNotIn("aurc", receipt)
        finally:
            output_tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
