from __future__ import annotations

import copy
import contextlib
import io
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from bench.engine.kaggle_capture import (
    load_capture_payload,
    verify_capture_payload,
)
from bench.engine.platform_package_v0_2 import (
    CHECKSUMS_NAME,
    MANIFEST_NAME,
    _expected_package,
)
from bench.tasks.kaggle import generate_v0_2_capture as capture_generator


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "bench/tasks/kaggle/aleph_bench_v0_2_capture.py"
RUNTIME_311 = {
    "pythonVersion": "3.11",
    "pythonFullVersion": "3.11.9",
    "unicodeDatabaseVersion": "14.0.0",
    "platform": "Linux-x86_64",
}
RUNTIME_313 = {
    "pythonVersion": "3.13",
    "pythonFullVersion": "3.13.2",
    "unicodeDatabaseVersion": "15.1.0",
    "platform": "Linux-x86_64",
}


class _TaskWrapper:
    def __init__(self, function: Any, metadata: dict[str, Any]) -> None:
        self.function = function
        self.metadata = metadata
        self.run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        self.run_calls.append((args, kwargs))
        return {"status": "recorded"}


class _Assertions:
    def assert_true(self, value: Any, *, expectation: str) -> None:
        del value, expectation


def _fake_kaggle_module() -> types.ModuleType:
    module = types.ModuleType("kaggle_benchmarks")

    def task(**metadata: Any):
        def decorate(function: Any) -> _TaskWrapper:
            # Match Kaggle Benchmarks 0.6.1 result inference closely enough to
            # reject postponed string annotations before a hosted run starts.
            if function.__annotations__.get("return") is not dict:
                raise TypeError("task return annotation is not the built-in dict type")
            return _TaskWrapper(function, metadata)

        return decorate

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
        "aleph_bench_v0_2_capture_generated", OUTPUT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load generated Kaggle capture task")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, previous


def _write_package(root: Path) -> None:
    manifest, artifacts, checksums = _expected_package()
    for relative, content in artifacts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / MANIFEST_NAME).write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    (root / CHECKSUMS_NAME).write_bytes(checksums)


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"2026-09-22T00:00:{self.value:02d}Z"


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


class _Message:
    def __init__(self, finish_reason: str | None) -> None:
        self._meta = {}
        if finish_reason is not None:
            self._meta["finish_reason"] = finish_reason


class _Chat:
    def __init__(
        self,
        name: str,
        *,
        usage: _Usage | None = None,
        finish_reason: str | None = "stop",
        enter_error: Exception | None = None,
        exit_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.usage = usage or _Usage()
        self.messages = [_Message(finish_reason)]
        self.enter_error = enter_error
        self.exit_error = exit_error

    def __enter__(self) -> "_Chat":
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, *args: Any) -> None:
        del args
        if self.exit_error is not None:
            raise self.exit_error


class _Chats:
    def __init__(
        self,
        *,
        finish_reason: str | None = "stop",
        enter_error: Exception | None = None,
        exit_error: Exception | None = None,
    ) -> None:
        self.finish_reason = finish_reason
        self.enter_error = enter_error
        self.exit_error = exit_error
        self.names: list[str] = []

    def new(self, name: str) -> _Chat:
        self.names.append(name)
        return _Chat(
            name,
            finish_reason=self.finish_reason,
            enter_error=self.enter_error,
            exit_error=self.exit_error,
        )


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
        extra_api_params: dict[str, Any] | None = None,
    ) -> Any:
        del schema, tools, image, video, audio
        index = len(self.calls)
        self.calls.append(
            {
                "text": text,
                "seed": seed,
                "temperature": temperature,
                "extra_api_params": extra_api_params,
            }
        )
        output = self.outputs[index]
        if isinstance(output, BaseException):
            raise output
        return output


OpenAI.__module__ = "kaggle_benchmarks.actors.proxy_openai"


class _GenAIClient:
    def __init__(self, *, retry_options: Any = None) -> None:
        self._api_client = types.SimpleNamespace(
            _http_options=types.SimpleNamespace(retry_options=retry_options)
        )


class GoogleGenAI(OpenAI):
    def __init__(
        self,
        outputs: list[Any] | None = None,
        *,
        retry_options: Any = None,
    ) -> None:
        super().__init__(outputs=outputs)
        self.client = _GenAIClient(retry_options=retry_options)


GoogleGenAI.__module__ = "kaggle_benchmarks.actors.llms"


class _Bomb:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"unexpected access: {name}")


def _contains_score_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"score", "aurc", "ecl", "elicit", "rank"}:
                return True
            if _contains_score_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_score_key(child) for child in value)
    return False


class KaggleCaptureTaskV02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated, cls.previous_kaggle_module = _load_generated_task()
        cls.package_tmp = tempfile.TemporaryDirectory()
        cls.package_root = Path(cls.package_tmp.name) / "package"
        cls.package_root.mkdir()
        # Building the canonical package is intentionally an authority-runtime
        # operation. Portable capture tests stub only that already-tested gate.
        if sys.version_info[:2] == (3, 13):
            _write_package(cls.package_root)

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
        runtime: dict[str, str] = RUNTIME_311,
        package_root: Path | None = None,
        verify_package: bool = False,
    ) -> tuple[dict[str, Any], Path, tempfile.TemporaryDirectory[str]]:
        output_tmp: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        capture_path = Path(output_tmp.name) / "capture.json"
        package_gate = (
            mock.patch.object(self.generated, "_verify_package", return_value=None)
            if not verify_package
            else mock.patch.object(
                self.generated,
                "_verify_package",
                wraps=self.generated._verify_package,
            )
        )
        with package_gate:
            payload = self.generated.run_capture_canary(
                llm,
                chats=chats,
                package_root=package_root or self.package_root,
                capture_path=capture_path,
                observed_runtime=copy.deepcopy(runtime),
                clock=_Clock(),
            )
        self.assertEqual(load_capture_payload(capture_path), payload)
        self.assertEqual(verify_capture_payload(payload), payload)
        return payload, capture_path, output_tmp

    def test_generated_source_is_current_and_score_free(self) -> None:
        if sys.version_info[:2] == (3, 13):
            self.assertEqual(
                OUTPUT_PATH.read_text(encoding="utf-8"),
                capture_generator.render_task_source(),
            )
        source = OUTPUT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("from bench.engine.scoring", source)
        self.assertNotIn("import _scoring", source)
        self.assertNotIn("unicodedata.normalize", source)
        wrapper = self.generated.aleph_bench_v0_2_capture_canary
        self.assertEqual(wrapper.metadata["name"], self.generated.TASK_NAME)
        self.assertEqual(len(wrapper.run_calls), 1)

    def test_complete_six_call_capture_is_contract_valid(self) -> None:
        llm = OpenAI()
        chats = _Chats(finish_reason="stop")
        payload, _, output_tmp = self._run(llm=llm, chats=chats)
        self.addCleanup(output_tmp.cleanup)

        self.assertTrue(payload["captureComplete"])
        self.assertTrue(payload["canonicalReplayEligible"])
        self.assertEqual(len(payload["rows"]), 6)
        self.assertEqual(payload["calls"]["attemptedCallCount"], 6)
        self.assertEqual(payload["calls"]["terminalCallCount"], 6)
        self.assertEqual(chats.names, [row["conversationName"] for row in payload["rows"]])
        self.assertFalse(_contains_score_key(payload))
        self.assertIsNone(payload["requestPolicy"]["reasoning"])
        self.assertNotIn("reasoning=REQUEST_POLICY", OUTPUT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(all(call["seed"] == 0 for call in llm.calls))
        self.assertTrue(all(call["temperature"] == 0 for call in llm.calls))
        self.assertTrue(
            all(call["extra_api_params"] == {"max_tokens": 2048} for call in llm.calls)
        )
        self.assertEqual(llm.client.max_retries, 0)

    def test_provider_revision_model_slug_is_preserved(self) -> None:
        llm = OpenAI()
        llm.model = "anthropic/claude-haiku-4-5@20251001"

        payload, _, output_tmp = self._run(llm=llm, chats=_Chats())
        self.addCleanup(output_tmp.cleanup)

        self.assertTrue(payload["captureComplete"])
        self.assertEqual(
            payload["modelObservation"]["slug"],
            "anthropic/claude-haiku-4-5@20251001",
        )

    def test_malformed_provider_revision_model_slug_fails_before_dispatch(self) -> None:
        llm = OpenAI()
        llm.model = "anthropic/claude-haiku-4-5@@20251001"

        payload, _, output_tmp = self._run(llm=llm, chats=_Chats())
        self.addCleanup(output_tmp.cleanup)

        self.assertFalse(payload["captureComplete"])
        self.assertEqual(payload["calls"]["attemptedCallCount"], 0)
        self.assertEqual(llm.calls, [])

    @unittest.skipUnless(
        sys.version_info[:2] == (3, 13),
        "canonical package construction requires the authority runtime",
    )
    def test_frozen_package_verifies_on_authority_runtime(self) -> None:
        payload, _, output_tmp = self._run(
            llm=OpenAI(),
            chats=_Chats(),
            runtime=RUNTIME_313,
            verify_package=True,
        )
        self.addCleanup(output_tmp.cleanup)
        self.assertTrue(payload["captureComplete"])

    @unittest.skipUnless(
        sys.version_info[:2] == (3, 13),
        "canonical package construction requires the authority runtime",
    )
    def test_package_resolver_accepts_fully_qualified_kaggle_layout(self) -> None:
        input_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(input_tmp.cleanup)
        input_root = Path(input_tmp.name)
        package_root = (
            input_root
            / "datasets"
            / "owner"
            / self.generated.PACKAGE_SLUG
        )
        package_root.mkdir(parents=True)
        _write_package(package_root)

        resolved = self.generated._resolve_and_verify_package(
            None, input_root=input_root
        )

        self.assertEqual(resolved, package_root)

    @unittest.skipUnless(
        sys.version_info[:2] == (3, 13),
        "canonical package construction requires the authority runtime",
    )
    def test_package_resolver_accepts_flat_kaggle_layout(self) -> None:
        input_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(input_tmp.cleanup)
        input_root = Path(input_tmp.name)
        package_root = input_root / self.generated.PACKAGE_SLUG
        package_root.mkdir()
        _write_package(package_root)

        resolved = self.generated._resolve_and_verify_package(
            None, input_root=input_root
        )

        self.assertEqual(resolved, package_root)

    @unittest.skipUnless(
        sys.version_info[:2] == (3, 13),
        "canonical package construction requires the authority runtime",
    )
    def test_package_resolver_rejects_ambiguous_exact_mounts(self) -> None:
        input_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(input_tmp.cleanup)
        input_root = Path(input_tmp.name)
        package_roots = (
            input_root / self.generated.PACKAGE_SLUG,
            input_root / "datasets" / "owner" / self.generated.PACKAGE_SLUG,
        )
        for package_root in package_roots:
            package_root.mkdir(parents=True)
            _write_package(package_root)

        with self.assertRaisesRegex(ValueError, "found 2"):
            self.generated._resolve_and_verify_package(None, input_root=input_root)

    def test_package_resolver_bounds_kaggle_directory_inspection(self) -> None:
        input_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(input_tmp.cleanup)
        input_root = Path(input_tmp.name)
        for index in range(self.generated.MAX_KAGGLE_INPUT_ENTRIES + 1):
            (input_root / f"source-{index:03d}").mkdir()

        with self.assertRaisesRegex(ValueError, "entry limit exceeded"):
            self.generated._resolve_and_verify_package(None, input_root=input_root)

    def test_package_resolver_retains_first_bounded_validation_error(self) -> None:
        candidates = [Path("candidate-one"), Path("candidate-two")]
        first_reason = "first-invalid-package-" + "x" * 300

        def reject(candidate: Path) -> None:
            if candidate == candidates[0]:
                raise ValueError(first_reason)
            raise ValueError("second-invalid-package")

        with (
            mock.patch.object(
                self.generated,
                "_kaggle_package_candidates",
                return_value=candidates,
            ),
            mock.patch.object(self.generated, "_verify_package", side_effect=reject),
            self.assertRaises(ValueError) as raised,
        ):
            self.generated._resolve_and_verify_package(None)

        expected_reason = first_reason[
            : self.generated.MAX_PACKAGE_DISCOVERY_REASON_CODEPOINTS
        ]
        self.assertEqual(
            str(raised.exception),
            "expected exactly one hash-verified package mount in supported "
            "Kaggle layouts; found 0 across 2 candidates; "
            f"first validation error: {expected_reason}",
        )
        self.assertNotIn("second-invalid-package", str(raised.exception))

    def test_package_resolver_does_not_invent_an_absence_reason(self) -> None:
        with (
            mock.patch.object(
                self.generated,
                "_kaggle_package_candidates",
                return_value=[Path("missing-package")],
            ),
            mock.patch.object(
                self.generated,
                "_verify_package",
                side_effect=FileNotFoundError("not mounted"),
            ),
            self.assertRaises(ValueError) as raised,
        ):
            self.generated._resolve_and_verify_package(None)

        self.assertEqual(
            str(raised.exception),
            "expected exactly one hash-verified package mount in supported "
            "Kaggle layouts; found 0 across 1 candidates",
        )

    def test_current_sdk_missing_finish_reason_is_explicit_and_replayable(self) -> None:
        observations = []
        temporary_directories = []
        for runtime in (RUNTIME_311, RUNTIME_313):
            payload, _, output_tmp = self._run(
                llm=OpenAI(), chats=_Chats(finish_reason=None), runtime=runtime
            )
            temporary_directories.append(output_tmp)
            observations.append(payload)
            self.assertTrue(payload["captureComplete"])
            self.assertTrue(payload["canonicalReplayEligible"])
            self.assertEqual(
                payload["diagnostics"]["missingFinishReasonRowIds"],
                [row["rowId"] for row in payload["rows"]],
            )
            self.assertEqual(payload["diagnostics"]["missingUsageRowIds"], [])
            self.assertEqual(payload["diagnostics"]["replayBlockedReasons"], [])
        for output_tmp in temporary_directories:
            self.addCleanup(output_tmp.cleanup)
        left, right = observations
        for field in ("rows", "calls", "diagnostics", "requestPolicy", "shard"):
            self.assertEqual(left[field], right[field])
        self.assertNotEqual(left["runtimeObservation"], right["runtimeObservation"])

    def test_raw_outputs_are_preserved_without_text_interpretation(self) -> None:
        outputs = [
            "  leading and trailing\t",
            "line one\r\nline two\rlast\n",
            "é|e\u0301|Straße|STRASSE",
            "👩\u200d💻",
            "a\0b\x1f",
            "",
        ]
        payload, _, output_tmp = self._run(
            llm=OpenAI(outputs=outputs), chats=_Chats()
        )
        self.addCleanup(output_tmp.cleanup)
        self.assertEqual(
            [row["rawOutput"] for row in payload["rows"]],
            outputs,
        )
        self.assertTrue(payload["canonicalReplayEligible"])

    @unittest.skipUnless(
        sys.version_info[:2] == (3, 13),
        "canonical package construction requires the authority runtime",
    )
    def test_package_drift_blocks_before_model_or_chat_access(self) -> None:
        package_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(package_tmp.cleanup)
        package_root = Path(package_tmp.name) / "package"
        package_root.mkdir()
        _write_package(package_root)
        readme = package_root / "README.md"
        readme.write_bytes(readme.read_bytes() + b"drift")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            payload, _, output_tmp = self._run(
                llm=_Bomb(),
                chats=_Bomb(),
                package_root=package_root,
                verify_package=True,
            )
        self.addCleanup(output_tmp.cleanup)
        self.assertEqual(payload["rows"], [])
        self.assertEqual(payload["calls"]["attemptedCallCount"], 0)
        self.assertFalse(payload["captureComplete"])
        self.assertIn("stage=package", stdout.getvalue())
        self.assertIn("package artifact size mismatch", stdout.getvalue())

    def test_chat_open_model_call_and_close_failures_are_distinct(self) -> None:
        cases = [
            (
                OpenAI(),
                _Chats(enter_error=RuntimeError("open secret")),
                "chatOpenFailed",
                False,
                None,
            ),
            (
                OpenAI(outputs=[RuntimeError("model secret")]),
                _Chats(),
                "modelCallFailed",
                True,
                None,
            ),
            (
                OpenAI(),
                _Chats(exit_error=RuntimeError("close secret")),
                None,
                True,
                "chatCloseFailed",
            ),
        ]
        for llm, chats, terminal, dispatched, lifecycle in cases:
            with self.subTest(terminal=terminal, lifecycle=lifecycle):
                payload, _, output_tmp = self._run(llm=llm, chats=chats)
                self.addCleanup(output_tmp.cleanup)
                row = payload["rows"][0]
                self.assertEqual(row["dispatchStarted"], dispatched)
                self.assertEqual(
                    row["terminalFailure"]["failureKind"]
                    if row["terminalFailure"] is not None
                    else None,
                    terminal,
                )
                self.assertEqual(
                    row["lifecycleFailure"]["failureKind"]
                    if row["lifecycleFailure"] is not None
                    else None,
                    lifecycle,
                )
                serialized = json.dumps(payload, ensure_ascii=True)
                self.assertNotIn("open secret", serialized)
                self.assertNotIn("model secret", serialized)
                self.assertNotIn("close secret", serialized)

    def test_genai_default_no_retry_contract_does_not_require_private_controller(self) -> None:
        llm = GoogleGenAI()
        payload, _, output_tmp = self._run(llm=llm, chats=_Chats())
        self.addCleanup(output_tmp.cleanup)

        self.assertTrue(payload["captureComplete"])
        self.assertEqual(payload["calls"]["attemptedCallCount"], 6)
        self.assertEqual(
            llm.calls[0]["extra_api_params"], {"max_output_tokens": 2048}
        )

    def test_genai_explicit_retry_options_block_before_model_call(self) -> None:
        llm = GoogleGenAI(retry_options=object())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            payload, _, output_tmp = self._run(llm=llm, chats=_Chats())
        self.addCleanup(output_tmp.cleanup)

        self.assertFalse(payload["captureComplete"])
        self.assertEqual(payload["calls"]["attemptedCallCount"], 0)
        self.assertEqual(llm.calls, [])
        self.assertIn("stage=model", stdout.getvalue())
        self.assertIn("transportRetryPolicyUnverified", stdout.getvalue())

    def test_non_string_oversized_and_lone_surrogate_outputs_fail_closed(self) -> None:
        cases = [
            (None, "nonStringOutput", False),
            ("x" * 65_537, "oversizedOutput", False),
            ("raw\ud800evidence", "returnedString", False),
        ]
        for output, outcome, eligible in cases:
            with self.subTest(outcome=outcome):
                payload, _, output_tmp = self._run(
                    llm=OpenAI(outputs=[output]), chats=_Chats()
                )
                self.addCleanup(output_tmp.cleanup)
                self.assertEqual(payload["rows"][0]["outcome"], outcome)
                self.assertEqual(payload["canonicalReplayEligible"], eligible)
                if outcome == "oversizedOutput":
                    self.assertNotIn("rawOutput", payload["rows"][0])
                if outcome == "returnedString":
                    self.assertEqual(payload["rows"][0]["rawOutput"], output)

    def test_unrepresentable_surrogate_pair_keeps_dispatching_checkpoint(self) -> None:
        output_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(output_tmp.cleanup)
        capture_path = Path(output_tmp.name) / "capture.json"
        with self.assertRaisesRegex(ValueError, "surrogate pair"):
            with mock.patch.object(
                self.generated, "_verify_package", return_value=None
            ):
                self.generated.run_capture_canary(
                    OpenAI(outputs=["\ud83d\ude00"]),
                    chats=_Chats(),
                    package_root=self.package_root,
                    capture_path=capture_path,
                    observed_runtime=RUNTIME_311,
                    clock=_Clock(),
                )
        payload = load_capture_payload(capture_path)
        self.assertEqual(payload["rows"], [])
        self.assertEqual(payload["calls"]["attemptedCallCount"], 1)
        self.assertEqual(payload["calls"]["activeCall"]["state"], "dispatching")

    def test_atomic_replace_failure_preserves_previous_checkpoint(self) -> None:
        payload, capture_path, output_tmp = self._run(
            llm=OpenAI(), chats=_Chats()
        )
        self.addCleanup(output_tmp.cleanup)
        before = capture_path.read_bytes()
        with mock.patch.object(self.generated.os, "replace", side_effect=OSError("stop")):
            with self.assertRaisesRegex(OSError, "stop"):
                self.generated._atomic_checkpoint(payload, capture_path)
        self.assertEqual(capture_path.read_bytes(), before)
        self.assertEqual(list(capture_path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
