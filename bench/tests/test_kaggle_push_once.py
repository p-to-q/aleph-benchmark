from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from bench.engine import kaggle_push_once as push_once
from bench.engine.kaggle_push_once import (
    KagglePushOnceError,
    KagglePushOutcomeAmbiguous,
    _normalize_response,
    _is_http_not_found,
    dispatch_create_once,
    preflight_and_dispatch_once,
    push_capture_once,
    push_diagnostic_once,
)


def _task_response(
    *,
    task: str = "aleph-bench-deployment-diagnostic-2048-none",
    version: int = 2,
    source_kernel_id: int = 12345,
    datasets: tuple[str, ...] = (),
    state: str = "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED",
) -> SimpleNamespace:
    return SimpleNamespace(
        slug=SimpleNamespace(
            owner_slug="owner",
            task_slug=task,
            version_number=version,
        ),
        source_kernel_id=source_kernel_id,
        creation_state=state,
        error=None,
        url=(
            "https://www.kaggle.com/benchmarks/owner/"
            f"{task}/{version}"
        ),
        options=SimpleNamespace(dataset_data_sources=list(datasets)),
    )


class HardStop(BaseException):
    """A synthetic process-level interruption, not a normal Exception."""


class KagglePushOnceTests(unittest.TestCase):
    def test_only_http_404_means_remote_task_is_missing(self) -> None:
        for status_code in (401, 403, 429, 500, None):
            with self.subTest(status_code=status_code):
                error = RuntimeError("remote preflight failed")
                error.response = SimpleNamespace(status_code=status_code)  # type: ignore[attr-defined]
                self.assertFalse(_is_http_not_found(error))

        missing = RuntimeError("not found")
        missing.response = SimpleNamespace(status_code=404)  # type: ignore[attr-defined]
        self.assertTrue(_is_http_not_found(missing))

    def test_public_preflights_propagate_http_403_without_dispatch(self) -> None:
        class HTTPError(Exception):
            pass

        class Request:
            pass

        class Options:
            pass

        forbidden = HTTPError("forbidden")
        forbidden.response = SimpleNamespace(status_code=403)  # type: ignore[attr-defined]
        task_client = SimpleNamespace(
            get_benchmark_task=mock.Mock(side_effect=forbidden),
            create_benchmark_task=mock.Mock(),
        )
        client = SimpleNamespace(
            benchmarks=SimpleNamespace(
                benchmark_tasks_api_client=task_client
            )
        )
        context = mock.MagicMock()
        context.__enter__.return_value = client
        api = mock.Mock()
        api.build_kaggle_client.return_value = context
        api.with_retry.side_effect = lambda function: function

        modules: dict[str, ModuleType] = {}
        for name in (
            "kaggle",
            "kaggle.api",
            "kaggle.api.kaggle_api_extended",
            "kagglesdk",
            "kagglesdk.benchmarks",
            "kagglesdk.benchmarks.types",
            "kagglesdk.benchmarks.types.benchmark_types",
            "kagglesdk.benchmarks.types.benchmark_tasks_api_service",
            "requests",
            "requests.exceptions",
        ):
            modules[name] = ModuleType(name)
        modules["kaggle.api.kaggle_api_extended"].KaggleApi = mock.Mock(
            return_value=api
        )
        modules[
            "kagglesdk.benchmarks.types.benchmark_types"
        ].BenchmarkTaskOptions = Options
        task_types = modules[
            "kagglesdk.benchmarks.types.benchmark_tasks_api_service"
        ]
        task_types.ApiBenchmarkTaskSlug = Request
        task_types.ApiCreateBenchmarkTaskRequest = Request
        task_types.ApiGetBenchmarkTaskRequest = Request
        modules["requests.exceptions"].HTTPError = HTTPError

        cases = (
            (
                "diagnostic",
                push_diagnostic_once,
                {
                    "datasets": (),
                    "gate": "zero-call",
                },
            ),
            (
                "capture",
                push_capture_once,
                {
                    "datasets": ("owner/capture-package",),
                },
            ),
        )
        for name, caller, arguments in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as tmp,
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    push_once, "_verified_client_versions", return_value={}
                ),
                mock.patch.object(
                    push_once,
                    "_source_and_notebook",
                    return_value=(b"# source\n", "# notebook"),
                ),
                mock.patch.object(
                    push_once,
                    "_exact_source_and_notebook",
                    return_value=(b"# source\n", "# notebook"),
                ),
                mock.patch.object(
                    push_once,
                    "render_capture_task_source",
                    return_value="# source\n",
                ),
                mock.patch.object(push_once, "dispatch_create_once") as dispatch,
                self.assertRaisesRegex(HTTPError, "forbidden"),
            ):
                caller(
                    source_path=Path(tmp) / f"{name}.py",
                    journal_path=Path(tmp) / "push.json",
                    **arguments,
                )
            dispatch.assert_not_called()
            task_client.create_benchmark_task.assert_not_called()

    def test_capture_task_identity_is_supported_without_weakening_diagnostic_default(self) -> None:
        response = _task_response(
            task=push_once.CAPTURE_TASK_SLUG,
            datasets=("owner/capture-package",),
        )
        observed = _normalize_response(
            response,
            expected_datasets=("owner/capture-package",),
            expected_task=push_once.CAPTURE_TASK_SLUG,
        )

        self.assertEqual(observed["task"], push_once.CAPTURE_TASK_SLUG)
        with self.assertRaisesRegex(KagglePushOnceError, "expected"):
            _normalize_response(
                response,
                expected_datasets=("owner/capture-package",),
            )

    def test_capture_requires_one_reviewed_dataset_before_client_or_source(self) -> None:
        for datasets in ((), ("owner/one", "owner/two")):
            with self.subTest(datasets=datasets), tempfile.TemporaryDirectory() as tmp:
                with (
                    mock.patch.object(push_once, "_verified_client_versions") as clients,
                    mock.patch.object(push_once, "_exact_source_and_notebook") as source,
                    self.assertRaisesRegex(KagglePushOnceError, "exactly one"),
                ):
                    push_capture_once(
                        source_path=Path(tmp) / "capture.py",
                        datasets=datasets,
                        journal_path=Path(tmp) / "push.json",
                    )
                clients.assert_not_called()
                source.assert_not_called()

    def test_capture_cli_routes_default_source_to_capture_generator(self) -> None:
        journal = {
            "response": {
                "task": push_once.CAPTURE_TASK_SLUG,
                "version": 1,
                "sourceKernelId": None,
            }
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            push_once, "push_capture_once", return_value=journal
        ) as submit:
            status = push_once.main(
                [
                    "--task-kind",
                    "capture",
                    "--gate",
                    "six-call",
                    "--dataset",
                    "owner/capture-package",
                    "--journal",
                    str(Path(tmp) / "push.json"),
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(
            submit.call_args.kwargs["source_path"], push_once.CAPTURE_OUTPUT_PATH
        )
        self.assertEqual(
            submit.call_args.kwargs["datasets"], ("owner/capture-package",)
        )

    def test_client_version_mismatch_blocks_before_remote_read_or_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "push.json"
            with (
                mock.patch.object(
                    push_once,
                    "_distribution_version",
                    return_value="unreviewed",
                ),
                mock.patch.object(
                    push_once, "_source_and_notebook"
                ) as source_and_notebook,
                mock.patch.object(
                    push_once, "preflight_and_dispatch_once"
                ) as remote_preflight,
                self.assertRaisesRegex(
                    KagglePushOnceError, "no remote request was made"
                ),
            ):
                push_diagnostic_once(
                    source_path=Path(tmp) / "diagnostic.py",
                    datasets=(),
                    gate="zero-call",
                    journal_path=journal_path,
                )

            source_and_notebook.assert_not_called()
            remote_preflight.assert_not_called()
            self.assertFalse(journal_path.exists())

    def test_pending_remote_creation_blocks_before_journal_and_create(self) -> None:
        create_calls = 0

        def create(_request: object) -> object:
            nonlocal create_calls
            create_calls += 1
            return _task_response()

        pending_states = (
            "BENCHMARK_TASK_VERSION_CREATION_STATE_QUEUED",
            "BENCHMARK_TASK_VERSION_CREATION_STATE_RUNNING",
            "BENCHMARK_TASK_VERSION_CREATION_STATE_UNSPECIFIED",
        )
        for state in pending_states:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                journal_path = Path(tmp) / "push.json"
                with self.assertRaisesRegex(
                    KagglePushOnceError, "pending or has an unknown state"
                ):
                    preflight_and_dispatch_once(
                        fetch_latest_task=lambda: _task_response(state=state),
                        create_task=create,
                        request=object(),
                        journal_path=journal_path,
                        prepared_journal={"operationId": "test-operation"},
                        response_record=lambda response: response,
                    )
                self.assertFalse(journal_path.exists())
        self.assertEqual(create_calls, 0)

    def test_terminal_remote_preflight_is_bound_into_journal(self) -> None:
        calls: list[object] = []
        prior = _task_response(
            version=1,
            source_kernel_id=111,
            datasets=("owner/prior-dataset",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            journal = preflight_and_dispatch_once(
                fetch_latest_task=lambda: prior,
                create_task=(
                    lambda request: calls.append(request) or _task_response()
                ),
                request=object(),
                journal_path=Path(tmp) / "push.json",
                prepared_journal={"operationId": "test-operation"},
                response_record=lambda response: _normalize_response(
                    response, expected_datasets=()
                ),
                clock=lambda: "2026-09-18T00:00:00Z",
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(
                journal["remotePreflight"]["priorTask"],
                {
                    "owner": "owner",
                    "task": "aleph-bench-deployment-diagnostic-2048-none",
                    "version": 1,
                    "creationState": (
                        "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED"
                    ),
                    "sourceKernelId": 111,
                    "datasets": ["owner/prior-dataset"],
                },
            )

    def test_malformed_prior_kernel_identity_blocks_before_create(self) -> None:
        calls = 0

        def create(_request: object) -> object:
            nonlocal calls
            calls += 1
            return _task_response()

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            KagglePushOnceError, "invalid source kernel ID"
        ):
            preflight_and_dispatch_once(
                fetch_latest_task=lambda: _task_response(
                    source_kernel_id=False
                ),
                create_task=create,
                request=object(),
                journal_path=Path(tmp) / "push.json",
                prepared_journal={"operationId": "test-operation"},
                response_record=lambda response: response,
            )

        self.assertEqual(calls, 0)

    def test_success_dispatches_exactly_once_and_records_server_identity(self) -> None:
        calls: list[object] = []
        prepared = {"operationId": "test-operation", "task": "diagnostic"}
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "push.json"
            journal = dispatch_create_once(
                create_task=(
                    lambda request: calls.append(request) or _task_response()
                ),
                request=object(),
                journal_path=journal_path,
                prepared_journal=prepared,
                response_record=lambda response: _normalize_response(
                    response, expected_datasets=()
                ),
                clock=lambda: "2026-09-18T00:00:00Z",
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(journal["state"], "returned")
            self.assertEqual(journal["response"]["version"], 2)
            self.assertEqual(journal["response"]["sourceKernelId"], 12345)
            self.assertEqual(
                json.loads(journal_path.read_text(encoding="utf-8")), journal
            )

    def test_acknowledged_version_may_defer_source_kernel_identity(self) -> None:
        calls = 0

        def create(_request: object) -> object:
            nonlocal calls
            calls += 1
            return _task_response(source_kernel_id=0)

        with tempfile.TemporaryDirectory() as tmp:
            journal = dispatch_create_once(
                create_task=create,
                request=object(),
                journal_path=Path(tmp) / "push.json",
                prepared_journal={"operationId": "test-operation"},
                response_record=lambda response: _normalize_response(
                    response, expected_datasets=()
                ),
                clock=lambda: "2026-09-18T00:00:00Z",
            )

        self.assertEqual(calls, 1)
        self.assertEqual(journal["state"], "returned")
        self.assertEqual(journal["response"]["task"], push_once.TASK_SLUG)
        self.assertEqual(journal["response"]["version"], 2)
        self.assertIsNone(journal["response"]["sourceKernelId"])

    def test_malformed_source_kernel_identity_is_not_treated_as_missing(self) -> None:
        for value in (False, 0.0, -1, "12345"):
            with self.subTest(value=value), self.assertRaisesRegex(
                KagglePushOnceError, "invalid source kernel ID"
            ):
                _normalize_response(
                    _task_response(source_kernel_id=value),
                    expected_datasets=(),
                )

    def test_connection_failure_is_ambiguous_without_retry(self) -> None:
        calls = 0

        def fail(_request: object) -> object:
            nonlocal calls
            calls += 1
            raise ConnectionError("response lost")

        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "push.json"
            with self.assertRaisesRegex(
                KagglePushOutcomeAmbiguous, "do not retry"
            ):
                dispatch_create_once(
                    create_task=fail,
                    request=object(),
                    journal_path=journal_path,
                    prepared_journal={"operationId": "test-operation"},
                    response_record=lambda response: response,
                    clock=lambda: "2026-09-18T00:00:00Z",
                )

            self.assertEqual(calls, 1)
            retained = json.loads(journal_path.read_text(encoding="utf-8"))
            self.assertEqual(retained["state"], "ambiguous")
            self.assertEqual(retained["failure"]["type"], "ConnectionError")

    def test_hard_stop_is_journaled_and_propagated_without_retry(self) -> None:
        calls = 0

        def stop(_request: object) -> object:
            nonlocal calls
            calls += 1
            raise HardStop("process interrupted")

        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "push.json"
            with self.assertRaisesRegex(HardStop, "process interrupted"):
                dispatch_create_once(
                    create_task=stop,
                    request=object(),
                    journal_path=journal_path,
                    prepared_journal={"operationId": "test-operation"},
                    response_record=lambda response: response,
                    clock=lambda: "2026-09-18T00:00:00Z",
                )

            self.assertEqual(calls, 1)
            retained = json.loads(journal_path.read_text(encoding="utf-8"))
            self.assertEqual(retained["state"], "ambiguous")
            self.assertEqual(retained["failure"]["type"], "HardStop")

    def test_existing_journal_blocks_before_dispatch(self) -> None:
        calls = 0

        def create(_request: object) -> object:
            nonlocal calls
            calls += 1
            return _task_response()

        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "push.json"
            journal_path.write_text("existing evidence\n", encoding="utf-8")
            with self.assertRaisesRegex(KagglePushOnceError, "already exists"):
                dispatch_create_once(
                    create_task=create,
                    request=object(),
                    journal_path=journal_path,
                    prepared_journal={"operationId": "test-operation"},
                    response_record=lambda response: response,
                )

            self.assertEqual(calls, 0)
            self.assertEqual(
                journal_path.read_text(encoding="utf-8"), "existing evidence\n"
            )

    def test_invalid_server_response_is_ambiguous_without_retry(self) -> None:
        calls = 0

        def create(_request: object) -> object:
            nonlocal calls
            calls += 1
            return _task_response(task="wrong-task")

        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "push.json"
            with self.assertRaisesRegex(
                KagglePushOutcomeAmbiguous, "do not retry"
            ):
                dispatch_create_once(
                    create_task=create,
                    request=object(),
                    journal_path=journal_path,
                    prepared_journal={"operationId": "test-operation"},
                    response_record=lambda response: _normalize_response(
                        response, expected_datasets=()
                    ),
                    clock=lambda: "2026-09-18T00:00:00Z",
                )

            self.assertEqual(calls, 1)
            retained = json.loads(journal_path.read_text(encoding="utf-8"))
            self.assertEqual(retained["state"], "ambiguous")
            self.assertEqual(retained["failure"]["type"], "KagglePushOnceError")

    def test_source_mismatch_blocks_before_journal_or_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "diagnostic.py"
            source_path.write_text("# stale diagnostic\n", encoding="utf-8")
            journal_path = Path(tmp) / "push.json"

            with (
                mock.patch.object(
                    push_once,
                    "_verified_client_versions",
                    return_value={
                        "python": "3.13.2",
                        "kaggle": "2.2.4",
                        "kagglesdk": "0.1.37",
                        "jupytext": "1.19.5",
                    },
                ),
                self.assertRaisesRegex(
                    KagglePushOnceError, "current checked-in generated"
                ),
            ):
                push_diagnostic_once(
                    source_path=source_path,
                    datasets=(),
                    gate="zero-call",
                    journal_path=journal_path,
                )

            self.assertFalse(journal_path.exists())

    def test_invalid_gate_and_dataset_block_before_source_or_dispatch(self) -> None:
        cases = {
            "gate": ("unknown", (), "unknown diagnostic gate"),
            "dataset-shape": (
                "six-call",
                ("not-an-owner-dataset",),
                "OWNER/DATASET",
            ),
            "zero-call-attachment": (
                "zero-call",
                ("owner/dataset",),
                "must not attach",
            ),
            "six-call-count": ("six-call", (), "exactly one"),
        }
        for name, (gate, datasets, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                journal_path = Path(tmp) / "push.json"
                with self.assertRaisesRegex(KagglePushOnceError, message):
                    push_diagnostic_once(
                        source_path=Path(tmp) / "missing.py",
                        datasets=datasets,
                        gate=gate,
                        journal_path=journal_path,
                    )
                self.assertFalse(journal_path.exists())


if __name__ == "__main__":
    unittest.main()
