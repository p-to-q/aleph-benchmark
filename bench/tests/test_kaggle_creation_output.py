from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench.engine import kaggle_creation_output as creation_output
from bench.engine.kaggle_creation_output import (
    KaggleCreationOutputError,
    RECEIPT_FILENAME,
    _read_bounded_download,
    _run_metadata,
    _select_exact_run,
    _task_metadata,
    _validate_gate_datasets,
    write_creation_bundle,
)


COMPLETED = "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED"


class HardStop(BaseException):
    """Synthetic process-level interruption used to verify bundle cleanup."""


def _task_info(
    *,
    owner: str = "owner",
    task: str = "aleph-bench-deployment-diagnostic-2048-none",
    version: int = 2,
    kernel_id: int = 12345,
    datasets: tuple[str, ...] = (),
    state: str = COMPLETED,
) -> SimpleNamespace:
    return SimpleNamespace(
        slug=SimpleNamespace(
            owner_slug=owner,
            task_slug=task,
            version_number=version,
        ),
        creation_state=state,
        creation_error_message=None,
        error=None,
        source_kernel_id=kernel_id,
        options=SimpleNamespace(dataset_data_sources=list(datasets)),
        create_time=datetime(2026, 9, 18, tzinfo=timezone.utc),
        url=f"https://www.kaggle.com/benchmarks/{owner}/{task}/{version}",
    )


def _run_info(
    *,
    owner: str = "owner",
    task: str = "aleph-bench-deployment-diagnostic-2048-none",
    version: int = 2,
    run_id: int = 24680,
    model: str = "google/gemini-test",
    state: str = "BENCHMARK_TASK_RUN_STATE_COMPLETED",
) -> SimpleNamespace:
    return SimpleNamespace(
        task_slug=SimpleNamespace(
            owner_slug=owner,
            task_slug=task,
            version_number=version,
        ),
        id=run_id,
        model_version_slug=model,
        state=state,
        error_message=None,
        start_time=datetime(2026, 9, 18, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 9, 18, 1, 1, tzinfo=timezone.utc),
    )


def _archive(
    receipt_bytes: bytes = b"{}", *, receipt_name: str = RECEIPT_FILENAME
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(receipt_name, receipt_bytes)
        archive.writestr("creation.log", "complete\n")
    return output.getvalue()


def _receipt(*, complete: bool) -> dict[str, object]:
    attempted = 6 if complete else 0
    return {
        "id": "receipt-id",
        "status": "complete" if complete else "blocked",
        "phase": "complete" if complete else "package_preflight",
        "calls": {
            "attempted": attempted,
            "completed": attempted,
            "activeCall": None,
        },
        "rows": [{} for _ in range(attempted)],
    }


class _SdkSlug:
    pass


class _SdkGetRequest:
    pass


class _SdkListRequest:
    pass


class _SdkDownloadRequest:
    pass


class _SdkHarness:
    """Small fake for the exact benchmark-task SDK surface we permit."""

    def __init__(
        self,
        *,
        task_info: SimpleNamespace,
        pages: list[SimpleNamespace],
        archive_bytes: bytes = b"creation-archive",
    ) -> None:
        self.task_info = task_info
        self.pages = list(pages)
        self.get_requests: list[tuple[str | None, str, int]] = []
        self.list_requests: list[tuple[str | None, str, int, int, str]] = []
        self.download_requests: list[tuple[int, bool]] = []
        self.authenticate = mock.Mock()
        self.download_response = SimpleNamespace(
            headers={"Content-Length": str(len(archive_bytes))},
            raise_for_status=mock.Mock(),
            iter_content=mock.Mock(return_value=iter((archive_bytes,))),
            close=mock.Mock(),
        )
        task_client = SimpleNamespace(
            get_benchmark_task=self._get_task,
            list_benchmark_task_runs=self._list_runs,
            download_benchmark_task_run_output=self._download,
        )
        # Expose only the benchmark-task client. A regression to a kernel or
        # schedule endpoint therefore fails this test immediately.
        client = SimpleNamespace(
            benchmarks=SimpleNamespace(
                benchmark_tasks_api_client=task_client
            )
        )
        self.api = SimpleNamespace(
            authenticate=self.authenticate,
            build_kaggle_client=mock.Mock(return_value=nullcontext(client)),
            with_retry=lambda function: function,
        )

    def sdk(self) -> tuple[object, object, object, object, object]:
        return (
            lambda: self.api,
            _SdkSlug,
            _SdkDownloadRequest,
            _SdkGetRequest,
            _SdkListRequest,
        )

    def _get_task(self, request: _SdkGetRequest) -> SimpleNamespace:
        slug = request.slug
        self.get_requests.append(
            (slug.owner_slug, slug.task_slug, slug.version_number)
        )
        return self.task_info

    def _list_runs(self, request: _SdkListRequest) -> SimpleNamespace:
        if not self.pages:
            raise AssertionError("unexpected task-run page request")
        slug = request.task_slug
        self.list_requests.append(
            (
                slug.owner_slug,
                slug.task_slug,
                slug.version_number,
                request.page_size,
                getattr(request, "page_token", ""),
            )
        )
        return self.pages.pop(0)

    def _download(self, request: _SdkDownloadRequest) -> SimpleNamespace:
        self.download_requests.append((request.run_id, request.include_source))
        return self.download_response


class KaggleCreationOutputTests(unittest.TestCase):
    def test_exclusive_write_removes_file_after_durability_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            failure = OSError("synthetic fsync failure")
            with (
                mock.patch.object(creation_output.os, "fsync", side_effect=failure),
                self.assertRaises(OSError) as raised,
            ):
                creation_output._write_exclusive(path, b"partial")

            self.assertIs(raised.exception, failure)
            self.assertFalse(path.exists())

    def test_exclusive_write_never_removes_preexisting_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_bytes(b"retained evidence")

            with self.assertRaises(FileExistsError):
                creation_output._write_exclusive(path, b"replacement")

            self.assertEqual(path.read_bytes(), b"retained evidence")

    def test_bundle_removes_prior_files_after_process_level_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            creation_output,
            "parse_kaggle_diagnostic_receipt_bytes",
            return_value=_receipt(complete=False),
        ):
            original_write = creation_output._write_exclusive
            interruption = HardStop("synthetic interruption")
            call_count = 0

            def interrupt_third_write(path: Path, data: bytes) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise interruption
                original_write(path, data)

            with (
                mock.patch.object(
                    creation_output,
                    "_write_exclusive",
                    side_effect=interrupt_third_write,
                ),
                self.assertRaises(HardStop) as raised,
            ):
                write_creation_bundle(
                    task_info=_task_info(),
                    run_info=_run_info(),
                    requested_task=(
                        "owner/aleph-bench-deployment-diagnostic-2048-none"
                    ),
                    expected_version=2,
                    expected_source_kernel_id=12345,
                    expected_run_id=24680,
                    expected_datasets=(),
                    archive_bytes=_archive(),
                    gate_mode="zero-call",
                    output_dir=Path(tmp),
                )

            self.assertIs(raised.exception, interruption)
            self.assertEqual(call_count, 3)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_gate_dataset_contract_is_fail_closed(self) -> None:
        _validate_gate_datasets("zero-call", ())
        _validate_gate_datasets("six-call", ("owner/dataset",))
        cases = {
            "zero-with-dataset": (
                "zero-call",
                ("owner/dataset",),
                "must expect no",
            ),
            "six-with-none": ("six-call", (), "exactly one"),
            "six-with-two": (
                "six-call",
                ("owner/one", "owner/two"),
                "exactly one",
            ),
            "unknown": ("other", (), "unknown gate"),
        }
        for name, (mode, datasets, message) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                KaggleCreationOutputError, message
            ):
                _validate_gate_datasets(mode, datasets)

    def test_stream_download_is_bounded_without_content_length(self) -> None:
        response = SimpleNamespace(
            headers={},
            raise_for_status=mock.Mock(),
            iter_content=mock.Mock(return_value=iter((b"abc", b"", b"def"))),
            close=mock.Mock(),
        )
        self.assertEqual(_read_bounded_download(response), b"abcdef")
        response.raise_for_status.assert_called_once_with()
        response.iter_content.assert_called_once_with(chunk_size=1_048_576)
        response.close.assert_called_once_with()

    def test_stream_download_rejects_oversize_before_buffering(self) -> None:
        response = SimpleNamespace(
            headers={},
            raise_for_status=mock.Mock(),
            iter_content=mock.Mock(return_value=iter((b"abc", b"d"))),
            close=mock.Mock(),
        )
        with mock.patch.object(creation_output, "MAX_ARCHIVE_BYTES", 3):
            with self.assertRaisesRegex(
                KaggleCreationOutputError, "archive safety limit"
            ):
                _read_bounded_download(response)
        response.close.assert_called_once_with()

    def test_stream_download_rejects_announced_size_before_read(self) -> None:
        response = SimpleNamespace(
            headers={"Content-Length": "4"},
            raise_for_status=mock.Mock(),
            iter_content=mock.Mock(return_value=iter((b"data",))),
            close=mock.Mock(),
        )
        with mock.patch.object(creation_output, "MAX_ARCHIVE_BYTES", 3):
            with self.assertRaisesRegex(
                KaggleCreationOutputError, "archive safety limit"
            ):
                _read_bounded_download(response)
        response.iter_content.assert_not_called()
        response.close.assert_called_once_with()

    def test_sdk_download_uses_exact_task_run_and_source_archive(self) -> None:
        unrelated_run = _run_info(version=1, run_id=11111)
        exact_run = _run_info()
        harness = _SdkHarness(
            task_info=_task_info(),
            pages=[
                SimpleNamespace(
                    runs=[unrelated_run], next_page_token="second-page"
                ),
                SimpleNamespace(runs=[exact_run], next_page_token=""),
            ],
        )
        with mock.patch.object(
            creation_output, "_load_kaggle_sdk", return_value=harness.sdk()
        ):
            task_info, run_info, archive_bytes = (
                creation_output._download_creation_archive(
                    "owner/aleph-bench-deployment-diagnostic-2048-none",
                    2,
                    24680,
                    12345,
                    (),
                )
            )

        self.assertIs(task_info, harness.task_info)
        self.assertIs(run_info, exact_run)
        self.assertEqual(archive_bytes, b"creation-archive")
        self.assertEqual(
            harness.get_requests,
            [("owner", "aleph-bench-deployment-diagnostic-2048-none", 2)],
        )
        self.assertEqual(
            harness.list_requests,
            [
                (
                    "owner",
                    "aleph-bench-deployment-diagnostic-2048-none",
                    2,
                    creation_output.MAX_RUNS_TO_INSPECT,
                    "",
                ),
                (
                    "owner",
                    "aleph-bench-deployment-diagnostic-2048-none",
                    2,
                    creation_output.MAX_RUNS_TO_INSPECT,
                    "second-page",
                ),
            ],
        )
        self.assertEqual(harness.download_requests, [(24680, True)])
        harness.authenticate.assert_called_once_with()
        harness.download_response.close.assert_called_once_with()

    def test_sdk_download_caps_distinct_empty_pages(self) -> None:
        harness = _SdkHarness(
            task_info=_task_info(),
            pages=[
                SimpleNamespace(runs=[], next_page_token="second-page"),
                SimpleNamespace(runs=[], next_page_token="third-page"),
            ],
        )
        with mock.patch.object(
            creation_output, "_load_kaggle_sdk", return_value=harness.sdk()
        ), mock.patch.object(creation_output, "MAX_RUN_PAGES_TO_INSPECT", 2):
            with self.assertRaisesRegex(
                KaggleCreationOutputError, "too many task-run pages"
            ):
                creation_output._download_creation_archive(
                    "owner/aleph-bench-deployment-diagnostic-2048-none",
                    2,
                    None,
                    12345,
                    (),
                )

        self.assertEqual(len(harness.list_requests), 2)
        self.assertEqual(harness.download_requests, [])

    def test_sdk_download_stops_before_fetch_on_identity_or_state_drift(
        self,
    ) -> None:
        cases = {
            "task-version": (
                _task_info(version=3),
                [SimpleNamespace(runs=[_run_info()], next_page_token="")],
                (),
                "task version",
            ),
            "dataset": (
                _task_info(datasets=("owner/wrong",)),
                [SimpleNamespace(runs=[_run_info()], next_page_token="")],
                ("owner/expected",),
                "datasets differ",
            ),
            "run-state": (
                _task_info(),
                [
                    SimpleNamespace(
                        runs=[
                            _run_info(
                                state="BENCHMARK_TASK_RUN_STATE_RUNNING"
                            )
                        ],
                        next_page_token="",
                    )
                ],
                (),
                "not terminal",
            ),
        }
        for name, (task_info, pages, datasets, message) in cases.items():
            with self.subTest(name=name):
                harness = _SdkHarness(task_info=task_info, pages=pages)
                with mock.patch.object(
                    creation_output,
                    "_load_kaggle_sdk",
                    return_value=harness.sdk(),
                ), self.assertRaisesRegex(KaggleCreationOutputError, message):
                    creation_output._download_creation_archive(
                        "owner/aleph-bench-deployment-diagnostic-2048-none",
                        2,
                        None,
                        12345,
                        datasets,
                    )
                self.assertEqual(harness.download_requests, [])

    def test_creation_run_must_be_unique_and_exact(self) -> None:
        requested = "owner/aleph-bench-deployment-diagnostic-2048-none"
        run = _run_info()
        self.assertIs(
            _select_exact_run(
                [run],
                requested_task=requested,
                expected_version=2,
                expected_run_id=24680,
            ),
            run,
        )
        second_run = _run_info(run_id=24681)
        self.assertIs(
            _select_exact_run(
                [run, second_run],
                requested_task=requested,
                expected_version=2,
                expected_run_id=24681,
            ),
            second_run,
        )
        cases = {
            "missing": ([], None, "found 0"),
            "multiple": ([run, _run_info(run_id=24681)], None, "found 2"),
            "wrong-id": ([run], 999, "run ID 999.*found 0"),
            "duplicate-id": ([run, _run_info()], 24680, "run ID 24680.*found 2"),
        }
        for name, (runs, run_id, message) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                KaggleCreationOutputError, message
            ):
                _select_exact_run(
                    runs,
                    requested_task=requested,
                    expected_version=2,
                    expected_run_id=run_id,
                )

        with self.assertRaisesRegex(KaggleCreationOutputError, "found 0"):
            _select_exact_run(
                [_run_info(version=True)],
                requested_task=requested,
                expected_version=1,
                expected_run_id=None,
            )

    def test_run_metadata_rejects_identity_and_state_drift(self) -> None:
        requested = "owner/aleph-bench-deployment-diagnostic-2048-none"
        cases = {
            "owner": (_run_info(owner="other"), "run owner"),
            "task": (_run_info(task="other-task"), "run task"),
            "version": (_run_info(version=3), "run task version"),
            "id": (_run_info(run_id=0), "positive run ID"),
            "model": (_run_info(model=""), "model version slug"),
            "pending": (
                _run_info(state="BENCHMARK_TASK_RUN_STATE_RUNNING"),
                "not terminal",
            ),
        }
        for name, (run_info, message) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                KaggleCreationOutputError, message
            ):
                _run_metadata(
                    run_info,
                    requested_task=requested,
                    expected_version=2,
                    expected_run_id=24680,
                )

        with self.assertRaisesRegex(
            KaggleCreationOutputError, "positive integer task version"
        ):
            _run_metadata(
                _run_info(version=True),
                requested_task=requested,
                expected_version=1,
                expected_run_id=24680,
            )

    def test_run_metadata_restores_sdk_naive_timestamps_as_utc(self) -> None:
        run_info = _run_info()
        run_info.start_time = datetime(2026, 9, 18, 1)
        run_info.end_time = datetime(2026, 9, 18, 1, 1)

        metadata = _run_metadata(
            run_info,
            requested_task=(
                "owner/aleph-bench-deployment-diagnostic-2048-none"
            ),
            expected_version=2,
            expected_run_id=24680,
        )

        self.assertEqual(metadata["startTime"], "2026-09-18T01:00:00+00:00")
        self.assertEqual(metadata["endTime"], "2026-09-18T01:01:00+00:00")

    def test_task_metadata_restores_sdk_naive_timestamp_as_utc(self) -> None:
        task_info = _task_info()
        task_info.create_time = datetime(2026, 9, 18)

        metadata = _task_metadata(
            task_info,
            requested_task=(
                "owner/aleph-bench-deployment-diagnostic-2048-none"
            ),
            expected_version=2,
            expected_source_kernel_id=12345,
            expected_datasets=(),
        )

        self.assertEqual(metadata["createTime"], "2026-09-18T00:00:00+00:00")

    def test_task_metadata_rejects_boolean_version_one(self) -> None:
        with self.assertRaisesRegex(
            KaggleCreationOutputError, "positive integer version"
        ):
            creation_output._task_metadata(
                _task_info(version=True),
                requested_task=(
                    "owner/aleph-bench-deployment-diagnostic-2048-none"
                ),
                expected_version=1,
                expected_source_kernel_id=12345,
                expected_datasets=(),
            )

    def test_zero_call_bundle_binds_exact_task_version_and_kernel(self) -> None:
        receipt = _receipt(complete=False)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            creation_output,
            "parse_kaggle_diagnostic_receipt_bytes",
            return_value=receipt,
        ) as parse_receipt:
            metadata, passed = write_creation_bundle(
                task_info=_task_info(),
                run_info=_run_info(),
                requested_task=(
                    "owner/aleph-bench-deployment-diagnostic-2048-none"
                ),
                expected_version=2,
                expected_source_kernel_id=12345,
                expected_run_id=24680,
                expected_datasets=(),
                archive_bytes=_archive(),
                gate_mode="zero-call",
                output_dir=Path(tmp),
            )

            self.assertTrue(passed)
            self.assertEqual(metadata["task"]["sourceKernelId"], 12345)
            self.assertEqual(metadata["task"]["version"], 2)
            self.assertEqual(metadata["run"]["id"], 24680)
            self.assertTrue(metadata["gate"]["passed"])
            self.assertEqual(parse_receipt.call_count, 1)
            written = sorted(Path(tmp).iterdir())
            self.assertEqual(len(written), 3)
            metadata_path = next(
                path for path in written if path.name.endswith("-metadata.json")
            )
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8")), metadata
            )

    def test_bundle_accepts_kernel_observed_after_create_ack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            creation_output,
            "parse_kaggle_diagnostic_receipt_bytes",
            return_value=_receipt(complete=False),
        ):
            metadata, passed = write_creation_bundle(
                task_info=_task_info(),
                run_info=_run_info(),
                requested_task=(
                    "owner/aleph-bench-deployment-diagnostic-2048-none"
                ),
                expected_version=2,
                expected_source_kernel_id=None,
                expected_run_id=None,
                expected_datasets=(),
                archive_bytes=_archive(),
                gate_mode="zero-call",
                output_dir=Path(tmp),
            )

            self.assertTrue(passed)
            self.assertEqual(metadata["task"]["sourceKernelId"], 12345)
            self.assertEqual(metadata["run"]["id"], 24680)

    def test_runtime_mismatch_is_retained_but_does_not_pass_zero_call_gate(
        self,
    ) -> None:
        receipt = _receipt(complete=False)
        receipt["phase"] = "runtime_preflight"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            creation_output,
            "parse_kaggle_diagnostic_receipt_bytes",
            return_value=receipt,
        ):
            metadata, passed = write_creation_bundle(
                task_info=_task_info(),
                run_info=_run_info(),
                requested_task=(
                    "owner/aleph-bench-deployment-diagnostic-2048-none"
                ),
                expected_version=2,
                expected_source_kernel_id=None,
                expected_run_id=24680,
                expected_datasets=(),
                archive_bytes=_archive(),
                gate_mode="zero-call",
                output_dir=Path(tmp),
            )

            self.assertFalse(passed)
            self.assertIn("zero calls proven", metadata["gate"]["message"])
            self.assertIn("package gate was not reached", metadata["gate"]["message"])

    def test_incomplete_six_call_receipt_is_retained_but_fails_gate(self) -> None:
        receipt = _receipt(complete=False)
        dataset = "owner/aleph-bench-v02-scorer-conformance"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            creation_output,
            "parse_kaggle_diagnostic_receipt_bytes",
            return_value=receipt,
        ):
            metadata, passed = write_creation_bundle(
                task_info=_task_info(datasets=(dataset,)),
                run_info=_run_info(),
                requested_task=(
                    "owner/aleph-bench-deployment-diagnostic-2048-none"
                ),
                expected_version=2,
                expected_source_kernel_id=12345,
                expected_run_id=24680,
                expected_datasets=(dataset,),
                archive_bytes=_archive(),
                gate_mode="six-call",
                output_dir=Path(tmp),
            )

            self.assertFalse(passed)
            self.assertFalse(metadata["gate"]["passed"])
            self.assertEqual(len(list(Path(tmp).iterdir())), 3)

    def test_complete_six_call_receipt_passes(self) -> None:
        receipt = _receipt(complete=True)
        dataset = "owner/aleph-bench-v02-scorer-conformance"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            creation_output,
            "parse_kaggle_diagnostic_receipt_bytes",
            return_value=receipt,
        ):
            _, passed = write_creation_bundle(
                task_info=_task_info(datasets=(dataset,)),
                run_info=_run_info(),
                requested_task="aleph-bench-deployment-diagnostic-2048-none",
                expected_version=2,
                expected_source_kernel_id=12345,
                expected_run_id=24680,
                expected_datasets=(dataset,),
                archive_bytes=_archive(),
                gate_mode="six-call",
                output_dir=Path(tmp),
            )
            self.assertTrue(passed)

    def test_errored_creation_retains_call_ahead_receipt(self) -> None:
        receipt = _receipt(complete=False)
        receipt["phase"] = "model_calls"
        receipt["calls"] = {
            "attempted": 1,
            "completed": 0,
            "activeCall": {"state": "dispatching"},
        }
        task_info = _task_info(
            state="BENCHMARK_TASK_VERSION_CREATION_STATE_ERRORED",
            datasets=("owner/aleph-bench-v02-scorer-conformance",),
        )
        task_info.creation_error_message = "worker stopped"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            creation_output,
            "parse_kaggle_diagnostic_receipt_bytes",
            return_value=receipt,
        ):
            metadata, passed = write_creation_bundle(
                task_info=task_info,
                run_info=_run_info(),
                requested_task=(
                    "owner/aleph-bench-deployment-diagnostic-2048-none"
                ),
                expected_version=2,
                expected_source_kernel_id=12345,
                expected_run_id=24680,
                expected_datasets=(
                    "owner/aleph-bench-v02-scorer-conformance",
                ),
                archive_bytes=_archive(),
                gate_mode="six-call",
                output_dir=Path(tmp),
            )
            self.assertFalse(passed)
            self.assertEqual(metadata["task"]["creationError"], "worker stopped")
            self.assertTrue(metadata["receipt"]["manualReviewRequired"])
            self.assertEqual(len(list(Path(tmp).iterdir())), 3)

    def test_errored_run_retains_receipt_and_fails_gate(self) -> None:
        receipt = _receipt(complete=True)
        run_info = _run_info(state="BENCHMARK_TASK_RUN_STATE_ERRORED")
        run_info.error_message = "worker stopped"
        dataset = "owner/aleph-bench-v02-scorer-conformance"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            creation_output,
            "parse_kaggle_diagnostic_receipt_bytes",
            return_value=receipt,
        ):
            metadata, passed = write_creation_bundle(
                task_info=_task_info(datasets=(dataset,)),
                run_info=run_info,
                requested_task=(
                    "owner/aleph-bench-deployment-diagnostic-2048-none"
                ),
                expected_version=2,
                expected_source_kernel_id=12345,
                expected_run_id=24680,
                expected_datasets=(dataset,),
                archive_bytes=_archive(),
                gate_mode="six-call",
                output_dir=Path(tmp),
            )

            self.assertFalse(passed)
            self.assertEqual(metadata["run"]["error"], "worker stopped")
            self.assertTrue(metadata["receipt"]["manualReviewRequired"])

    def test_refuses_task_version_dataset_and_state_drift(self) -> None:
        cases = {
            "version": (_task_info(version=3), (), "zero-call", "version 3"),
            "dataset": (
                _task_info(datasets=("owner/wrong",)),
                ("owner/expected",),
                "six-call",
                "datasets differ",
            ),
            "state": (
                _task_info(state="BENCHMARK_TASK_VERSION_CREATION_STATE_RUNNING"),
                (),
                "zero-call",
                "not complete",
            ),
            "kernel": (
                _task_info(kernel_id=0),
                (),
                "zero-call",
                "expected 12345",
            ),
            "kernel-drift": (
                _task_info(kernel_id=54321),
                (),
                "zero-call",
                "expected 12345",
            ),
            "kernel-invalid": (
                _task_info(kernel_id=False),
                (),
                "zero-call",
                "source kernel ID is invalid",
            ),
            "kernel-float": (
                _task_info(kernel_id=0.0),
                (),
                "zero-call",
                "source kernel ID is invalid",
            ),
        }
        for name, (task_info, datasets, gate, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(
                    KaggleCreationOutputError, message
                ):
                    write_creation_bundle(
                        task_info=task_info,
                        run_info=_run_info(),
                        requested_task=(
                            "owner/aleph-bench-deployment-diagnostic-2048-none"
                        ),
                        expected_version=2,
                        expected_source_kernel_id=12345,
                        expected_run_id=24680,
                        expected_datasets=datasets,
                        archive_bytes=_archive(),
                        gate_mode=gate,
                        output_dir=Path(tmp),
                    )

    def test_rejects_duplicate_missing_and_unsafe_receipt_entries(self) -> None:
        duplicate = io.BytesIO()
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr(RECEIPT_FILENAME, "{}")
            archive.writestr(f"nested/{RECEIPT_FILENAME}", "{}")
        cases = {
            "duplicate": (duplicate.getvalue(), "exactly one"),
            "missing": (_archive(receipt_name="other.json"), "exactly one"),
            "unsafe": (
                _archive(receipt_name=f"../{RECEIPT_FILENAME}"),
                "unsafe creation archive entry",
            ),
        }
        for name, (archive_bytes, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(
                    KaggleCreationOutputError, message
                ):
                    write_creation_bundle(
                        task_info=_task_info(),
                        run_info=_run_info(),
                        requested_task=(
                            "owner/aleph-bench-deployment-diagnostic-2048-none"
                        ),
                        expected_version=2,
                        expected_source_kernel_id=12345,
                        expected_run_id=24680,
                        expected_datasets=(),
                        archive_bytes=archive_bytes,
                        gate_mode="zero-call",
                        output_dir=Path(tmp),
                    )

    def test_refuses_to_overwrite_retained_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            creation_output,
            "parse_kaggle_diagnostic_receipt_bytes",
            return_value=_receipt(complete=False),
        ):
            kwargs = {
                "task_info": _task_info(),
                "run_info": _run_info(),
                "requested_task": (
                    "owner/aleph-bench-deployment-diagnostic-2048-none"
                ),
                "expected_version": 2,
                "expected_source_kernel_id": 12345,
                "expected_run_id": 24680,
                "expected_datasets": (),
                "archive_bytes": _archive(),
                "gate_mode": "zero-call",
                "output_dir": Path(tmp),
            }
            write_creation_bundle(**kwargs)
            with self.assertRaisesRegex(
                KaggleCreationOutputError, "refusing to overwrite"
            ):
                write_creation_bundle(**kwargs)

if __name__ == "__main__":
    unittest.main()
