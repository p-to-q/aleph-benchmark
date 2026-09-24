from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from bench.engine import kaggle_run_once as run_once
from bench.engine.kaggle_run_once import (
    KaggleRunOnceError,
    KaggleRunOutcomeAmbiguous,
    schedule_and_reconcile_once,
)


OWNER = "jahyee"
TASK = "aleph-bench-v0-2-capture-canary"
VERSION = 6
MODEL = "gpt-5.4-mini"
MODEL_ID = 4301


def _slug() -> SimpleNamespace:
    return SimpleNamespace(
        owner_slug=OWNER,
        task_slug=TASK,
        version_number=VERSION,
    )


def _task(
    state: str = "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED",
) -> SimpleNamespace:
    return SimpleNamespace(
        slug=_slug(),
        creation_state=state,
        source_kernel_id=120,
        url=f"https://www.kaggle.com/benchmarks/{OWNER}/{TASK}/{VERSION}",
    )


def _model(
    *, slug: str = MODEL, model_id: int = MODEL_ID
) -> SimpleNamespace:
    return SimpleNamespace(
        id=301,
        version=SimpleNamespace(
            id=model_id,
            slug=slug,
            published=True,
            allow_model_proxy=True,
            model_proxy_slug="openai/gpt-5.4-mini",
            display_name="GPT-5.4 mini",
            deprecation_time=None,
        ),
    )


def _run(
    run_id: int,
    *,
    model: str = MODEL,
    state: str = "BENCHMARK_TASK_RUN_STATE_COMPLETED",
) -> SimpleNamespace:
    return SimpleNamespace(
        task_slug=_slug(),
        model_version_slug=model,
        id=run_id,
        state=state,
        start_time=datetime(2026, 9, 23, tzinfo=timezone.utc),
        end_time=(
            datetime(2026, 9, 23, 0, 1, tzinfo=timezone.utc)
            if state == "BENCHMARK_TASK_RUN_STATE_COMPLETED"
            else None
        ),
        error_message="",
    )


def _quota(used: float = 0.04) -> SimpleNamespace:
    refill = datetime(2026, 9, 24, tzinfo=timezone.utc)
    return SimpleNamespace(
        quota_balances=[
            SimpleNamespace(
                refill_period="DAILY",
                quota_used=used,
                total_quota_allowed=10.0,
                refill_time=refill,
            ),
            SimpleNamespace(
                refill_period="MONTHLY",
                quota_used=used,
                total_quota_allowed=100.0,
                refill_time=refill,
            ),
        ]
    )


def _response(
    *,
    scheduled: bool = True,
    model_id: int = MODEL_ID,
    parent_id: int = 0,
    reason: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                run_scheduled=scheduled,
                run_skipped_reason=reason,
                benchmark_task_version_id=901,
                benchmark_model_version_id=model_id,
                parent_task_version_id=parent_id,
            )
        ]
    )


class HardStop(BaseException):
    """Synthetic process interruption that must not be converted or retried."""


class KaggleRunOnceTests(unittest.TestCase):
    def _schedule(
        self,
        journal_path: Path,
        *,
        runs: list[list[SimpleNamespace]],
        schedule_run: object | None = None,
        fetch_task: object | None = None,
        fetch_model: object | None = None,
        fetch_quota: object | None = None,
    ) -> tuple[dict[str, object], mock.Mock]:
        schedule = mock.Mock(
            return_value=_response()
        ) if schedule_run is None else schedule_run
        run_snapshots = iter(runs)
        journal = schedule_and_reconcile_once(
            owner=OWNER,
            task=TASK,
            version=VERSION,
            model=MODEL,
            journal_path=journal_path,
            fetch_task=fetch_task or (lambda: _task()),
            fetch_model=fetch_model or (lambda: _model()),
            fetch_runs=lambda: next(run_snapshots),
            fetch_quota=fetch_quota or (lambda: _quota()),
            schedule_run=schedule,
            client_versions={"python": "3.13.7", "kaggle": "2.2.4"},
            reconcile_delays=(0, 0),
            clock=lambda: "2026-09-23T00:00:00Z",
            sleeper=lambda _delay: None,
        )
        return journal, schedule  # type: ignore[return-value]

    def test_success_schedules_once_and_binds_one_exact_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "run.json"
            journal, schedule = self._schedule(
                journal_path,
                runs=[[_run(10)], [_run(10), _run(11, state="BENCHMARK_TASK_RUN_STATE_QUEUED")]],
            )

            schedule.assert_called_once_with()
            self.assertEqual(journal["state"], "reconciled")
            self.assertEqual(journal["reconciliation"]["run"]["id"], 11)
            self.assertEqual(
                journal["remotePreflight"]["model"]["benchmarkModelVersionId"],
                MODEL_ID,
            )
            self.assertIsNone(journal["failure"])
            self.assertEqual(
                json.loads(journal_path.read_text(encoding="utf-8")), journal
            )

    def test_lost_schedule_response_can_be_reconciled_without_retry(self) -> None:
        schedule = mock.Mock(side_effect=ConnectionError("response lost"))
        with tempfile.TemporaryDirectory() as tmp:
            journal, _ = self._schedule(
                Path(tmp) / "run.json",
                runs=[[], [_run(21, state="BENCHMARK_TASK_RUN_STATE_RUNNING")]],
                schedule_run=schedule,
            )

        schedule.assert_called_once_with()
        self.assertEqual(journal["state"], "reconciled")
        self.assertEqual(journal["reconciliation"]["run"]["id"], 21)
        self.assertEqual(journal["dispatchFailure"]["type"], "ConnectionError")
        self.assertIsNone(journal["failure"])

    def test_unobserved_run_after_dispatch_is_ambiguous_and_not_retried(self) -> None:
        schedule = mock.Mock(return_value=_response())
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "run.json"
            with self.assertRaisesRegex(
                KaggleRunOutcomeAmbiguous, "do not schedule again"
            ):
                self._schedule(
                    journal_path,
                    runs=[[], [], []],
                    schedule_run=schedule,
                )

            retained = json.loads(journal_path.read_text(encoding="utf-8"))
        schedule.assert_called_once_with()
        self.assertEqual(retained["state"], "ambiguous")
        self.assertEqual(retained["failure"]["type"], "RunNotObserved")

    def test_wrong_or_multiple_new_runs_are_ambiguous(self) -> None:
        cases = {
            "wrong-model": [_run(31, model="gemini-3.7-flash")],
            "multiple": [_run(31), _run(32)],
        }
        for name, after in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                schedule = mock.Mock(return_value=_response())
                with self.assertRaisesRegex(
                    KaggleRunOutcomeAmbiguous, "ambiguous"
                ):
                    self._schedule(
                        Path(tmp) / "run.json",
                        runs=[[], after],
                        schedule_run=schedule,
                    )
                schedule.assert_called_once_with()

    def test_contradictory_schedule_identity_remains_ambiguous(self) -> None:
        schedule = mock.Mock(return_value=_response(model_id=999))
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "run.json"
            with self.assertRaisesRegex(
                KaggleRunOutcomeAmbiguous, "contradictory"
            ):
                self._schedule(
                    journal_path,
                    runs=[[], [_run(41)]],
                    schedule_run=schedule,
                )
            retained = json.loads(journal_path.read_text(encoding="utf-8"))

        self.assertEqual(retained["state"], "ambiguous")
        self.assertEqual(
            retained["responseFailure"]["type"], "KaggleRunOnceError"
        )
        self.assertEqual(retained["reconciliation"]["run"]["id"], 41)

    def test_skipped_response_is_conclusive_and_retained(self) -> None:
        schedule = mock.Mock(
            return_value=_response(scheduled=False, reason="already exists")
        )
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "run.json"
            with self.assertRaisesRegex(KaggleRunOnceError, "skipped"):
                self._schedule(
                    journal_path,
                    runs=[[]],
                    schedule_run=schedule,
                )
            retained = json.loads(journal_path.read_text(encoding="utf-8"))

        schedule.assert_called_once_with()
        self.assertEqual(retained["state"], "not_scheduled")
        self.assertEqual(retained["failure"]["message"], "already exists")

    def test_active_run_blocks_before_journal_and_schedule(self) -> None:
        schedule = mock.Mock(return_value=_response())
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "run.json"
            with self.assertRaisesRegex(KaggleRunOnceError, "unresolved"):
                self._schedule(
                    journal_path,
                    runs=[[_run(51, state="BENCHMARK_TASK_RUN_STATE_RUNNING")]],
                    schedule_run=schedule,
                )
            self.assertFalse(journal_path.exists())
        schedule.assert_not_called()

    def test_task_and_model_preflight_fail_closed(self) -> None:
        cases = {
            "task": (
                lambda: _task("BENCHMARK_TASK_VERSION_CREATION_STATE_RUNNING"),
                lambda: _model(),
                "not ready",
            ),
            "model": (
                lambda: _task(),
                lambda: _model(slug="other-model"),
                "different canonical",
            ),
        }
        for name, (task, model, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                journal_path = Path(tmp) / "run.json"
                schedule = mock.Mock(return_value=_response())
                with self.assertRaisesRegex(KaggleRunOnceError, message):
                    self._schedule(
                        journal_path,
                        runs=[[]],
                        schedule_run=schedule,
                        fetch_task=task,
                        fetch_model=model,
                    )
                self.assertFalse(journal_path.exists())
                schedule.assert_not_called()

    def test_invalid_target_and_existing_journal_block_before_remote_reads(self) -> None:
        for model in ("openai/gpt-5.4-mini", "gpt@default", ""):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                journal_path = Path(tmp) / "run.json"
                reads = mock.Mock()
                with self.assertRaisesRegex(KaggleRunOnceError, "canonical"):
                    schedule_and_reconcile_once(
                        owner=OWNER,
                        task=TASK,
                        version=VERSION,
                        model=model,
                        journal_path=journal_path,
                        fetch_task=reads,
                        fetch_model=reads,
                        fetch_runs=reads,
                        fetch_quota=reads,
                        schedule_run=reads,
                        client_versions={},
                    )
                reads.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "run.json"
            journal_path.write_text("{}\n", encoding="utf-8")
            reads = mock.Mock()
            with self.assertRaisesRegex(KaggleRunOnceError, "already exists"):
                schedule_and_reconcile_once(
                    owner=OWNER,
                    task=TASK,
                    version=VERSION,
                    model=MODEL,
                    journal_path=journal_path,
                    fetch_task=reads,
                    fetch_model=reads,
                    fetch_runs=reads,
                    fetch_quota=reads,
                    schedule_run=reads,
                    client_versions={},
                )
            reads.assert_not_called()

    def test_hard_stop_is_journaled_and_propagated_once(self) -> None:
        schedule = mock.Mock(side_effect=HardStop("interrupted"))
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "run.json"
            with self.assertRaises(HardStop):
                self._schedule(
                    journal_path,
                    runs=[[]],
                    schedule_run=schedule,
                )
            retained = json.loads(journal_path.read_text(encoding="utf-8"))
        schedule.assert_called_once_with()
        self.assertEqual(retained["state"], "ambiguous")
        self.assertEqual(retained["dispatchFailure"]["type"], "HardStop")

    def test_quota_after_failure_does_not_hide_unique_run(self) -> None:
        quotas = iter([_quota(), ConnectionError("quota read failed")])

        def fetch_quota() -> SimpleNamespace:
            result = next(quotas)
            if isinstance(result, Exception):
                raise result
            return result

        with tempfile.TemporaryDirectory() as tmp:
            journal, _ = self._schedule(
                Path(tmp) / "run.json",
                runs=[[], [_run(61)]],
                fetch_quota=fetch_quota,
            )

        self.assertEqual(journal["state"], "reconciled")
        self.assertIsNone(journal["quotaAfter"])
        self.assertEqual(
            journal["quotaAfterFailure"]["type"], "ConnectionError"
        )

    def test_run_once_sets_explicit_version_and_never_retries_schedule(self) -> None:
        class Request:
            pass

        task_client = SimpleNamespace(
            get_benchmark_task=mock.Mock(return_value=_task()),
            list_benchmark_task_runs=mock.Mock(
                side_effect=[
                    SimpleNamespace(runs=[], next_page_token=""),
                    SimpleNamespace(runs=[_run(71)], next_page_token=""),
                ]
            ),
            batch_schedule_benchmark_task_runs=mock.Mock(
                return_value=_response()
            ),
        )
        model_client = SimpleNamespace(
            list_benchmark_models=mock.Mock(
                return_value=SimpleNamespace(
                    benchmark_models=[_model()], next_page_token=""
                )
            )
        )
        quota_client = SimpleNamespace(
            get_model_proxy_quotas=mock.Mock(side_effect=[_quota(), _quota()])
        )
        client = SimpleNamespace(
            benchmarks=SimpleNamespace(
                benchmark_tasks_api_client=task_client,
                benchmarks_api_client=model_client,
            ),
            models=SimpleNamespace(model_proxy_api_client=quota_client),
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
            "kagglesdk.benchmarks.types.benchmark_tasks_api_service",
            "kagglesdk.benchmarks.types.benchmarks_api_service",
            "kagglesdk.models",
            "kagglesdk.models.types",
            "kagglesdk.models.types.model_proxy_api_service",
        ):
            modules[name] = ModuleType(name)
        modules["kaggle.api.kaggle_api_extended"].KaggleApi = mock.Mock(
            return_value=api
        )
        task_types = modules[
            "kagglesdk.benchmarks.types.benchmark_tasks_api_service"
        ]
        task_types.ApiBatchScheduleBenchmarkTaskRunsRequest = Request
        task_types.ApiBenchmarkTaskSlug = Request
        task_types.ApiGetBenchmarkTaskRequest = Request
        task_types.ApiListBenchmarkTaskRunsRequest = Request
        modules[
            "kagglesdk.benchmarks.types.benchmarks_api_service"
        ].ApiListBenchmarkModelsRequest = Request
        modules[
            "kagglesdk.models.types.model_proxy_api_service"
        ].ApiGetModelProxyQuotasRequest = Request

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                run_once, "_verified_client_versions", return_value={}
            ),
            mock.patch.dict(sys.modules, modules),
        ):
            journal = run_once.run_once(
                owner=OWNER,
                task=TASK,
                version=VERSION,
                model=MODEL,
                journal_path=Path(tmp) / "run.json",
                reconcile_delays=(0,),
            )

        schedule_request = task_client.batch_schedule_benchmark_task_runs.call_args.args[0]
        requested_slug = schedule_request.task_slugs[0]
        self.assertEqual(requested_slug.owner_slug, OWNER)
        self.assertEqual(requested_slug.task_slug, TASK)
        self.assertEqual(requested_slug.version_number, VERSION)
        self.assertEqual(schedule_request.model_version_slugs, [MODEL])
        task_client.batch_schedule_benchmark_task_runs.assert_called_once()
        self.assertNotIn(
            task_client.batch_schedule_benchmark_task_runs,
            [call.args[0] for call in api.with_retry.call_args_list],
        )
        self.assertEqual(journal["reconciliation"]["run"]["id"], 71)


if __name__ == "__main__":
    unittest.main()
