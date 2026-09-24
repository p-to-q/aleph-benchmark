from __future__ import annotations

import io
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from bench.engine.kaggle_capture import (
    artifact_id_for,
    serialize_capture_payload,
)
from bench.engine.kaggle_capture_evidence import (
    CAPTURE_FILENAME,
    KaggleCaptureEvidenceError,
    _artifact_id,
    verify_capture_evidence,
    write_capture_evidence_bundle,
)
from bench.engine.kaggle_run_once import schedule_and_reconcile_once
from bench.tasks.kaggle.generate_v0_2_capture import OUTPUT_PATH
from bench.tests.test_kaggle_capture_task_v0_2 import (
    OpenAI,
    RUNTIME_313,
    _Chats,
    _Clock,
    _load_generated_task,
    _write_package,
)


TASK_SLUG = "aleph-bench-v0-2-capture-canary"
DATASET = "owner/aleph-bench-v02-scorer-conformance"
DOWNLOADED_AT = "2026-09-22T00:02:00Z"
GEMMA_SCHEDULED_SLUG = "gemma-4-26b-a4b-it"
GEMMA_PROXY_SLUG = "google/gemma-4-26b-a4b"
OPERATION_ID = "01234567-89ab-4def-8123-456789abcdef"


def _task_info(
    *,
    version: int = 3,
    state: str = "BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED",
    datasets: tuple[str, ...] = (DATASET,),
) -> SimpleNamespace:
    return SimpleNamespace(
        slug=SimpleNamespace(
            owner_slug="owner",
            task_slug=TASK_SLUG,
            version_number=version,
        ),
        creation_state=state,
        creation_error_message="platform detail" if state.endswith("ERRORED") else None,
        error=None,
        source_kernel_id=12345,
        options=SimpleNamespace(dataset_data_sources=list(datasets)),
        create_time=datetime(2026, 9, 21, 23, 58, tzinfo=timezone.utc),
        url=f"https://www.kaggle.com/benchmarks/owner/{TASK_SLUG}/{version}",
    )


def _run_info(
    *,
    version: int = 3,
    model: str = "google/gemini-2.5-flash",
    state: str = "BENCHMARK_TASK_RUN_STATE_COMPLETED",
    start: datetime | None = None,
    end: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_slug=SimpleNamespace(
            owner_slug="owner",
            task_slug=TASK_SLUG,
            version_number=version,
        ),
        id=24680,
        model_version_slug=model,
        state=state,
        error_message="run detail" if state.endswith("ERRORED") else None,
        start_time=start or datetime(2026, 9, 21, 23, 59, tzinfo=timezone.utc),
        end_time=end or datetime(2026, 9, 22, 0, 1, tzinfo=timezone.utc),
    )


def _archive(
    payload_bytes: bytes,
    source_bytes: bytes,
    *,
    notebook: bool = False,
    strip_notebook_shebang: bool = False,
    extra_payload: bool = False,
    extra_source: bool = False,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(f"output/{CAPTURE_FILENAME}", payload_bytes)
        if extra_payload:
            archive.writestr(f"duplicate/{CAPTURE_FILENAME}", payload_bytes)
        if notebook:
            notebook_source = source_bytes.decode("utf-8")
            if strip_notebook_shebang:
                self_contained_shebang = "#!/usr/bin/env python3\n"
                if not notebook_source.startswith(self_contained_shebang):
                    raise AssertionError("test source is missing the expected shebang")
                notebook_source = notebook_source[len(self_contained_shebang) :]
                if not notebook_source.endswith("\n"):
                    raise AssertionError("test source is missing the expected final newline")
                notebook_source = notebook_source[:-1]
            notebook_bytes = (
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "metadata": {},
                                "source": notebook_source,
                            }
                        ],
                        "metadata": {},
                        "nbformat": 4,
                        "nbformat_minor": 5,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            archive.writestr("source/__notebook__.ipynb", notebook_bytes)
        else:
            archive.writestr("source/task.py", source_bytes)
        if extra_source:
            archive.writestr("source/task-copy.py", source_bytes)
        archive.writestr("logs/run.log", "complete\n")
    return output.getvalue()


def _dispatch_journal_bytes(
    *,
    owner: str = "owner",
    task: str = TASK_SLUG,
    version: int = 3,
    run_id: int = 24680,
    scheduled_slug: str = GEMMA_SCHEDULED_SLUG,
    proxy_slug: str = GEMMA_PROXY_SLUG,
    state: str = "reconciled",
    dispatch_failure: dict[str, Any] | None = None,
    response_failure: dict[str, str] | None = None,
    failure: dict[str, str] | None = None,
) -> bytes:
    reconciled_run = {
        "id": run_id,
        "model": scheduled_slug,
        "state": "BENCHMARK_TASK_RUN_STATE_QUEUED",
        "startTime": "2026-09-22T00:00:00Z",
        "endTime": None,
        "errorMessage": None,
    }
    journal = {
        "journalVersion": 1,
        "artifactKind": "kaggle_benchmark_run_dispatch",
        "operationId": OPERATION_ID,
        "state": state,
        "target": {
            "owner": owner,
            "task": task,
            "version": version,
            "model": scheduled_slug,
        },
        "remotePreflight": {
            "task": {
                "owner": owner,
                "task": task,
                "version": version,
            },
            "model": {
                "benchmarkModelId": 141,
                "benchmarkModelVersionId": 139,
                "slug": scheduled_slug,
                "modelProxySlug": proxy_slug,
            },
            "runs": [
                {
                    "id": 13579,
                    "model": "another-model",
                    "state": "BENCHMARK_TASK_RUN_STATE_COMPLETED",
                    "startTime": "2026-09-21T00:00:00Z",
                    "endTime": "2026-09-21T00:01:00Z",
                    "errorMessage": None,
                }
            ],
        },
        "response": (
            None
            if dispatch_failure is not None
            else {
                "runScheduled": True,
                "runSkippedReason": None,
                "benchmarkTaskVersionId": 777,
                "benchmarkModelVersionId": 139,
                "parentTaskVersionId": None,
            }
        ),
        "reconciliation": {
            "observations": [
                {
                    "attempt": 1,
                    "observedAt": "2026-09-22T00:00:00Z",
                    "delaySeconds": 0.0,
                    "runIds": sorted([13579, run_id]),
                    "newRuns": [reconciled_run],
                }
            ],
            "run": reconciled_run,
        },
        "dispatchFailure": dispatch_failure,
        "responseFailure": response_failure,
        "failure": failure,
    }
    return (json.dumps(journal, sort_keys=True) + "\n").encode("utf-8")


class KaggleCaptureEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated, cls.previous_kaggle_module = _load_generated_task()
        cls.package_tmp = tempfile.TemporaryDirectory()
        cls.package_root = Path(cls.package_tmp.name) / "package"
        cls.package_root.mkdir()
        _write_package(cls.package_root)
        cls.source_bytes = OUTPUT_PATH.read_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.package_tmp.cleanup()
        if cls.previous_kaggle_module is None:
            sys.modules.pop("kaggle_benchmarks", None)
        else:
            sys.modules["kaggle_benchmarks"] = cls.previous_kaggle_module

    def _payload_bytes(
        self,
        *,
        finish_reason: str | None = "stop",
        model_slug: str = "google/gemini-2.5-flash",
    ) -> bytes:
        output_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(output_tmp.cleanup)
        capture_path = Path(output_tmp.name) / "capture.json"
        llm = OpenAI()
        llm.model = model_slug
        self.generated.run_capture_canary(
            llm,
            chats=_Chats(finish_reason=finish_reason),
            package_root=self.package_root,
            capture_path=capture_path,
            observed_runtime=RUNTIME_313,
            clock=_Clock(),
        )
        return capture_path.read_bytes()

    def _blocked_payload_bytes(self) -> bytes:
        output_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(output_tmp.cleanup)
        empty_package = Path(output_tmp.name) / "empty-package"
        empty_package.mkdir()
        capture_path = Path(output_tmp.name) / "capture.json"
        self.generated.run_capture_canary(
            OpenAI(),
            chats=_Chats(),
            package_root=empty_package,
            capture_path=capture_path,
            observed_runtime=RUNTIME_313,
            clock=_Clock(),
        )
        return capture_path.read_bytes()

    def _write(
        self,
        *,
        archive_bytes: bytes,
        task_info: SimpleNamespace | None = None,
        run_info: SimpleNamespace | None = None,
        output_dir: Path | None = None,
        dispatch_journal_bytes: bytes | None = None,
    ) -> tuple[dict[str, Any], Path, tempfile.TemporaryDirectory[str] | None]:
        output_tmp = None
        if output_dir is None:
            output_tmp = tempfile.TemporaryDirectory()
            self.addCleanup(output_tmp.cleanup)
            output_dir = Path(output_tmp.name) / "evidence"
        evidence = write_capture_evidence_bundle(
            task_info=task_info or _task_info(),
            run_info=run_info or _run_info(),
            requested_task=f"owner/{TASK_SLUG}",
            expected_version=3,
            expected_source_kernel_id=12345,
            expected_run_id=24680,
            expected_datasets=(DATASET,),
            archive_bytes=archive_bytes,
            output_dir=output_dir,
            downloaded_at=DOWNLOADED_AT,
            dispatch_journal_bytes=dispatch_journal_bytes,
        )
        return evidence, output_dir, output_tmp

    def test_exact_run_bundle_preserves_and_verifies_all_bytes(self) -> None:
        payload_bytes = self._payload_bytes()
        archive_bytes = _archive(payload_bytes, self.source_bytes)
        evidence, output_dir, output_tmp = self._write(archive_bytes=archive_bytes)
        self.assertIsNotNone(output_tmp)

        self.assertEqual(evidence["evidenceSchemaVersion"], "1.0.0")
        self.assertTrue(evidence["assemblyEligible"])
        self.assertNotIn("modelCatalogBinding", evidence)
        self.assertIsNone(evidence["task"]["creationErrorStringSha256"])
        self.assertIsNone(evidence["run"]["errorStringSha256"])
        archive_copy = (output_dir / evidence["archive"]["file"]).read_bytes()
        payload_copy = (output_dir / evidence["payload"]["file"]).read_bytes()
        source_copy = (output_dir / evidence["source"]["file"]).read_bytes()
        self.assertEqual(archive_copy, archive_bytes)
        self.assertEqual(payload_copy, payload_bytes)
        self.assertEqual(source_copy, self.source_bytes)
        self.assertEqual(
            verify_capture_evidence(
                evidence,
                archive_bytes=archive_copy,
                payload_bytes=payload_copy,
                source_bytes=source_copy,
            ),
            evidence,
        )

        missing_binding = json.loads(json.dumps(evidence))
        missing_binding["evidenceSchemaVersion"] = "1.1.0"
        missing_binding["id"] = _artifact_id(missing_binding)
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "schema validation failed"
        ):
            verify_capture_evidence(
                missing_binding,
                archive_bytes=archive_copy,
                payload_bytes=payload_copy,
                source_bytes=source_copy,
            )

        null_binding = json.loads(json.dumps(missing_binding))
        null_binding["modelCatalogBinding"] = None
        null_binding["id"] = _artifact_id(null_binding)
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "schema validation failed"
        ):
            verify_capture_evidence(
                null_binding,
                archive_bytes=archive_copy,
                payload_bytes=payload_copy,
                source_bytes=source_copy,
            )

    def test_missing_finish_reason_is_bound_and_assembly_eligible(self) -> None:
        payload_bytes = self._payload_bytes(finish_reason=None)
        evidence, output_dir, output_tmp = self._write(
            archive_bytes=_archive(payload_bytes, self.source_bytes)
        )
        self.assertIsNotNone(output_tmp)
        self.assertTrue(evidence["assemblyEligible"])
        self.assertTrue(evidence["payload"]["canonicalReplayEligible"])

    def test_providerless_run_api_model_alias_is_bound_without_rewriting(self) -> None:
        payload_bytes = self._payload_bytes()
        evidence, _, output_tmp = self._write(
            archive_bytes=_archive(payload_bytes, self.source_bytes),
            run_info=_run_info(model="gemini-2.5-flash"),
        )
        self.assertIsNotNone(output_tmp)
        self.assertTrue(evidence["assemblyEligible"])
        self.assertEqual(
            evidence["run"]["modelVersionSlug"],
            "gemini-2.5-flash",
        )

    def test_versioned_providerless_run_alias_is_bound_without_rewriting(self) -> None:
        payload_bytes = self._payload_bytes(
            model_slug="anthropic/claude-haiku-4-5@20251001"
        )
        evidence, output_dir, output_tmp = self._write(
            archive_bytes=_archive(payload_bytes, self.source_bytes),
            run_info=_run_info(model="claude-haiku-4-5-20251001"),
        )
        self.assertIsNotNone(output_tmp)
        self.assertTrue(evidence["assemblyEligible"])
        self.assertEqual(
            evidence["run"]["modelVersionSlug"],
            "claude-haiku-4-5-20251001",
        )
        retained_payload = json.loads(
            (output_dir / evidence["payload"]["file"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            retained_payload["modelObservation"]["slug"],
            "anthropic/claude-haiku-4-5@20251001",
        )

    def test_reconciled_catalog_alias_is_bound_to_exact_journal_bytes(self) -> None:
        payload_bytes = self._payload_bytes(model_slug=GEMMA_PROXY_SLUG)
        archive_bytes = _archive(payload_bytes, self.source_bytes)
        journal_bytes = _dispatch_journal_bytes()
        evidence, output_dir, output_tmp = self._write(
            archive_bytes=archive_bytes,
            run_info=_run_info(model=GEMMA_SCHEDULED_SLUG),
            dispatch_journal_bytes=journal_bytes,
        )
        self.assertIsNotNone(output_tmp)
        self.assertEqual(evidence["evidenceSchemaVersion"], "1.1.0")
        self.assertTrue(evidence["assemblyEligible"])
        self.assertEqual(
            evidence["modelCatalogBinding"],
            {
                "operationId": OPERATION_ID,
                "benchmarkModelId": 141,
                "benchmarkModelVersionId": 139,
                "scheduledSlug": GEMMA_SCHEDULED_SLUG,
                "modelProxySlug": GEMMA_PROXY_SLUG,
                "dispatchJournal": {
                    "file": (
                        f"{TASK_SLUG}-v3-run-24680-dispatch-journal.json"
                    ),
                    "bytes": len(journal_bytes),
                    "sha256": hashlib.sha256(journal_bytes).hexdigest(),
                },
            },
        )
        retained_journal = (
            output_dir
            / evidence["modelCatalogBinding"]["dispatchJournal"]["file"]
        ).read_bytes()
        self.assertEqual(retained_journal, journal_bytes)
        self.assertEqual(
            verify_capture_evidence(
                evidence,
                archive_bytes=archive_bytes,
                payload_bytes=payload_bytes,
                source_bytes=self.source_bytes,
                dispatch_journal_bytes=retained_journal,
            ),
            evidence,
        )

        legacy = json.loads(json.dumps(evidence))
        legacy["evidenceSchemaVersion"] = "1.0.0"
        legacy["id"] = _artifact_id(legacy)
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "schema validation failed"
        ):
            verify_capture_evidence(
                legacy,
                archive_bytes=archive_bytes,
                payload_bytes=payload_bytes,
                source_bytes=self.source_bytes,
                dispatch_journal_bytes=journal_bytes,
            )

    def test_lost_schedule_response_reconciled_by_scheduler_is_bindable(self) -> None:
        task_slug = SimpleNamespace(
            owner_slug="owner",
            task_slug=TASK_SLUG,
            version_number=3,
        )
        scheduler_task = SimpleNamespace(
            slug=task_slug,
            creation_state="BENCHMARK_TASK_VERSION_CREATION_STATE_COMPLETED",
            source_kernel_id=12345,
            url=f"https://www.kaggle.com/benchmarks/owner/{TASK_SLUG}/3",
        )
        scheduler_model = SimpleNamespace(
            id=141,
            version=SimpleNamespace(
                id=139,
                slug=GEMMA_SCHEDULED_SLUG,
                published=True,
                allow_model_proxy=True,
                model_proxy_slug=GEMMA_PROXY_SLUG,
                display_name="Gemma 4 26B A4B",
                deprecation_time=None,
            ),
        )
        scheduler_run = SimpleNamespace(
            task_slug=task_slug,
            model_version_slug=GEMMA_SCHEDULED_SLUG,
            id=24680,
            state="BENCHMARK_TASK_RUN_STATE_RUNNING",
            start_time=datetime(2026, 9, 21, 23, 59, tzinfo=timezone.utc),
            end_time=None,
            error_message="",
        )
        refill = datetime(2026, 9, 24, tzinfo=timezone.utc)
        quota = SimpleNamespace(
            quota_balances=[
                SimpleNamespace(
                    refill_period=period,
                    quota_used=0.1,
                    total_quota_allowed=allowance,
                    refill_time=refill,
                )
                for period, allowance in (("DAILY", 10.0), ("MONTHLY", 100.0))
            ]
        )
        run_snapshots = iter([[], OSError("read failed"), [scheduler_run]])

        def fetch_runs() -> list[SimpleNamespace]:
            snapshot = next(run_snapshots)
            if isinstance(snapshot, OSError):
                raise snapshot
            return snapshot

        schedule = mock.Mock(side_effect=ConnectionError("response lost"))
        journal_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(journal_tmp.cleanup)
        journal_path = Path(journal_tmp.name) / "run-journal.json"
        journal = schedule_and_reconcile_once(
            owner="owner",
            task=TASK_SLUG,
            version=3,
            model=GEMMA_SCHEDULED_SLUG,
            journal_path=journal_path,
            fetch_task=lambda: scheduler_task,
            fetch_model=lambda: scheduler_model,
            fetch_runs=fetch_runs,
            fetch_quota=lambda: quota,
            schedule_run=schedule,
            client_versions={"python": "3.13.7", "kaggle": "2.2.4"},
            reconcile_delays=(0, 0),
            clock=lambda: "2026-09-22T00:00:00Z",
            sleeper=lambda _delay: None,
        )
        schedule.assert_called_once_with()
        self.assertEqual(journal["state"], "reconciled")
        self.assertIsNone(journal["response"])
        self.assertEqual(journal["dispatchFailure"]["type"], "ConnectionError")
        self.assertEqual(
            journal["reconciliation"]["observations"][0]["failure"]["type"],
            "OSError",
        )

        journal_bytes = journal_path.read_bytes()
        payload_bytes = self._payload_bytes(model_slug=GEMMA_PROXY_SLUG)
        evidence, _, output_tmp = self._write(
            archive_bytes=_archive(payload_bytes, self.source_bytes),
            run_info=_run_info(model=GEMMA_SCHEDULED_SLUG),
            dispatch_journal_bytes=journal_bytes,
        )
        self.assertIsNotNone(output_tmp)
        self.assertTrue(evidence["assemblyEligible"])
        self.assertEqual(
            evidence["modelCatalogBinding"]["scheduledSlug"],
            GEMMA_SCHEDULED_SLUG,
        )

    def test_catalog_alias_requires_one_exact_runtime_proxy_slug(self) -> None:
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "run model differs"
        ):
            self._write(
                archive_bytes=_archive(
                    self._payload_bytes(model_slug=GEMMA_PROXY_SLUG),
                    self.source_bytes,
                ),
                run_info=_run_info(model=GEMMA_SCHEDULED_SLUG),
            )

        for observed_slug in (
            "gemma-4-26b-a4b",
            "google/gemma-4-26b-a4b-it",
            "other/gemma-4-26b-a4b",
        ):
            with self.subTest(observed_slug=observed_slug), self.assertRaisesRegex(
                KaggleCaptureEvidenceError,
                "Model Proxy slug differs|unavailable runtime model",
            ):
                self._write(
                    archive_bytes=_archive(
                        self._payload_bytes(model_slug=observed_slug),
                        self.source_bytes,
                    ),
                    run_info=_run_info(model=GEMMA_SCHEDULED_SLUG),
                    dispatch_journal_bytes=_dispatch_journal_bytes(),
                )

    def test_catalog_binding_tampering_fails_semantic_verification(self) -> None:
        payload_bytes = self._payload_bytes(model_slug=GEMMA_PROXY_SLUG)
        archive_bytes = _archive(payload_bytes, self.source_bytes)
        journal_bytes = _dispatch_journal_bytes()
        evidence, _, output_tmp = self._write(
            archive_bytes=archive_bytes,
            run_info=_run_info(model=GEMMA_SCHEDULED_SLUG),
            dispatch_journal_bytes=journal_bytes,
        )
        self.assertIsNotNone(output_tmp)
        cases = {
            "operation-id": (
                "operationId",
                "11234567-89ab-4def-8123-456789abcdef",
            ),
            "model-id": ("benchmarkModelId", 142),
            "model-version-id": ("benchmarkModelVersionId", 140),
            "scheduled-slug": ("scheduledSlug", "gemma-4-26b-a4b"),
            "proxy-slug": (
                "modelProxySlug",
                "google/gemma-4-26b-a4b-it",
            ),
        }
        for name, (field, value) in cases.items():
            mutated = json.loads(json.dumps(evidence))
            mutated["modelCatalogBinding"][field] = value
            mutated["id"] = _artifact_id(mutated)
            with self.subTest(name=name), self.assertRaisesRegex(
                KaggleCaptureEvidenceError,
                "model catalog binding disagrees",
            ):
                verify_capture_evidence(
                    mutated,
                    archive_bytes=archive_bytes,
                    payload_bytes=payload_bytes,
                    source_bytes=self.source_bytes,
                    dispatch_journal_bytes=journal_bytes,
                )

        mutated = json.loads(json.dumps(evidence))
        mutated["modelCatalogBinding"]["dispatchJournal"]["sha256"] = "0" * 64
        mutated["id"] = _artifact_id(mutated)
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "model catalog binding disagrees"
        ):
            verify_capture_evidence(
                mutated,
                archive_bytes=archive_bytes,
                payload_bytes=payload_bytes,
                source_bytes=self.source_bytes,
                dispatch_journal_bytes=journal_bytes,
            )

    def test_dispatch_journal_identity_and_terminal_state_fail_closed(self) -> None:
        payload_bytes = self._payload_bytes(model_slug=GEMMA_PROXY_SLUG)
        archive_bytes = _archive(payload_bytes, self.source_bytes)
        dispatch_with_response = json.loads(
            _dispatch_journal_bytes(
                dispatch_failure={"type": "Error", "message": "response lost"}
            )
        )
        dispatch_with_response["response"] = json.loads(
            _dispatch_journal_bytes()
        )["response"]
        dispatch_with_response_bytes = (
            json.dumps(dispatch_with_response, sort_keys=True) + "\n"
        ).encode("utf-8")
        mismatched_observation = json.loads(_dispatch_journal_bytes())
        mismatched_observation["reconciliation"]["observations"][-1][
            "newRuns"
        ] = []
        mismatched_observation_bytes = (
            json.dumps(mismatched_observation, sort_keys=True) + "\n"
        ).encode("utf-8")
        duplicate_preflight = json.loads(_dispatch_journal_bytes())
        duplicate_preflight["remotePreflight"]["runs"].append(
            dict(duplicate_preflight["remotePreflight"]["runs"][0])
        )
        duplicate_preflight_bytes = (
            json.dumps(duplicate_preflight, sort_keys=True) + "\n"
        ).encode("utf-8")
        missing_final_run_ids = json.loads(_dispatch_journal_bytes())
        del missing_final_run_ids["reconciliation"]["observations"][-1][
            "runIds"
        ]
        missing_final_run_ids_bytes = (
            json.dumps(missing_final_run_ids, sort_keys=True) + "\n"
        ).encode("utf-8")
        missing_preflight_from_final = json.loads(_dispatch_journal_bytes())
        missing_preflight_from_final["reconciliation"]["observations"][-1][
            "runIds"
        ] = [24680]
        missing_preflight_from_final_bytes = (
            json.dumps(missing_preflight_from_final, sort_keys=True) + "\n"
        ).encode("utf-8")
        extra_final_run = json.loads(_dispatch_journal_bytes())
        extra_run_record = {
            "id": 30000,
            "model": "unexpected-model",
            "state": "BENCHMARK_TASK_RUN_STATE_QUEUED",
            "startTime": None,
            "endTime": None,
            "errorMessage": None,
        }
        extra_final_observation = extra_final_run["reconciliation"][
            "observations"
        ][-1]
        extra_final_observation["runIds"] = [13579, 24680, 30000]
        extra_final_observation["newRuns"].append(extra_run_record)
        extra_final_run_bytes = (
            json.dumps(extra_final_run, sort_keys=True) + "\n"
        ).encode("utf-8")
        earlier_new_run = json.loads(_dispatch_journal_bytes())
        unexpected_run = {
            "id": 20000,
            "model": "another-new-model",
            "state": "BENCHMARK_TASK_RUN_STATE_QUEUED",
            "startTime": None,
            "endTime": None,
            "errorMessage": None,
        }
        earlier_new_run["reconciliation"]["observations"].insert(
            0,
            {
                "attempt": 1,
                "observedAt": "2026-09-22T00:00:00Z",
                "delaySeconds": 0.0,
                "runIds": [13579, 20000],
                "newRuns": [unexpected_run],
            },
        )
        earlier_new_run["reconciliation"]["observations"][-1]["attempt"] = 2
        earlier_new_run_bytes = (
            json.dumps(earlier_new_run, sort_keys=True) + "\n"
        ).encode("utf-8")
        boolean_journal_version = json.loads(_dispatch_journal_bytes())
        boolean_journal_version["journalVersion"] = True
        boolean_journal_version_bytes = (
            json.dumps(boolean_journal_version, sort_keys=True) + "\n"
        ).encode("utf-8")
        numeric_preflight_start = json.loads(_dispatch_journal_bytes())
        numeric_preflight_start["remotePreflight"]["runs"][0][
            "startTime"
        ] = 123
        numeric_preflight_start_bytes = (
            json.dumps(numeric_preflight_start, sort_keys=True) + "\n"
        ).encode("utf-8")
        numeric_preflight_end = json.loads(_dispatch_journal_bytes())
        numeric_preflight_end["remotePreflight"]["runs"][0]["endTime"] = 123
        numeric_preflight_end_bytes = (
            json.dumps(numeric_preflight_end, sort_keys=True) + "\n"
        ).encode("utf-8")
        numeric_observed_at = json.loads(_dispatch_journal_bytes())
        numeric_observed_at["reconciliation"]["observations"][-1][
            "observedAt"
        ] = 123
        numeric_observed_at_bytes = (
            json.dumps(numeric_observed_at, sort_keys=True) + "\n"
        ).encode("utf-8")
        cases = {
            "other-owner": (
                _dispatch_journal_bytes(owner="other"),
                "different Kaggle task version",
            ),
            "other-task": (
                _dispatch_journal_bytes(task="other-task"),
                "different Kaggle task version",
            ),
            "other-version": (
                _dispatch_journal_bytes(version=4),
                "different Kaggle task version",
            ),
            "other-run": (
                _dispatch_journal_bytes(run_id=24681),
                "different Kaggle run",
            ),
            "other-model": (
                _dispatch_journal_bytes(scheduled_slug="other-model"),
                "different Kaggle run model",
            ),
            "not-reconciled": (
                _dispatch_journal_bytes(state="dispatching"),
                "not reconciled",
            ),
            "response-failure": (
                _dispatch_journal_bytes(
                    response_failure={"type": "Error", "message": "invalid"}
                ),
                "retains responseFailure",
            ),
            "retained-failure": (
                _dispatch_journal_bytes(
                    failure={"type": "Error", "message": "ambiguous"}
                ),
                "retains failure",
            ),
            "invalid-dispatch-failure": (
                _dispatch_journal_bytes(
                    dispatch_failure={
                        "type": "Error",
                        "message": "response lost",
                        "unexpected": "field",
                    }
                ),
                "invalid dispatchFailure",
            ),
            "dispatch-failure-with-response": (
                dispatch_with_response_bytes,
                "both a response and dispatchFailure",
            ),
            "mismatched-final-observation": (
                mismatched_observation_bytes,
                "newRuns differs from its run-set delta",
            ),
            "duplicate-preflight-run-id": (
                duplicate_preflight_bytes,
                "preflight run IDs are not ordered and unique",
            ),
            "missing-final-run-ids": (
                missing_final_run_ids_bytes,
                "invalid success fields",
            ),
            "final-run-ids-lost-preflight": (
                missing_preflight_from_final_bytes,
                "lost a preflight run ID",
            ),
            "final-run-ids-have-extra-run": (
                extra_final_run_bytes,
                "final runIds differs from preflight plus bound run",
            ),
            "continued-after-earlier-new-run": (
                earlier_new_run_bytes,
                "continued after observing a new run",
            ),
            "boolean-journal-version": (
                boolean_journal_version_bytes,
                "version is unsupported",
            ),
            "numeric-preflight-start-time": (
                numeric_preflight_start_bytes,
                "not a valid timestamp",
            ),
            "numeric-preflight-end-time": (
                numeric_preflight_end_bytes,
                "not a valid timestamp",
            ),
            "numeric-observation-time": (
                numeric_observed_at_bytes,
                "not a valid timestamp",
            ),
        }
        for name, (journal_bytes, message) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                KaggleCaptureEvidenceError, message
            ):
                self._write(
                    archive_bytes=archive_bytes,
                    run_info=_run_info(model=GEMMA_SCHEDULED_SLUG),
                    dispatch_journal_bytes=journal_bytes,
                )

    def test_zero_call_preflight_block_is_retained_as_ineligible_evidence(self) -> None:
        payload_bytes = self._blocked_payload_bytes()
        evidence, _, output_tmp = self._write(
            archive_bytes=_archive(payload_bytes, self.source_bytes)
        )
        self.assertIsNotNone(output_tmp)
        self.assertFalse(evidence["assemblyEligible"])
        self.assertFalse(evidence["payload"]["captureComplete"])
        self.assertEqual(
            evidence["run"]["modelVersionSlug"],
            "google/gemini-2.5-flash",
        )

    def test_exact_notebook_code_cell_source_is_accepted(self) -> None:
        payload_bytes = self._payload_bytes()
        evidence, _, output_tmp = self._write(
            archive_bytes=_archive(
                payload_bytes,
                self.source_bytes,
                notebook=True,
            )
        )
        self.assertIsNotNone(output_tmp)
        self.assertEqual(evidence["source"]["archiveKind"], "notebookCodeCell")
        self.assertEqual(evidence["source"]["notebookCellIndex"], 0)

    def test_jupytext_shebang_elision_is_accepted_and_full_source_retained(self) -> None:
        payload_bytes = self._payload_bytes()
        evidence, output_dir, output_tmp = self._write(
            archive_bytes=_archive(
                payload_bytes,
                self.source_bytes,
                notebook=True,
                strip_notebook_shebang=True,
            )
        )
        self.assertIsNotNone(output_tmp)
        self.assertEqual(evidence["source"]["archiveKind"], "notebookCodeCell")
        retained = (output_dir / evidence["source"]["file"]).read_bytes()
        self.assertEqual(retained, self.source_bytes)

    def test_archive_confusion_fails_closed(self) -> None:
        payload_bytes = self._payload_bytes()
        cases = {
            "duplicate-payload": (
                _archive(
                    payload_bytes,
                    self.source_bytes,
                    extra_payload=True,
                ),
                "exactly one raw payload",
            ),
            "duplicate-source": (
                _archive(
                    payload_bytes,
                    self.source_bytes,
                    extra_source=True,
                ),
                "exactly one matching task source",
            ),
            "wrong-source": (
                _archive(payload_bytes, self.source_bytes + b"# drift\n"),
                "exactly one matching task source",
            ),
        }
        for name, (archive_bytes, message) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                KaggleCaptureEvidenceError, message
            ):
                self._write(archive_bytes=archive_bytes)

    def test_payload_contract_and_run_model_drift_fail_closed(self) -> None:
        payload = json.loads(self._payload_bytes().decode("utf-8"))
        payload["taskIdentity"]["definitionSha256"] = "0" * 64
        payload["id"] = artifact_id_for(payload)
        drifted_payload = serialize_capture_payload(payload)
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "taskIdentity drifted"
        ):
            self._write(
                archive_bytes=_archive(drifted_payload, self.source_bytes)
            )

        payload_bytes = self._payload_bytes()
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "run model differs"
        ):
            self._write(
                archive_bytes=_archive(payload_bytes, self.source_bytes),
                run_info=_run_info(model="other/model"),
            )
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "run model differs"
        ):
            self._write(
                archive_bytes=_archive(payload_bytes, self.source_bytes),
                run_info=_run_info(model="other/gemini-2.5-flash"),
            )
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "run model differs"
        ):
            self._write(
                archive_bytes=_archive(
                    self._payload_bytes(
                        model_slug="anthropic/claude-haiku-4-5@20251001"
                    ),
                    self.source_bytes,
                ),
                run_info=_run_info(model="claude-haiku-4-5-default"),
            )

    def test_terminal_error_is_retained_without_raw_platform_message(self) -> None:
        payload_bytes = self._payload_bytes()
        evidence, _, output_tmp = self._write(
            archive_bytes=_archive(payload_bytes, self.source_bytes),
            task_info=_task_info(
                state="BENCHMARK_TASK_VERSION_CREATION_STATE_ERRORED"
            ),
            run_info=_run_info(state="BENCHMARK_TASK_RUN_STATE_ERRORED"),
        )
        self.assertIsNotNone(output_tmp)
        self.assertFalse(evidence["assemblyEligible"])
        serialized = json.dumps(evidence)
        self.assertNotIn("platform detail", serialized)
        self.assertNotIn("run detail", serialized)
        self.assertRegex(
            evidence["task"]["creationErrorStringSha256"], r"^[0-9a-f]{64}$"
        )
        self.assertRegex(
            evidence["run"]["errorStringSha256"], r"^[0-9a-f]{64}$"
        )

    def test_run_interval_and_existing_outputs_fail_closed(self) -> None:
        payload_bytes = self._payload_bytes()
        archive_bytes = _archive(payload_bytes, self.source_bytes)
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "starts before"
        ):
            self._write(
                archive_bytes=archive_bytes,
                run_info=_run_info(
                    start=datetime(2026, 9, 22, 0, 0, 30, tzinfo=timezone.utc)
                ),
            )

        output_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(output_tmp.cleanup)
        output_dir = Path(output_tmp.name) / "evidence"
        self._write(archive_bytes=archive_bytes, output_dir=output_dir)
        before = {
            path.name: path.read_bytes() for path in output_dir.iterdir()
        }
        with self.assertRaisesRegex(
            KaggleCaptureEvidenceError, "refusing to overwrite"
        ):
            self._write(archive_bytes=archive_bytes, output_dir=output_dir)
        self.assertEqual(
            {path.name: path.read_bytes() for path in output_dir.iterdir()},
            before,
        )


if __name__ == "__main__":
    unittest.main()
