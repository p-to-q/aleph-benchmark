from __future__ import annotations

import hashlib
import fcntl
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .adapters import ModelAdapter
from .protocol import PROTOCOL_VERSION, RESPONSE_CAPTURE_VERSION


MAX_CACHE_RECORD_BYTES = 1_048_576


class ResponseCacheAdapter(ModelAdapter):
    """Persistent per-call cache for resumable hosted benchmark runs."""

    def __init__(self, wrapped: ModelAdapter, cache_dir: Path, *, seed: int):
        super().__init__(
            model_id=wrapped.model_id,
            observation_mode=wrapped.observation_mode,
            temperature=wrapped.temperature,
        )
        self.wrapped = wrapped
        self.cache_dir = cache_dir
        self.seed = seed
        self.cache_hits = 0
        self.cache_misses = 0
        self._last_response_receipt: dict[str, str] | None = None

    def reruns(self, configured: int) -> int:
        return self.wrapped.reruns(configured)

    def evidence_identity(self) -> dict[str, Any]:
        return self.wrapped.evidence_identity()

    def last_response_receipt(self) -> dict[str, str] | None:
        return (
            dict(self._last_response_receipt)
            if self._last_response_receipt is not None
            else None
        )

    def _request_identity(
        self,
        prompt: str,
        item: dict[str, Any],
        ladder_prompt: dict[str, Any],
        *,
        seed: int,
        rerun_index: int,
    ) -> dict[str, Any]:
        if seed != self.seed:
            raise ValueError(
                f"cache seed mismatch: wrapper uses {self.seed}, request uses {seed}"
            )
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "responseCaptureVersion": RESPONSE_CAPTURE_VERSION,
            "adapter": self.wrapped.cache_identity(),
            "seed": seed,
            "rerunIndex": rerun_index,
            "itemId": item["id"],
            "promptId": ladder_prompt["id"],
            "prompt": prompt,
            "target": item["target"]["text"],
        }

    @staticmethod
    def _cache_key(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        model_directory = hashlib.sha256(self.model_id.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / model_directory / f"{key}.json"

    @staticmethod
    def _ensure_private_directory(path: Path) -> None:
        target = Path(os.path.abspath(path))
        current = Path(target.anchor)
        parts = target.parts[1:]
        if not parts:
            raise ValueError("cache directory must not be the filesystem root")

        for index, part in enumerate(parts):
            current /= part
            created = False
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                try:
                    current.mkdir(mode=0o700)
                    created = True
                except FileExistsError:
                    pass
                metadata = current.lstat()

            is_target = index == len(parts) - 1
            if stat.S_ISLNK(metadata.st_mode):
                parent_metadata = current.parent.stat()
                trusted_system_alias = (
                    metadata.st_uid == 0
                    and parent_metadata.st_uid == 0
                    and stat.S_IMODE(parent_metadata.st_mode) & 0o022 == 0
                )
                try:
                    resolved_metadata = current.resolve(strict=True).lstat()
                except (FileNotFoundError, RuntimeError):
                    resolved_metadata = None
                if trusted_system_alias and resolved_metadata is not None:
                    if stat.S_ISDIR(resolved_metadata.st_mode):
                        if (
                            is_target
                            and stat.S_IMODE(resolved_metadata.st_mode) != 0o700
                        ):
                            raise ValueError(
                                "existing cache directory must have mode 0700; "
                                "choose a dedicated cache directory: "
                                f"{target}"
                            )
                        continue
                raise ValueError(
                    "cache path component must be a real directory, not a link: "
                    f"{current}"
                )

            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    "cache path component must be a real directory, not a link: "
                    f"{current}"
                )

            if created and stat.S_IMODE(metadata.st_mode) != 0o700:
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
                    os, "O_NOFOLLOW", 0
                )
                descriptor = os.open(current, flags)
                try:
                    opened = os.fstat(descriptor)
                    if (opened.st_dev, opened.st_ino) != (
                        metadata.st_dev,
                        metadata.st_ino,
                    ):
                        raise ValueError(
                            f"cache directory changed while being created: {current}"
                        )
                    os.fchmod(descriptor, 0o700)
                finally:
                    os.close(descriptor)
                metadata = current.lstat()

            if is_target and stat.S_IMODE(metadata.st_mode) != 0o700:
                raise ValueError(
                    "existing cache directory must have mode 0700; choose a "
                    f"dedicated cache directory: {target}"
                )

    @staticmethod
    def _read_record(path: Path) -> dict[str, Any] | None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"cache entry must be a regular file, not a link: {path}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ValueError(f"could not safely open cache entry {path}: {exc}") from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise ValueError(f"cache entry must be a singly linked regular file: {path}")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            os.close(descriptor)
            raise ValueError(f"existing cache entry must have mode 0600: {path}")
        if metadata.st_size > MAX_CACHE_RECORD_BYTES:
            os.close(descriptor)
            raise ValueError(
                f"cache entry exceeds {MAX_CACHE_RECORD_BYTES} bytes: {path}"
            )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"cache entry {path} must contain a JSON object")
        return loaded

    @staticmethod
    @contextmanager
    def _entry_lock(path: Path) -> Iterator[None]:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        created = False
        try:
            descriptor = os.open(
                path, os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow, 0o600
            )
            created = True
        except FileExistsError:
            try:
                descriptor = os.open(path, os.O_RDWR | nofollow)
            except OSError as exc:
                raise ValueError(
                    f"could not safely open cache lock {path}: {exc}"
                ) from exc
        except OSError as exc:
            raise ValueError(f"could not safely open cache lock {path}: {exc}") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError(f"cache lock must be a singly linked regular file: {path}")
            if created:
                os.fchmod(descriptor, 0o600)
            elif stat.S_IMODE(metadata.st_mode) != 0o600:
                raise ValueError(f"existing cache lock must have mode 0600: {path}")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _write_record(path: Path, record: dict[str, Any]) -> None:
        encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(encoded) > MAX_CACHE_RECORD_BYTES:
            raise ValueError(
                f"cache record exceeds {MAX_CACHE_RECORD_BYTES} bytes: {path}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def generate(
        self,
        prompt: str,
        item: dict[str, Any],
        ladder_prompt: dict[str, Any],
        *,
        seed: int,
        rerun_index: int,
    ) -> str:
        self._last_response_receipt = None
        request_identity = self._request_identity(
            prompt,
            item,
            ladder_prompt,
            seed=seed,
            rerun_index=rerun_index,
        )
        key = self._cache_key(request_identity)
        path = self._path_for_key(key)
        self._ensure_private_directory(self.cache_dir)
        self._ensure_private_directory(path.parent)
        lock_path = path.with_suffix(".lock")
        with self._entry_lock(lock_path):
            cached = self._read_record(path)
            if cached is not None:
                self.cache_hits += 1
                if cached.get("protocolVersion") != PROTOCOL_VERSION:
                    raise ValueError(f"cache entry {path} has an incompatible protocolVersion")
                if cached.get("responseCaptureVersion") != RESPONSE_CAPTURE_VERSION:
                    raise ValueError(f"cache entry {path} has an incompatible responseCaptureVersion")
                if cached.get("key") != key or cached.get("request") != request_identity:
                    raise ValueError(f"cache entry {path} identity does not match its request")
                output = cached.get("output")
                if not isinstance(output, str):
                    raise TypeError(f"cache entry {path} output must be a string")
                captured_at = cached.get("capturedAt")
                if not isinstance(captured_at, str) or not captured_at:
                    raise ValueError(f"cache entry {path} lacks capturedAt")
                self._last_response_receipt = {
                    "capturedAt": captured_at,
                    "source": "cache",
                }
                return output

            self.cache_misses += 1
            output = self.wrapped.generate(
                prompt,
                item,
                ladder_prompt,
                seed=seed,
                rerun_index=rerun_index,
            )
            if not isinstance(output, str):
                raise TypeError(
                    f"adapter {self.wrapped.model_id!r} returned {type(output).__name__}; expected str"
                )
            wrapped_receipt = self.wrapped.last_response_receipt()
            captured_at = (
                wrapped_receipt.get("capturedAt")
                if wrapped_receipt is not None
                else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            capture_source = (
                wrapped_receipt.get("source")
                if wrapped_receipt is not None
                else "adapter"
            )
            if not isinstance(captured_at, str) or not captured_at:
                raise ValueError("adapter response receipt lacks capturedAt")
            if not isinstance(capture_source, str) or not capture_source:
                raise ValueError("adapter response receipt lacks source")
            record = {
                "protocolVersion": PROTOCOL_VERSION,
                "responseCaptureVersion": RESPONSE_CAPTURE_VERSION,
                "key": key,
                "request": request_identity,
                "model": self.model_id,
                "observationMode": self.observation_mode,
                "seed": self.seed,
                "rerunIndex": rerun_index,
                "itemId": item["id"],
                "promptId": ladder_prompt["id"],
                "capturedAt": captured_at,
                "captureSource": capture_source,
                "output": output,
            }
            self._write_record(path, record)
            self._last_response_receipt = {
                "capturedAt": captured_at,
                "source": capture_source,
            }
            return output
