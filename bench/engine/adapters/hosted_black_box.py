from __future__ import annotations

import http.client
import json
import hashlib
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .base import ModelAdapter
from .identity import validate_deployment_id
from ..scoring_core import MAX_SCORING_TEXT_CHARACTERS

RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
# Worst-case JSON surrogate escaping for 16,384 astral code points is 196,608
# bytes before the response envelope. Keep a bounded 64 KiB envelope allowance.
MAX_RESPONSE_BODY_BYTES = 262_144
DEFAULT_MAX_RETRIES = 2
MAX_HOSTED_RETRIES = 5
DEFAULT_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT_SECONDS = 90
HOSTED_ADAPTER_ID = "aleph.hosted-openai-compatible"
HOSTED_ADAPTER_VERSION = "1"
HOSTED_WIRE_PROTOCOL = "openai-compatible-chat-completions-v1"
HOSTED_REQUEST_PAYLOAD_VERSION = "chat-completions-user-prompt-v1"


class HostedBlackBoxError(RuntimeError):
    pass


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep credentials bound to the configured, validated endpoint."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc


def configured_hosted_max_retries() -> int:
    value = env_int("ALEPH_CUSTOM_API_MAX_RETRIES", DEFAULT_MAX_RETRIES)
    if isinstance(value, bool) or not 0 <= value <= MAX_HOSTED_RETRIES:
        raise RuntimeError(
            "ALEPH_CUSTOM_API_MAX_RETRIES must be between 0 and "
            f"{MAX_HOSTED_RETRIES}"
        )
    return value


def configured_hosted_retry_delay_seconds() -> float:
    value = env_float(
        "ALEPH_CUSTOM_API_RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY_SECONDS
    )
    if not math.isfinite(value) or not 0.0 <= value <= MAX_RETRY_DELAY_SECONDS:
        raise RuntimeError(
            "ALEPH_CUSTOM_API_RETRY_DELAY_SECONDS must be finite and between 0 and "
            f"{MAX_RETRY_DELAY_SECONDS}"
        )
    return value


def validate_hosted_base_url(value: str) -> str:
    normalized = value.rstrip("/")
    if not normalized or any(char.isspace() for char in normalized):
        raise RuntimeError("ALEPH_CUSTOM_API_BASE_URL must be an absolute HTTP(S) URL")
    try:
        parsed = urllib.parse.urlsplit(normalized)
        _ = parsed.port
    except ValueError as exc:
        raise RuntimeError(
            "ALEPH_CUSTOM_API_BASE_URL must be an absolute HTTP(S) URL"
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "?" in normalized
        or "#" in normalized
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "ALEPH_CUSTOM_API_BASE_URL must be an absolute HTTP(S) URL without "
            "credentials, query, or fragment"
        )
    loopback_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "http" and parsed.hostname.lower() not in loopback_hosts:
        raise RuntimeError(
            "ALEPH_CUSTOM_API_BASE_URL must use HTTPS except for localhost loopback"
        )
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        raise RuntimeError(
            "ALEPH_CUSTOM_API_BASE_URL must be the API base prefix; the adapter "
            "appends /chat/completions"
        )
    return normalized


def validate_hosted_api_key(value: str) -> str:
    """Keep secrets out of header-encoding errors and request-line injection."""

    if not isinstance(value, str) or not value or any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in value
    ):
        raise RuntimeError(
            "ALEPH_CUSTOM_API_KEY must be non-empty printable ASCII without whitespace"
        )
    return value


class HostedBlackBoxAdapter(ModelAdapter):
    """OpenAI-compatible black-box adapter for future real M0 runs."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        deployment_id: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int | None = None,
        retry_delay_seconds: float | None = None,
    ):
        provider_model_id = model_id.removeprefix("hosted:")
        if not provider_model_id:
            raise ValueError("hosted model id must not be empty")
        super().__init__(
            model_id=f"hosted:{provider_model_id}",
            observation_mode="black_box",
            temperature=temperature,
        )
        self.provider_model_id = provider_model_id
        raw_base_url = base_url or os.environ.get("ALEPH_CUSTOM_API_BASE_URL") or ""
        self.base_url = (
            validate_hosted_base_url(raw_base_url) if raw_base_url else ""
        )
        raw_api_key = (
            api_key
            if api_key is not None
            else os.environ.get("ALEPH_CUSTOM_API_KEY") or ""
        )
        self.api_key = validate_hosted_api_key(raw_api_key) if raw_api_key else ""
        raw_deployment_id = (
            deployment_id or os.environ.get("ALEPH_CUSTOM_API_DEPLOYMENT_ID") or ""
        )
        self.deployment_id = (
            validate_deployment_id(raw_deployment_id) if raw_deployment_id else ""
        )
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = (
            max_retries
            if max_retries is not None
            else configured_hosted_max_retries()
        )
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or not 0 <= self.max_retries <= MAX_HOSTED_RETRIES
        ):
            raise RuntimeError(
                f"hosted max_retries must be between 0 and {MAX_HOSTED_RETRIES}"
            )
        self.retry_delay_seconds = (
            retry_delay_seconds
            if retry_delay_seconds is not None
            else configured_hosted_retry_delay_seconds()
        )
        if (
            isinstance(self.retry_delay_seconds, bool)
            or not isinstance(self.retry_delay_seconds, (int, float))
            or not math.isfinite(self.retry_delay_seconds)
            or not 0.0 <= self.retry_delay_seconds <= MAX_RETRY_DELAY_SECONDS
        ):
            raise RuntimeError(
                "hosted retry_delay_seconds must be finite and between 0 and "
                f"{MAX_RETRY_DELAY_SECONDS}"
            )
        if not self.base_url or not self.api_key or not self.deployment_id:
            raise RuntimeError(
                "hosted black-box adapter requires ALEPH_CUSTOM_API_BASE_URL, "
                "ALEPH_CUSTOM_API_KEY, and ALEPH_CUSTOM_API_DEPLOYMENT_ID"
            )
        self._opener = urllib.request.build_opener(_RejectRedirectHandler())
        self._last_response_receipt: dict[str, str] | None = None

    def cache_identity(self) -> dict[str, Any]:
        identity = super().cache_identity()
        endpoint = f"{self.base_url}/chat/completions".encode("utf-8")
        identity.update(
            {
                "adapterId": HOSTED_ADAPTER_ID,
                "adapterVersion": HOSTED_ADAPTER_VERSION,
                "wireProtocol": HOSTED_WIRE_PROTOCOL,
                "requestPayloadVersion": HOSTED_REQUEST_PAYLOAD_VERSION,
                "endpointSha256": hashlib.sha256(endpoint).hexdigest(),
                "credentialScopeSha256": hashlib.sha256(
                    self.api_key.encode("utf-8")
                ).hexdigest(),
                "maxTokens": self.max_tokens,
                "providerModel": self.provider_model_id,
                "deploymentId": self.deployment_id,
                "timeoutSeconds": self.timeout_seconds,
                "maxRetries": self.max_retries,
                "retryDelaySeconds": self.retry_delay_seconds,
            }
        )
        return identity

    def evidence_identity(self) -> dict[str, Any]:
        identity = self.cache_identity()
        identity.pop("credentialScopeSha256", None)
        return identity

    def last_response_receipt(self) -> dict[str, str] | None:
        return (
            dict(self._last_response_receipt)
            if self._last_response_receipt is not None
            else None
        )

    def reruns(self, configured: int) -> int:
        """Repeat hosted calls even at temperature zero to measure nondeterminism."""

        return max(1, configured)

    def _request_for_payload(self, payload: dict[str, Any]) -> urllib.request.Request:
        body = json.dumps(payload).encode("utf-8")
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    def _sleep_before_retry(self) -> None:
        if self.retry_delay_seconds > 0:
            time.sleep(self.retry_delay_seconds)

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            request = self._request_for_payload(payload)
            try:
                with self._opener.open(request, timeout=self.timeout_seconds) as response:
                    body = response.read(MAX_RESPONSE_BODY_BYTES + 1)
                    if len(body) > MAX_RESPONSE_BODY_BYTES:
                        raise HostedBlackBoxError(
                            "hosted black-box response exceeded "
                            f"{MAX_RESPONSE_BODY_BYTES} bytes"
                        )
                    try:
                        parsed = json.loads(body.decode("utf-8"))
                    except (UnicodeError, json.JSONDecodeError):
                        raise HostedBlackBoxError(
                            "hosted black-box response was not valid UTF-8 JSON"
                        ) from None
                    if not isinstance(parsed, dict):
                        raise HostedBlackBoxError(
                            "hosted black-box response must be a JSON object"
                        )
                    return parsed
            except urllib.error.HTTPError as exc:
                if 300 <= exc.code < 400:
                    raise HostedBlackBoxError(
                        "hosted black-box request failed: redirects are not allowed "
                        f"(HTTP {exc.code})"
                    ) from None
                # Provider-controlled error bodies can echo submitted credentials.
                # Consume only a bounded amount and never propagate the body to logs.
                try:
                    exc.read(MAX_RESPONSE_BODY_BYTES + 1)
                except (OSError, ValueError, http.client.HTTPException):
                    pass
                last_error = f"HTTP {exc.code}"
                if exc.code in RETRYABLE_HTTP_STATUSES and attempt < self.max_retries:
                    self._sleep_before_retry()
                    continue
                raise HostedBlackBoxError(
                    f"hosted black-box request failed: {last_error}"
                ) from None
            except urllib.error.URLError:
                last_error = "transport error"
                if attempt < self.max_retries:
                    self._sleep_before_retry()
                    continue
                raise HostedBlackBoxError(
                    f"hosted black-box request failed: {last_error}"
                ) from None
            except http.client.HTTPException:
                raise HostedBlackBoxError(
                    "hosted black-box request failed: malformed or incomplete HTTP response"
                ) from None
            except OSError:
                last_error = "transport error"
                if attempt < self.max_retries:
                    self._sleep_before_retry()
                    continue
                raise HostedBlackBoxError(
                    f"hosted black-box request failed: {last_error}"
                ) from None
        raise HostedBlackBoxError(f"hosted black-box request failed: {last_error or 'unknown error'}")

    def generate(
        self,
        prompt: str,
        item: dict[str, Any],
        ladder_prompt: dict[str, Any],
        *,
        seed: int,
        rerun_index: int,
    ) -> str:
        del item, ladder_prompt, seed, rerun_index
        self._last_response_receipt = None
        payload = {
            "model": self.provider_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            data = self._post_chat_completion(payload)
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise HostedBlackBoxError(
                    "hosted black-box response must contain choices[0].message.content"
                ) from None
            if not isinstance(content, str):
                raise HostedBlackBoxError(
                    "hosted black-box response choices[0].message.content must be a string"
                )
            if len(content) > MAX_SCORING_TEXT_CHARACTERS:
                raise HostedBlackBoxError(
                    "hosted black-box response content exceeded "
                    f"{MAX_SCORING_TEXT_CHARACTERS} characters"
                )
            self._last_response_receipt = {
                "capturedAt": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "source": "provider",
            }
            return content
        except (KeyError, IndexError, TypeError) as exc:
            raise HostedBlackBoxError("hosted black-box response missing choices[0].message.content") from exc
