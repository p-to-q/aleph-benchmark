from __future__ import annotations

import re

from .mock import PROFILES


_PROVIDER_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_DEPLOYMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}\Z")


def validate_deployment_id(value: str) -> str:
    if not isinstance(value, str) or not _DEPLOYMENT_ID.fullmatch(value):
        raise ValueError(
            "hosted deployment id must use 1-256 ASCII letters, digits, "
            "._:/@- characters and start with a letter or digit"
        )
    return value


def normalize_model_id(model: str) -> tuple[str, str, bool]:
    """Return canonical id, evidence mode, and whether configured reruns apply."""

    if not isinstance(model, str) or not model:
        raise ValueError("model id must be a non-empty string")
    if model.startswith("mock-"):
        if model not in PROFILES:
            raise ValueError(f"unknown mock model: {model}")
        return model, "mock", False
    if model.startswith("hosted:"):
        provider_model = model.removeprefix("hosted:")
        if provider_model.startswith("hosted:"):
            raise ValueError("hosted model id must contain exactly one hosted: prefix")
        if not _PROVIDER_MODEL_ID.fullmatch(provider_model):
            raise ValueError(
                "hosted provider model id must use 1-256 ASCII letters, digits, "
                "_:/- characters and start with a letter or digit"
            )
        return f"hosted:{provider_model}", "black_box", True
    raise ValueError(f"unsupported model id {model!r}; use a known mock-* or hosted:<model>")
