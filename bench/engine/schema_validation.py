from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


class SchemaValidationError(ValueError):
    pass


def _enum_value_equal(instance: Any, candidate: Any) -> bool:
    """Match JSON enum semantics without Python's True == 1 coercion."""

    if isinstance(instance, bool) or isinstance(candidate, bool):
        return type(instance) is type(candidate) and instance == candidate
    if isinstance(instance, dict) or isinstance(candidate, dict):
        return (
            isinstance(instance, dict)
            and isinstance(candidate, dict)
            and set(instance) == set(candidate)
            and all(
                _enum_value_equal(instance[key], candidate[key]) for key in instance
            )
        )
    if isinstance(instance, list) or isinstance(candidate, list):
        return (
            isinstance(instance, list)
            and isinstance(candidate, list)
            and len(instance) == len(candidate)
            and all(
                _enum_value_equal(left, right)
                for left, right in zip(instance, candidate)
            )
        )
    return instance == candidate


def load_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(instance: Any, schema: dict[str, Any], root: dict[str, Any] | None = None, path: str = "$") -> None:
    root = schema if root is None else root
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise SchemaValidationError(f"{path}: external refs are not supported: {ref}")
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        validate(instance, target, root, path)
        return

    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                validate(instance, option, root, path)
                return
            except SchemaValidationError as exc:
                errors.append(str(exc))
        raise SchemaValidationError(f"{path}: did not match any allowed schema: {errors[0] if errors else ''}")

    if "allOf" in schema:
        for option in schema["allOf"]:
            validate(instance, option, root, path)

    if "not" in schema:
        try:
            validate(instance, schema["not"], root, path)
        except SchemaValidationError:
            pass
        else:
            raise SchemaValidationError(f"{path}: matched a forbidden schema")

    if "if" in schema:
        try:
            validate(instance, schema["if"], root, path)
        except SchemaValidationError:
            if "else" in schema:
                validate(instance, schema["else"], root, path)
        else:
            if "then" in schema:
                validate(instance, schema["then"], root, path)

    if "enum" in schema and not any(
        _enum_value_equal(instance, candidate) for candidate in schema["enum"]
    ):
        raise SchemaValidationError(f"{path}: {instance!r} not in enum {schema['enum']!r}")
    if "const" in schema and not _enum_value_equal(instance, schema["const"]):
        raise SchemaValidationError(
            f"{path}: {instance!r} does not equal const {schema['const']!r}"
        )

    expected = schema.get("type")
    if expected:
        if expected == "object":
            if not isinstance(instance, dict):
                raise SchemaValidationError(f"{path}: expected object")
            required = schema.get("required", [])
            for key in required:
                if key not in instance:
                    raise SchemaValidationError(f"{path}: missing required key {key!r}")
            properties = schema.get("properties", {})
            if schema.get("additionalProperties") is False:
                extra = set(instance) - set(properties)
                if extra:
                    raise SchemaValidationError(f"{path}: unexpected keys {sorted(extra)!r}")
            additional = schema.get("additionalProperties")
            for key, value in instance.items():
                if key in properties:
                    validate(value, properties[key], root, f"{path}.{key}")
                elif isinstance(additional, dict):
                    validate(value, additional, root, f"{path}.{key}")
            return
        if expected == "array":
            if not isinstance(instance, list):
                raise SchemaValidationError(f"{path}: expected array")
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            if min_items is not None and len(instance) < min_items:
                raise SchemaValidationError(f"{path}: expected at least {min_items} items")
            if max_items is not None and len(instance) > max_items:
                raise SchemaValidationError(f"{path}: expected at most {max_items} items")
            if schema.get("uniqueItems") and any(
                _enum_value_equal(instance[left], instance[right])
                for left in range(len(instance))
                for right in range(left + 1, len(instance))
            ):
                raise SchemaValidationError(f"{path}: expected unique array items")
            item_schema = schema.get("items")
            if item_schema:
                for index, value in enumerate(instance):
                    validate(value, item_schema, root, f"{path}[{index}]")
            return
        if expected == "string":
            if not isinstance(instance, str):
                raise SchemaValidationError(f"{path}: expected string")
            if "minLength" in schema and len(instance) < schema["minLength"]:
                raise SchemaValidationError(
                    f"{path}: expected at least {schema['minLength']} characters"
                )
            if "maxLength" in schema and len(instance) > schema["maxLength"]:
                raise SchemaValidationError(
                    f"{path}: expected at most {schema['maxLength']} characters"
                )
        if expected == "integer" and not (isinstance(instance, int) and not isinstance(instance, bool)):
            raise SchemaValidationError(f"{path}: expected integer")
        if expected == "number" and not (
            (isinstance(instance, int) or isinstance(instance, float)) and not isinstance(instance, bool)
        ):
            raise SchemaValidationError(f"{path}: expected number")
        if expected == "boolean" and not isinstance(instance, bool):
            raise SchemaValidationError(f"{path}: expected boolean")
        if expected == "null" and instance is not None:
            raise SchemaValidationError(f"{path}: expected null")

    if isinstance(instance, str) and "pattern" in schema:
        if re.search(schema["pattern"], instance) is None:
            raise SchemaValidationError(f"{path}: {instance!r} does not match pattern {schema['pattern']!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if isinstance(instance, float) and not math.isfinite(instance):
            raise SchemaValidationError(f"{path}: expected a finite number")
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaValidationError(f"{path}: {instance} below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaValidationError(f"{path}: {instance} above maximum {schema['maximum']}")


def validate_file(instance_path: str | Path, schema_path: str | Path) -> None:
    validate(json.loads(Path(instance_path).read_text(encoding="utf-8")), load_schema(schema_path))
