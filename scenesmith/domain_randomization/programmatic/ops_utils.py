"""Shared utilities for programmatic DR ops."""

from __future__ import annotations

from scenesmith.agent_utils.room import ObjectType


def _resolve_types(type_strings: list[str]) -> set[ObjectType]:
    result = set()
    for s in type_strings:
        try:
            result.add(ObjectType(s))
        except ValueError:
            pass
    return result
