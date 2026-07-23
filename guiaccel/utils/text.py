"""Text normalization and lexical alignment helpers."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = " ".join(text.strip().split())
    return normalized.lower() if normalized else None


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def lexical_alignment(
    *,
    goal: str,
    screen_text: str | None,
    action_argument: str | None,
    action_text: str | None,
    target_text: str | None,
) -> dict[str, list[str]]:
    goal_tokens = unique_preserve_order(tokenize(goal))
    screen_tokens = unique_preserve_order(tokenize(screen_text))
    argument_tokens = unique_preserve_order(tokenize(action_argument))
    action_tokens = unique_preserve_order(tokenize(action_text))
    target_tokens = unique_preserve_order(tokenize(target_text))
    screen_union = unique_preserve_order(screen_tokens + target_tokens)

    return {
        "goal_tokens": goal_tokens,
        "screen_tokens": screen_union,
        "argument_tokens": argument_tokens,
        "action_tokens": action_tokens,
        "goal_screen_overlap": sorted(set(goal_tokens) & set(screen_union)),
        "goal_argument_overlap": sorted(set(goal_tokens) & set(argument_tokens)),
        "screen_argument_overlap": sorted(set(screen_union) & set(argument_tokens)),
        "action_screen_overlap": sorted(set(action_tokens) & set(screen_union)),
    }
