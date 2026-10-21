#!/usr/bin/env python3
"""Generate the adult's next turn for the listener study.

Providers: openai, anthropic, gemini, lmstudio. A run uses the models listed
in config.yaml. The pilot runs one provider at a time. config.yaml is not
created here: the OpenAI model id is chosen first.

Keys are read from OPENAI_API_KEY, ANTHROPIC_API_KEY, and GEMINI_API_KEY.
They are never written to a file. LM Studio is http://localhost:1234/v1.

Rows with exclude true are skipped. The pilot draws 50 usable items,
stratified on corpus x trouble_class with seed 42, and writes
data/processed/pilot_items.txt. Later models read that file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.items import load_items, usable_items

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

CONDITIONS = ("plain", "child", "repair")
PROMPTS = {
    "plain": (
        "You are taking part in a conversation. Write the next turn of the adult "
        "speaker only, as they would naturally say it. Output only the words of that turn."
    ),
    "child": (
        "You are an adult talking with a young child aged {age}. Write your next turn "
        "only, as you would naturally say it to the child. Output only the words of that turn."
    ),
    "repair": (
        "You are an adult talking with a young child aged {age}. If you are not sure "
        "what the child meant, check your understanding with the child before continuing. "
        "Write your next turn only, as you would naturally say it to the child. "
        "Output only the words of that turn."
    ),
}

# A line that starts a new speaker turn. "Other child" is checked before "Other".
SPEAKER_LINE = re.compile(
    r"^(?:Adult|Child|Other child|Other|Mother|Father|Mom|Dad|Mummy|Daddy|"
    r"Investigator|Grandmother|Grandfather|Sibling|Sister|Brother)\s*:",
    re.IGNORECASE,
)
LEADING_ADULT = re.compile(r"^\s*Adult:\s*", re.IGNORECASE)
# Whole words only. A substring such as the letters "ai" inside "said" does not match.
META_COMMENTARY = re.compile(
    r"("
    r"\bas an ai\b|"
    r"\bi am an ai\b|"
    r"\bi'm an ai\b|"
    r"\bi’m an ai\b|"
    r"\blanguage model\b|"
    r"\bas an artificial intelligence\b|"
    r"\bhere(?:'s| is) (?:a |the )?(?:possible )?(?:reply|response|turn)\b|"
    r"\bmy (?:reply|response)\b|"
    r"\bthe next turn\b|"
    r"\bi would say\b"
    r")",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\S+")
LENGTH_REASONS = {"length", "max_tokens", "MAX_TOKENS"}
REFUSAL_REASONS = {
    "content_filter",
    "refusal",
    "SAFETY",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "RECITATION",
}


class APIError(Exception):
    def __init__(self, status: int, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def format_age(months: float | None) -> str:
    """CHAT age in months, with the day fraction dropped. 30.5 is 2 years 6 months."""
    if months is None:
        return "unknown age"
    years = int(months // 12)
    remaining = int(months % 12)
    parts: list[str] = []
    if years:
        parts.append("1 year" if years == 1 else f"{years} years")
    if remaining or not parts:
        parts.append("1 month" if remaining == 1 else f"{remaining} months")
    return " ".join(parts)


def system_prompt(condition: str, age: str) -> str:
    template = PROMPTS[condition]
    return template.format(age=age) if "{age}" in template else template


def parse_reply(raw: str | None) -> tuple[str, bool, bool]:
    """Return the listener reply, whether a later speaker turn was cut, and meta.

    A leading "Adult:" is this turn's label and is removed. Any later line that
    starts with a speaker label, and everything after it, is dropped.
    """
    text = "" if raw is None else str(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = LEADING_ADULT.sub("", text, count=1)
    kept: list[str] = []
    multi = False
    for line in text.split("\n"):
        if SPEAKER_LINE.match(line.strip()):
            multi = True
            break
        kept.append(line)
    reply = "\n".join(kept).strip()
    meta = META_COMMENTARY.search("" if raw is None else str(raw)) is not None
    return reply, multi, meta


def clip_tokens(text: str, limit: int) -> tuple[str, bool]:
    """Cap the stored reply at `limit` whitespace-separated tokens."""
    matches = list(WORD_RE.finditer(text))
    if len(matches) <= limit:
        return text, False
    return text[: matches[limit - 1].end()].rstrip(), True


THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
# Claude Sonnet 5.5 rejects thinking disabled and budget_tokens. between_tools
# is its lowest setting and does no up-front thinking when the request has no tools.
CLAUDE_THINKING = "between_tools"
CLAUDE_EFFORT = "low"
CLAUDE_THINKING_BUDGET = 0


def claude_max_tokens(visible_tokens: int) -> int:
    """Leave `visible_tokens` for the reply after the thinking budget."""
    return CLAUDE_THINKING_BUDGET + visible_tokens


def strip_thinking(text: str) -> tuple[str, bool]:
    """Drop Qwen think blocks. The flag is true when a block contained text."""
    had = False

    def replace(match: re.Match) -> str:
        nonlocal had
        if match.group(1).strip():
            had = True
        return ""

    cleaned = THINK_BLOCK.sub(replace, "" if text is None else str(text))
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(), had


def listener_messages(system: str, user: str, *, system_delivery: str) -> list[dict]:
    """Gemma has no system role, so its instruction goes in the user message."""
    if system_delivery == "user_message":
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def stratify_pilot(items: list[dict], n: int = 50, seed: int = 42) -> list[dict]:
    """n usable items, proportional to corpus x trouble_class. Seed 42."""
    usable = usable_items(items)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in usable:
        strata[(item["corpus"], item["trouble_class"])].append(item)
    if not strata:
        raise ValueError("no usable items")
    total = len(usable)
    keys = sorted(strata)
    raw = {key: n * len(strata[key]) / total for key in keys}
    quota = {key: min(len(strata[key]), math.floor(raw[key])) for key in keys}
    left = n - sum(quota.values())
    order = sorted(keys, key=lambda key: (-(raw[key] - math.floor(raw[key])), key))
    for key in order:
        if left <= 0:
            break
        if quota[key] >= len(strata[key]):
            continue
        quota[key] += 1
        left -= 1
    if left > 0:
        raise ValueError(f"not enough usable items to draw {n}")
    rng = random.Random(seed)
    chosen: list[dict] = []
    for key in keys:
        pool = list(strata[key])
        rng.shuffle(pool)
        chosen.extend(pool[: quota[key]])
    return chosen


def projected_cost(actual: float, pilot_items: int, usable: int, conditions: int = 3) -> tuple[float, float]:
    """Scale the pilot bill to every usable item, and to subsample 0.33."""
    if pilot_items <= 0:
        return 0.0, 0.0
    per_item = actual / pilot_items
    full = per_item * usable
    return full, full * 0.33


def _scalar(text: str):
    text = text.strip()
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "Null", "~"}:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def load_config(path: Path) -> dict:
    """Read the study config. The file is a small YAML subset: maps and lists."""
    if not path.exists():
        raise SystemExit(
            f"{path} is not written yet. Choose an OpenAI model id before writing it."
        )
    rows: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        rows.append((len(stripped) - len(stripped.lstrip(" ")), stripped.strip()))

    def parse_map(index: int, indent: int) -> tuple[dict, int]:
        result: dict = {}
        while index < len(rows):
            level, text = rows[index]
            if level < indent or text.startswith("- "):
                break
            if level != indent:
                raise SystemExit(f"bad indent in {path}: {text}")
            key, sep, rest = text.partition(":")
            if not sep:
                raise SystemExit(f"expected key: value in {path}: {text}")
            index += 1
            rest = rest.strip()
            if rest:
                result[key.strip()] = _scalar(rest)
            elif index < len(rows) and rows[index][0] > level:
                child_level, child_text = rows[index]
                if child_text.startswith("- "):
                    result[key.strip()], index = parse_list(index, child_level)
                else:
                    result[key.strip()], index = parse_map(index, child_level)
            else:
                result[key.strip()] = None
        return result, index

    def parse_list(index: int, indent: int) -> tuple[list, int]:
        result: list = []
        while index < len(rows) and rows[index][0] == indent and rows[index][1].startswith("- "):
            level, text = rows[index]
            item = text[2:].strip()
            index += 1
            if ":" in item:
                key, _, rest = item.partition(":")
                mapping: dict = {key.strip(): _scalar(rest) if rest.strip() else None}
                while index < len(rows) and rows[index][0] > level and not rows[index][1].startswith("- "):
                    child = rows[index][1]
                    child_key, child_sep, child_rest = child.partition(":")
                    if not child_sep:
                        raise SystemExit(f"expected key: value in {path}: {child}")
                    mapping[child_key.strip()] = _scalar(child_rest) if child_rest.strip() else None
                    index += 1
                result.append(mapping)
            else:
                result.append(_scalar(item))
        return result, index

    if not rows:
        raise SystemExit(f"{path} is empty")
    config, used = parse_map(0, rows[0][0])
    if used != len(rows):
        raise SystemExit(f"could not parse all of {path}")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise SystemExit(f"{path} needs a models list")
    for model in models:
        if not isinstance(model, dict) or not model.get("id") or not model.get("provider"):
            raise SystemExit(f"each model needs id and provider: {model}")
        if model["provider"] not in {"openai", "anthropic", "gemini", "lmstudio"}:
            raise SystemExit(f"unknown provider {model['provider']}")
    return config


def _redact(message: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def _post_json(url: str, body: dict, headers: dict, secrets: list[str], timeout: int = 120) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        message = _redact(f"HTTP {exc.code}: {detail}", secrets)
        retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
        raise APIError(exc.code, message, retryable=retryable) from None
    except urllib.error.URLError as exc:
        raise APIError(0, _redact(str(exc.reason), secrets), retryable=True) from None


def _with_backoff(fn, attempts: int, initial: float, multiplier: float, sleep=time.sleep):
    delay = initial
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except APIError as exc:
            last = exc
            if not exc.retryable or attempt == attempts - 1:
                raise
            sleep(delay)
            delay *= multiplier
    raise last if last else RuntimeError("request failed")


@dataclass
class CallResult:
    raw_reply: str
    finish_reason: str
    input_tokens: int | None
    output_tokens: int | None
    refused: bool
    refusal_reason: str
    temperature: float | None
    reasoning_effort: str | None
    thinking_setting: str | None = None
    thinking_budget: int | None = None
    visible_output_tokens: int | None = None
    system_delivery: str | None = None
    thinking_output: bool = False


class Provider:
    name = ""

    def __init__(self, model: dict, config: dict) -> None:
        self.model = model
        self.config = config
        self.model_id = str(model["id"])
        self.reasoning = bool(model.get("reasoning"))
        self.max_output_tokens = int(config.get("max_output_tokens", 60))
        self.effort = str(config.get("reasoning_effort", "minimal"))
        self.reserve = int(config.get("reasoning_reserve", 256))
        self.attempts = int(config.get("max_attempts", 5))
        self.initial = float(config.get("retry_initial_seconds", 2))
        self.multiplier = float(config.get("retry_multiplier", 2))

    def limit(self, reserve_multiplier: int) -> int:
        if not self.reasoning:
            return self.max_output_tokens
        return self.max_output_tokens + self.reserve * reserve_multiplier

    def generate(self, system: str, user: str, *, reserve_multiplier: int = 1) -> CallResult:
        raise NotImplementedError


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: dict, config: dict) -> None:
        super().__init__(model, config)
        self.api_key = _env("OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/chat/completions"

    def generate(self, system: str, user: str, *, reserve_multiplier: int = 1) -> CallResult:
        body: dict = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.limit(reserve_multiplier),
            "temperature": 0,
        }
        if self.reasoning:
            body["reasoning_effort"] = self.effort
        temperature: float | None = 0
        effort: str | None = self.effort if self.reasoning else None
        payload = self._send(body)
        if payload.get("_dropped_temperature"):
            temperature = None
        if payload.get("_dropped_reasoning"):
            effort = None
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        refusal = message.get("refusal") or ""
        content = message.get("content") or ""
        finish = str(choice.get("finish_reason") or "")
        usage = payload.get("usage") or {}
        refused = bool(refusal) or finish in REFUSAL_REASONS
        if refused and finish not in REFUSAL_REASONS:
            finish = "refusal"
        return CallResult(
            raw_reply=refusal or content,
            finish_reason=finish or ("refusal" if refused else ""),
            input_tokens=_int_or_none(usage.get("prompt_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            refused=refused,
            refusal_reason=refusal or (finish if finish in REFUSAL_REASONS else ""),
            temperature=temperature,
            reasoning_effort=effort,
        )

    def _send(self, body: dict) -> dict:
        def once(current: dict) -> dict:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            return _with_backoff(
                lambda: _post_json(self.url, current, headers, [self.api_key]),
                self.attempts,
                self.initial,
                self.multiplier,
            )

        try:
            return once(body)
        except APIError as exc:
            if exc.status != 400:
                raise
            message = str(exc).lower()
            if "temperature" in message and "temperature" in body:
                dropped = dict(body)
                dropped.pop("temperature", None)
                payload = once(dropped)
                payload["_dropped_temperature"] = True
                return payload
            if "max_completion_tokens" in message and "max_completion_tokens" in body:
                dropped = dict(body)
                dropped["max_tokens"] = dropped.pop("max_completion_tokens")
                return once(dropped)
            if "reasoning_effort" in message and "reasoning_effort" in body and not self.reasoning:
                dropped = dict(body)
                dropped.pop("reasoning_effort", None)
                payload = once(dropped)
                payload["_dropped_reasoning"] = True
                return payload
            raise


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: dict, config: dict) -> None:
        super().__init__(model, config)
        self.api_key = _env("ANTHROPIC_API_KEY")
        self.url = "https://api.anthropic.com/v1/messages"
        self._temperature: float | None = 0

    def generate(self, system: str, user: str, *, reserve_multiplier: int = 1) -> CallResult:
        # between_tools is the lowest setting Sonnet 5.5 accepts. It has no
        # token budget, so the budget in max_tokens is 0 and the reply keeps 60.
        visible_cap = self.max_output_tokens
        max_tokens = claude_max_tokens(visible_cap)
        setting = f"{CLAUDE_THINKING}, effort {CLAUDE_EFFORT}"
        body: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "thinking": {"type": CLAUDE_THINKING},
            "output_config": {"effort": CLAUDE_EFFORT},
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        temperature = self._temperature
        if temperature is not None:
            body["temperature"] = temperature

        def once(current: dict) -> dict:
            return _with_backoff(
                lambda: _post_json(
                    self.url,
                    current,
                    {
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    [self.api_key],
                ),
                self.attempts,
                self.initial,
                self.multiplier,
            )

        try:
            payload = once(body)
        except APIError as exc:
            if exc.status == 400 and "temperature" in str(exc).lower():
                body.pop("temperature", None)
                temperature = None
                self._temperature = None
                payload = once(body)
            else:
                raise
        parts = payload.get("content") or []
        text = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("type") == "text"
        )
        thinking_parts = [
            part for part in parts if isinstance(part, dict) and part.get("type") == "thinking"
        ]
        finish = str(payload.get("stop_reason") or "")
        usage = payload.get("usage") or {}
        output_tokens = _int_or_none(usage.get("output_tokens"))
        visible_tokens = output_tokens if not thinking_parts else None
        refused = finish == "refusal"
        return CallResult(
            raw_reply=text,
            finish_reason=finish,
            input_tokens=_int_or_none(usage.get("input_tokens")),
            output_tokens=output_tokens,
            refused=refused,
            refusal_reason=finish if refused else "",
            temperature=temperature,
            reasoning_effort=setting,
            thinking_setting=setting,
            thinking_budget=CLAUDE_THINKING_BUDGET,
            visible_output_tokens=visible_tokens,
            system_delivery="system",
            thinking_output=bool(thinking_parts),
        )


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: dict, config: dict) -> None:
        super().__init__(model, config)
        self.api_key = _env("GEMINI_API_KEY")
        self._accepted_thinking = {"thinkingBudget": 0}
        self._thinking_label = "thinkingBudget 0"

    def _thinking_config(self) -> dict:
        return self._accepted_thinking

    def generate(self, system: str, user: str, *, reserve_multiplier: int = 1) -> CallResult:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent"
        )
        generation: dict = {
            "temperature": 0,
            "maxOutputTokens": self.max_output_tokens,
            "thinkingConfig": dict(self._thinking_config()),
        }
        effort = self._thinking_label
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        temperature: float | None = 0

        def once(current: dict) -> dict:
            return _with_backoff(
                lambda: _post_json(
                    url,
                    current,
                    {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                    [self.api_key],
                ),
                self.attempts,
                self.initial,
                self.multiplier,
            )

        try:
            payload = once(body)
        except APIError as exc:
            message = str(exc).lower()
            if exc.status == 400 and "temperature" in message:
                body["generationConfig"].pop("temperature", None)
                temperature = None
                payload = once(body)
            elif exc.status == 400 and "thinking" in message and self._thinking_label == "thinkingBudget 0":
                self._accepted_thinking = {"thinkingLevel": "low"}
                self._thinking_label = "thinkingLevel low; thinkingBudget 0 was rejected"
                body["generationConfig"]["thinkingConfig"] = dict(self._accepted_thinking)
                effort = self._thinking_label
                payload = once(body)
            else:
                raise
        prompt_feedback = payload.get("promptFeedback") or {}
        block = str(prompt_feedback.get("blockReason") or "")
        candidates = payload.get("candidates") or []
        candidate = candidates[0] if candidates else {}
        finish = str(candidate.get("finishReason") or block or "")
        parts = ((candidate.get("content") or {}).get("parts")) or []
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        usage = payload.get("usageMetadata") or {}
        refused = bool(block) or finish in REFUSAL_REASONS
        visible = usage.get("candidatesTokenCount")
        thoughts = int(usage.get("thoughtsTokenCount") or 0)
        output_tokens = None if visible is None and thoughts == 0 else int(visible or 0) + thoughts
        return CallResult(
            raw_reply=text,
            finish_reason=finish,
            input_tokens=_int_or_none(usage.get("promptTokenCount")),
            output_tokens=output_tokens,
            refused=refused,
            refusal_reason=block or (finish if finish in REFUSAL_REASONS else ""),
            temperature=temperature,
            reasoning_effort=effort,
            thinking_setting=effort,
            thinking_budget=0 if self._thinking_label == "thinkingBudget 0" else None,
            visible_output_tokens=_int_or_none(visible),
            system_delivery="system",
        )


class LMStudioProvider(OpenAIProvider):
    name = "lmstudio"

    def __init__(self, model: dict, config: dict) -> None:
        Provider.__init__(self, model, config)
        self.api_key = ""
        base = str(config.get("lmstudio_base_url", "http://localhost:1234/v1")).rstrip("/")
        self.url = f"{base}/chat/completions"

    def generate(self, system: str, user: str, *, reserve_multiplier: int = 1) -> CallResult:
        delivery = str(self.model.get("system_delivery") or "system")
        body: dict = {
            "model": self.model_id,
            "messages": listener_messages(system, user, system_delivery=delivery),
            "max_tokens": self.max_output_tokens,
            "temperature": 0,
        }
        if self.model.get("disable_thinking"):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        payload = self._send_local(body)
        if payload.get("_dropped_template_kwargs"):
            body.pop("chat_template_kwargs", None)
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        visible, had_thought = strip_thinking(content)
        finish = str(choice.get("finish_reason") or "")
        usage = payload.get("usage") or {}
        temperature = None if payload.get("_dropped_temperature") else 0
        return CallResult(
            raw_reply=visible,
            finish_reason=finish,
            input_tokens=_int_or_none(usage.get("prompt_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            refused=finish in REFUSAL_REASONS,
            refusal_reason=finish if finish in REFUSAL_REASONS else "",
            temperature=temperature,
            reasoning_effort="enable_thinking false" if self.model.get("disable_thinking") else None,
            thinking_setting="enable_thinking false" if self.model.get("disable_thinking") else None,
            thinking_budget=0 if self.model.get("disable_thinking") else None,
            visible_output_tokens=_int_or_none(usage.get("completion_tokens")),
            system_delivery=delivery,
            thinking_output=had_thought,
        )

    def _send_local(self, body: dict) -> dict:
        try:
            return self._send(body)
        except APIError as exc:
            if exc.status == 400 and "chat_template_kwargs" in body:
                dropped = dict(body)
                dropped.pop("chat_template_kwargs", None)
                payload = self._send(dropped)
                payload["_dropped_template_kwargs"] = True
                return payload
            raise


def _env(name: str) -> str:
    import os

    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"Set {name} in the environment. Do not put the key in a file.")
    return value


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    return int(value)


def build_provider(model: dict, config: dict) -> Provider:
    kind = model["provider"]
    if kind == "openai":
        return OpenAIProvider(model, config)
    if kind == "anthropic":
        return AnthropicProvider(model, config)
    if kind == "gemini":
        return GeminiProvider(model, config)
    if kind == "lmstudio":
        return LMStudioProvider(model, config)
    raise SystemExit(f"unknown provider {kind}")


def _done(path: Path) -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        found.add((record["item_id"], record["model"], record["prompt_condition"]))
    return found


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _price(model: dict) -> tuple[float, float]:
    return float(model.get("input_per_million", 0)), float(model.get("output_per_million", 0))


def record_call(
    item: dict,
    provider: Provider,
    condition: str,
    *,
    reserve_multiplier: int = 1,
) -> tuple[dict, dict | None]:
    age = format_age(item.get("child_age_months"))
    result = provider.generate(
        system_prompt(condition, age),
        item["context"],
        reserve_multiplier=reserve_multiplier,
    )
    reply, multi, meta = parse_reply(result.raw_reply)
    reply, clipped = clip_tokens(reply, provider.max_output_tokens)
    # A reasoning model can spend the whole token cap before it writes. One
    # larger request gives the reply room. The stored reply is still capped.
    if (
        provider.reasoning
        and not result.refused
        and reply == ""
        and result.finish_reason in LENGTH_REASONS
        and reserve_multiplier == 1
    ):
        return record_call(item, provider, condition, reserve_multiplier=4)
    truncated = clipped or result.finish_reason in LENGTH_REASONS
    record = {
        "item_id": item["item_id"],
        "model": provider.model_id,
        "provider": provider.name,
        "prompt_condition": condition,
        "raw_reply": result.raw_reply,
        "reply": reply,
        "truncated": truncated,
        "finish_reason": result.finish_reason,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temperature": result.temperature,
        "reasoning_effort": result.reasoning_effort,
        "thinking_setting": result.thinking_setting,
        "thinking_budget": result.thinking_budget,
        "visible_output_tokens": result.visible_output_tokens,
        "system_delivery": result.system_delivery,
        "thinking_output": result.thinking_output,
        "multi_turn": multi,
        "meta": meta,
    }
    refused = None
    if result.refused:
        refused = {
            "item_id": item["item_id"],
            "model": provider.model_id,
            "provider": provider.name,
            "prompt_condition": condition,
            "refusal_reason": result.refusal_reason,
            "finish_reason": result.finish_reason,
            "raw_reply": result.raw_reply,
            "timestamp": record["timestamp"],
        }
    return record, refused


def pilot_ids(config: dict, items: list[dict]) -> list[str]:
    path = PROJECT_ROOT / str(config.get("pilot_items_path", "data/processed/pilot_items.txt"))
    if path.exists():
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        known = {item["item_id"]: item for item in items}
        bad = [item_id for item_id in ids if item_id not in known or known[item_id].get("exclude")]
        if bad:
            raise SystemExit("pilot file contains unknown or excluded items: " + ", ".join(bad))
        return ids
    n = int(config.get("pilot_n", 50))
    seed = int(config.get("seed", 42))
    chosen = stratify_pilot(items, n=n, seed=seed)
    ids = sorted(item["item_id"] for item in chosen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return ids


def _thinking_note(records: list[dict]) -> str:
    settings = sorted({str(row.get("thinking_setting")) for row in records if row.get("thinking_setting")})
    budgets = sorted({row.get("thinking_budget") for row in records if row.get("thinking_budget") is not None})
    visible = [row.get("visible_output_tokens") for row in records if row.get("visible_output_tokens") is not None]
    thought = sum(1 for row in records if row.get("thinking_output"))
    parts = []
    if settings:
        parts.append("Thinking setting: " + ", ".join(settings) + ".")
    if budgets:
        parts.append("Thinking budget: " + ", ".join(str(item) for item in budgets) + ".")
    if visible:
        parts.append(f"Visible output tokens: {sum(int(item) for item in visible)} across {len(visible)} calls.")
    if thought:
        parts.append(f"Think blocks that contained text were removed from {thought} calls.")
    return " ".join(parts) if parts else "No separate thinking setting was recorded."


def _local_size_note(model: dict) -> str:
    if model.get("provider") != "lmstudio":
        return ""
    return "All three local models are 3B to 4B parameters because of an 8 GB memory limit."


def write_pilot_report(
    path: Path,
    *,
    records: list[dict],
    items_by_id: dict[str, dict],
    model: dict,
    usable: int,
    pilot_n: int,
) -> None:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_condition[record["prompt_condition"]].append(record)
    temperatures = {record.get("temperature") for record in records}
    efforts = {record.get("reasoning_effort") for record in records}
    effort_text = ", ".join(str(value) for value in sorted((item for item in efforts if item is not None), key=str))
    if temperatures == {0}:
        temperature_line = "Temperature 0 was accepted on every call."
    elif temperatures == {None}:
        temperature_line = "Temperature 0 was rejected. The API default was used on every call."
    else:
        temperature_line = (
            "Temperature 0 was rejected on some calls. Those calls used the API default. "
            f"Values recorded: {sorted(temperatures, key=lambda value: (value is None, value))}."
        )
    input_price, output_price = _price(model)
    input_tokens = sum(int(record.get("input_tokens") or 0) for record in records)
    output_tokens = sum(int(record.get("output_tokens") or 0) for record in records)
    actual = input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price
    full, subsample = projected_cost(actual, pilot_n, usable)
    lines = [
        f"# {model['id']} pilot",
        "",
        f"Model `{model['id']}` ({model['provider']}).",
        f"Reasoning setting: {effort_text or 'none'}.",
        temperature_line,
        f"Max output tokens: 60. Usable items: {usable}. Pilot items: {pilot_n}. Calls: {len(records)}.",
        _thinking_note(records),
        "",
        "Prices are US dollars per million tokens, from config "
        f"(`input_per_million` {input_price}, `output_per_million` {output_price}). "
        "Output tokens are the counts the API billed, which include reasoning tokens when the API counts them there. "
        "visible_output_tokens is the reply, without thinking tokens, when the API separates them.",
        "",
        _local_size_note(model),
        str(model.get("report_note") or ""),
        "",
        "## By condition",
        "",
        "| Condition | Calls | Empty | Refused | Truncated | Multi-turn | Meta | Input tokens | Output tokens | Visible output tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        rows = by_condition.get(condition, [])
        lines.append(
            "| {condition} | {n} | {empty} | {refused} | {truncated} | {multi} | {meta} | {inp} | {out} | {visible} |".format(
                condition=condition,
                n=len(rows),
                empty=sum(1 for row in rows if not str(row.get("reply") or "").strip()),
                refused=sum(1 for row in rows if row.get("finish_reason") in REFUSAL_REASONS or row.get("refused")),
                truncated=sum(1 for row in rows if row.get("truncated")),
                multi=sum(1 for row in rows if row.get("multi_turn")),
                meta=sum(1 for row in rows if row.get("meta")),
                inp=sum(int(row.get("input_tokens") or 0) for row in rows),
                out=sum(int(row.get("output_tokens") or 0) for row in rows),
                visible=sum(int(row.get("visible_output_tokens") or 0) for row in rows),
            )
        )
    lines.extend(
        [
            "",
            "## Cost",
            "",
            f"Actual pilot cost: ${actual:.4f} ({input_tokens} input tokens, {output_tokens} output tokens).",
            f"Projected full run, {usable} usable items x 3 conditions: ${full:.2f}.",
            f"Projected subsample 0.33, {usable} x 0.33 x 3 conditions: ${subsample:.2f}.",
            "",
            "## Examples",
            "",
            "Five items, the same ones under each condition, beside the real adult reply. "
            "These lines are transcript text and model output and must not be redistributed.",
            "",
        ]
    )
    example_ids: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record["item_id"] not in seen:
            seen.add(record["item_id"])
            example_ids.append(record["item_id"])
        if len(example_ids) == 5:
            break
    for condition in CONDITIONS:
        lines.append(f"### {condition}")
        lines.append("")
        rows = {row["item_id"]: row for row in by_condition.get(condition, [])}
        for item_id in example_ids:
            item = items_by_id.get(item_id, {})
            row = rows.get(item_id)
            lines.append(f"`{item_id}` ({item.get('trouble_class')}, {item.get('corpus')})")
            lines.append("")
            lines.append("Real adult reply:")
            lines.append("")
            lines.append("```")
            lines.append(str(item.get("human_next_turn") or ""))
            lines.append("```")
            lines.append("")
            lines.append("Model reply:")
            lines.append("")
            lines.append("```")
            lines.append("" if row is None else str(row.get("reply") or ""))
            lines.append("```")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: dict, *, pilot: bool, provider_name: str | None) -> int:
    items_path = PROJECT_ROOT / str(config.get("items_path", "data/processed/items.jsonl"))
    items = load_items(items_path)
    usable = usable_items(items)
    by_id = {item["item_id"]: item for item in items}
    models = [model for model in config["models"] if provider_name in {None, model["provider"]}]
    if not models:
        raise SystemExit(f"no models for provider {provider_name}")
    if pilot:
        ids = pilot_ids(config, items)
    else:
        ids = [item["item_id"] for item in usable]
    generations = PROJECT_ROOT / str(config.get("generations_path", "outputs/generations.jsonl"))
    refused_path = PROJECT_ROOT / str(config.get("refused_path", "outputs/refused.jsonl"))
    done = _done(generations)
    for model in models:
        provider = build_provider(model, config)
        for item_id in ids:
            item = by_id[item_id]
            if item.get("exclude"):
                continue
            for condition in CONDITIONS:
                key = (item_id, provider.model_id, condition)
                if key in done:
                    continue
                record, refused = record_call(item, provider, condition)
                _append(generations, record)
                if refused:
                    _append(refused_path, refused)
                done.add(key)
                print(f"{provider.model_id} {condition} {item_id}", flush=True)
    if pilot:
        incomplete = False
        stored = _load_generation_rows(generations)
        for model in models:
            written = [
                row for row in stored
                if row.get("model") == model["id"] and row.get("item_id") in set(ids)
            ]
            expected = len(ids) * len(CONDITIONS)
            if len(written) < expected:
                print(
                    f"{model['id']}: {len(written)} of {expected} pilot calls are stored. Report not written.",
                    flush=True,
                )
                incomplete = True
                continue
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model["id"]))
            report = PROJECT_ROOT / "reports" / f"pilot_{slug}.md"
            write_pilot_report(
                report,
                records=written,
                items_by_id=by_id,
                model=model,
                usable=len(usable),
                pilot_n=len(ids),
            )
            print(f"wrote {report}", flush=True)
        if incomplete:
            return 1
    return 0


def _load_generation_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate adult replies for listener items.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pilot", action="store_true", help="50 stratified items, three conditions")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini", "lmstudio"])
    parser.add_argument("--model-id", help="Run one model id from config.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.model_id:
        config["models"] = [model for model in config["models"] if model["id"] == args.model_id]
        if not config["models"]:
            raise SystemExit(f"no model {args.model_id} in {args.config}")
    return run(config, pilot=args.pilot, provider_name=args.provider)


if __name__ == "__main__":
    raise SystemExit(main())
