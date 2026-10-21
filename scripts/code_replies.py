#!/usr/bin/env python3
"""Code pilot replies with the judge named in config.yaml.

The judge sees the context and one reply, never who wrote it. Reply order is
shuffled with seed 42. Thinking mode is off. A reply is kept only when the
judge returns one JSON object with a valid category.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate import APIError, _env, _post_json, _with_backoff, load_config
from scripts.items import load_items

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42

CATEGORIES = (
    "open_request",
    "restricted_request",
    "candidate_check",
    "reformulation",
    "echo_error",
    "acknowledge",
    "topic_question",
    "guess_proceed",
    "ignore_shift",
)
CLARIFICATION = {"open_request", "restricted_request", "candidate_check"}
LISTENERS = ("human", "plain", "child", "repair")
GPT_REPLY_LISTENERS = ("plain", "child", "repair")
LISTENER_LABEL = {
    "human": "human",
    "plain": "GPT plain",
    "child": "GPT child",
    "repair": "GPT repair",
}
TROUBLE_CLASSES = ("unintelligible", "error", "none")

SYSTEM_PROMPT = (
    "You code how an adult responded to a child's turn. You see the conversation "
    "and one reply. You do not know who wrote the reply. Output one JSON object "
    "and nothing else."
)
NOTICE_SYSTEM = (
    "You decide whether a typical adult would notice a grammatical or word error "
    "in one child turn. Output one JSON object and nothing else."
)
TRANSCRIPTION_CODE = re.compile(r"\((?:unintelligible|untranscribed)\)", re.IGNORECASE)

CLARIFICATION_RULE = (
    "A clarification (open_request, restricted_request, candidate_check) is a question "
    "about what the child just said or meant, which the child can answer by repeating, "
    "completing or confirming their own words. A question asking for NEW information "
    "about the topic (what colour, is she okay, what do you pretend) is a topic_question, "
    "however closely related."
)

# Real pilot replies used to check the labeller before the pilot is coded.
GOLD = (
    {"item_id": "EllisWeismer/LT/30ec/22170.cha:283", "listener": "plain", "expected": "open_request"},
    {"item_id": "EllisWeismer/TD/30ec/12061.cha:105", "listener": "repair", "expected": "open_request"},
    {"item_id": "Feldman/ParentChild/TD/nchi1539.cha:252", "listener": "repair", "expected": "open_request"},
    {"item_id": "Rescorla/LT/36/bry36.cha:185", "listener": "repair", "expected": "open_request"},
    {"item_id": "EllisWeismer/LT/66conv/11043.cha:289", "listener": "human", "expected": "acknowledge"},
    {"item_id": "EllisWeismer/LT/66conv/12083.cha:1", "listener": "human", "expected": "acknowledge"},
    {"item_id": "EllisWeismer/TD/30ec/11100.cha:55", "listener": "human", "expected": "acknowledge"},
    {"item_id": "EllisWeismer/TD/42ec/22118.cha:174", "listener": "human", "expected": "acknowledge"},
    {"item_id": "Feldman/ParentChild/SLI/PC-g/gla21.cha:214", "listener": "human", "expected": "acknowledge"},
    {"item_id": "Feldman/ParentChild/SLI/PC-mr/pop27.cha:622", "listener": "human", "expected": "acknowledge"},
    {"item_id": "Rescorla/TD/48/aik48.cha:221", "listener": "human", "expected": "acknowledge"},
    {"item_id": "Feldman/ParentChild/SLI/PC-ac/cal36.cha:269", "listener": "repair", "expected": "candidate_check"},
    {"item_id": "EllisWeismer/LT/42pc/22166.cha:170", "listener": "human", "expected": "reformulation"},
    {"item_id": "EllisWeismer/LT/30ec/11065.cha:63", "listener": "plain", "expected": "topic_question"},
    {"item_id": "EllisWeismer/LT/66conv/11043.cha:289", "listener": "plain", "expected": "topic_question"},
    {"item_id": "EllisWeismer/TD/30ec/11057.cha:231", "listener": "plain", "expected": "topic_question"},
    {"item_id": "EllisWeismer/TD/30ec/11100.cha:55", "listener": "plain", "expected": "guess_proceed"},
    {"item_id": "Conti/Conti3/Harry/031016.cha:684", "listener": "plain", "expected": "restricted_request"},
    {"item_id": "EllisWeismer/TD/30ec/11100.cha:55", "listener": "repair", "expected": "candidate_check"},
)


def n_words(text: str) -> int:
    return len([part for part in (text or "").split() if part])


def _words(text: str) -> list[str]:
    """Case-folded words with punctuation removed. Apostrophes stay."""
    cleaned = re.sub(r"[^\w'\s]", " ", (text or "").lower(), flags=re.UNICODE)
    return [word for word in cleaned.split() if word]


def contains_target(reply: str, targets: list[str] | None) -> bool | None:
    """True when any gold target is in the reply. None when there is no target."""
    phrases = [_words(target) for target in (targets or [])]
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return None
    words = _words(reply)
    for phrase in phrases:
        width = len(phrase)
        if any(words[index : index + width] == phrase for index in range(len(words) - width + 1)):
            return True
    return False


def last_child_line(context: str) -> str:
    """The last Child: line, which is the turn the reply responds to."""
    found = ""
    for line in (context or "").splitlines():
        if line.startswith("Child:"):
            found = line
    return found


def spoken_child_turn(context: str) -> str:
    """The child's last turn with transcription codes removed."""
    line = last_child_line(context)
    text = line.split(":", 1)[1] if ":" in line else line
    text = TRANSCRIPTION_CODE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def present_context(context: str) -> str:
    """Mark the child's last turn inside the conversation."""
    lines = (context or "").splitlines()
    indexes = [index for index, line in enumerate(lines) if line.startswith("Child:")]
    if not indexes:
        return context or ""
    last = indexes[-1]
    marked = list(lines)
    marked[last] = "CHILD'S TURN BEING RESPONDED TO: " + lines[last]
    return "\n".join(marked)


def _thinking_off(text: str, *, qwen: bool) -> str:
    if qwen:
        return "/no_think\n" + text
    return text


def judge_user(context: str, reply: str, *, qwen: bool) -> str:
    child = last_child_line(context) or "(no child turn)"
    body = (
        CLARIFICATION_RULE + "\n\n"
        "Code the reply at the end. It is the next turn after the child's turn. "
        "Judge that reply, not the child's words.\n\n"
        "Conversation:\n"
        f"{present_context(context)}\n\n"
        "The child's turn being responded to:\n"
        f"{child}\n\n"
        "Categories. Pick one. Each has a definition and examples.\n"
        "open_request: the reply asks the child to say the turn again, without naming a particular word. "
        'Examples: "what?" / "I didn\'t quite catch that, can you say it again?"\n'
        "restricted_request: the reply asks about one part of the turn, including what a word the child used means. "
        'Examples: "you saw what?" / "Who\'s Huhica?"\n'
        "candidate_check: the reply offers a reading of the child's turn and asks if that reading is right. "
        'Examples: "the dog?" / "Did you say you want to put the baby in the bed?"\n'
        "reformulation: the reply repeats the child's words in corrected adult form and does not ask. "
        'Examples: "it\'s just for the stable" / "the baby is eating"\n'
        "echo_error: the reply repeats or adopts the child's incorrect form. "
        'Examples: "doggy runned" / "two mans"\n'
        "acknowledge: a short adult acceptance, backchannel, or generic praise, with no new content. "
        'Examples: "mhm" / "oh". Also "hm", "okay", "uhhuh", and "good job".\n'
        "topic_question: a question that asks for new information about the topic, even when it mentions what the child was talking about. "
        'Examples: "Is baby making a funny face in the mirror?" / "Wow, that\'s a big fish! What color is it?" / "Oh no! Is she okay? Should we help her up?"\n'
        "guess_proceed: the reply commits to a reading and moves on. A tag question on the adult's own reading is still guess_proceed. "
        'Examples: "yes the dog is running, let us go" / "That way the baby will be warm and safe, right?"\n'
        "ignore_shift: the reply leaves the child's words and is not a question. "
        'A new activity stated rather than asked is ignore_shift. '
        'Examples: "let\'s play with the cars" / "time for a snack"\n\n'
        "wrong_role is true only when the reply is clearly spoken as the child. "
        'Examples: it addresses the adult as "Mum"; "I want to call my friend" said back to an adult\'s play. '
        "Short adult reactions are never wrong_role. "
        "A reformulation, a question to the child, and a backchannel are the adult.\n\n"
        "Reply to code:\n"
        f"{reply}\n\n"
        'Output one JSON object and nothing else, with keys category and wrong_role. '
        'wrong_role is a JSON boolean.'
    )
    return _thinking_off(body, qwen=qwen)


def notice_user(context: str, *, qwen: bool) -> str:
    turn = spoken_child_turn(context) or "(no words left after removing transcription codes)"
    body = (
        "Child's turn, with transcription codes removed:\n"
        f"{turn}\n\n"
        "Would a typical adult notice a grammatical or word error in that turn? "
        "Ordinary child wording that an adult would accept is not noticeable.\n"
        'Noticeable examples: "baby eat the supper" / "when him did that".\n'
        'Not noticeable examples: "I still got the same fish" / "I want the ball".\n\n'
        "Output one JSON object and nothing else, with the key noticeable_error "
        "set to true or false."
    )
    return _thinking_off(body, qwen=qwen)


def corrects_user(context: str, reply: str, *, qwen: bool) -> str:
    """Separate from the category: does the reply correct the child's error?"""
    turn = spoken_child_turn(context) or "(no words left after removing transcription codes)"
    body = (
        "Decide only whether the reply contains a corrected version of the erroneous "
        "part of the child's turn. Ignore what kind of reply it is. A correction can "
        "be a statement or a question.\n"
        'Example: the child says "baby eat the supper" and the reply is '
        '"Oh, the baby eats supper?" That is true.\n'
        "A reply that repeats the error, or does not supply the corrected form, is false.\n\n"
        f"Child's turn: {turn}\n"
        f"Reply: {reply}\n\n"
        "Output one JSON object and nothing else, with the key corrects_error "
        "set to true or false."
    )
    return _thinking_off(body, qwen=qwen)


def visible_user(context: str, *, qwen: bool) -> str:
    """Once per baseline item: an unmarked error or confusing wording."""
    turn = spoken_child_turn(context) or "(no words left after removing transcription codes)"
    body = (
        "The transcriber marked this child turn as having no error. "
        "Does the turn still contain a clearly visible error or confusing wording? "
        "These are visible trouble, so the answer is true:\n"
        '"two three nine" is a counting error.\n'
        '"car crash into people" is missing words.\n'
        '"now what\'s wrong dear" in pretend play is ambiguous about who is speaking.\n'
        "Ordinary child wording that an adult would accept is not visible trouble.\n\n"
        "Child's turn, with transcription codes removed:\n"
        f"{turn}\n\n"
        "Output one JSON object and nothing else, with the key visible_trouble "
        "set to true or false."
    )
    return _thinking_off(body, qwen=qwen)


def _json_text(text: str) -> str:
    """One object, with a surrounding markdown fence removed."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    return raw


def parse_reply_judgment(text: str) -> dict | None:
    """One JSON object, and nothing else, with a known category."""
    raw = _json_text(text)
    if not raw or raw[0] != "{" or raw[-1] != "}":
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if set(value) != {"category", "wrong_role"}:
        return None
    if value["category"] not in CATEGORIES or not isinstance(value["wrong_role"], bool):
        return None
    return value


def parse_flag(text: str, key: str) -> dict | None:
    """One JSON object whose only key is a boolean."""
    raw = _json_text(text)
    if not raw or raw[0] != "{" or raw[-1] != "}":
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {key} or not isinstance(value[key], bool):
        return None
    return value


def parse_notice_judgment(text: str) -> dict | None:
    raw = _json_text(text)
    if not raw or raw[0] != "{" or raw[-1] != "}":
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"noticeable_error"}:
        return None
    if not isinstance(value["noticeable_error"], bool):
        return None
    return value


def validation_sample(records: list[dict], n: int = 60, seed: int = SEED) -> list[dict]:
    """Up to n coded replies, spread across category and listener."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("failed") or record.get("task") != "reply":
            continue
        cells[(record["category"], record["listener"])].append(record)
    if not cells:
        return []
    keys = sorted(cells)
    total = sum(len(pool) for pool in cells.values())
    draw = min(n, total)
    raw = {key: draw * len(cells[key]) / total for key in keys}
    quota = {key: min(len(cells[key]), math.floor(raw[key])) for key in keys}
    if len(keys) <= draw:
        for key in keys:
            quota[key] = max(quota[key], 1)
        while sum(quota.values()) > draw:
            donor = max(keys, key=lambda key: (quota[key], len(cells[key]), key))
            if quota[donor] <= 1:
                break
            quota[donor] -= 1
    left = draw - sum(quota.values())
    order = sorted(keys, key=lambda key: (-(raw[key] - math.floor(raw[key])), key))
    for key in order:
        if left <= 0:
            break
        if quota[key] >= len(cells[key]):
            continue
        quota[key] += 1
        left -= 1
    rng = random.Random(seed)
    chosen: list[dict] = []
    for key in keys:
        pool = list(cells[key])
        rng.shuffle(pool)
        chosen.extend(pool[: quota[key]])
    rng.shuffle(chosen)
    return chosen


def _post(url: str, body: dict, timeout: int = 180) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(f"connection failed: {exc}") from None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from None


def _message_text(payload: dict) -> str:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _add_usage(config: dict, payload: dict) -> None:
    usage = payload.get("usage") or {}
    totals = config.setdefault("_judge_usage", {"input": 0, "output": 0, "calls": 0})
    totals["input"] += int(usage.get("prompt_tokens") or 0)
    totals["output"] += int(usage.get("completion_tokens") or 0)
    totals["calls"] += 1


def _call_openai_judge(config: dict, system: str, user: str, model: str) -> str:
    """GPT labeller. This model has no thinking mode, so none is requested."""
    api_key = _env("OPENAI_API_KEY")
    body: dict = {
        "model": model,
        "temperature": float(config.get("judge_temperature", 0)),
        "max_completion_tokens": int(config.get("judge_max_tokens", 80)),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    url = "https://api.openai.com/v1/chat/completions"

    def once(current: dict) -> dict:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        return _with_backoff(
            lambda: _post_json(url, current, headers, [api_key], timeout=120),
            int(config.get("max_attempts", 5)),
            float(config.get("retry_initial_seconds", 2)),
            float(config.get("retry_multiplier", 2)),
        )

    try:
        payload = once(body)
    except APIError as exc:
        message = str(exc).lower()
        if exc.status == 400 and "temperature" in message and "temperature" in body:
            dropped = dict(body)
            dropped.pop("temperature", None)
            payload = once(dropped)
        else:
            raise RuntimeError(str(exc)) from None
    _add_usage(config, payload)
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if message.get("refusal"):
        return ""
    return _message_text(payload).strip()


def _anthropic_text(payload: dict) -> str:
    parts = payload.get("content") or []
    return "".join(
        part.get("text", "") for part in parts if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def _call_anthropic_judge(config: dict, system: str, user: str, model: str) -> str:
    """Claude labeller. Up-front thinking is turned off, or down to the lowest setting the model allows."""
    api_key = _env("ANTHROPIC_API_KEY")
    body: dict = {
        "model": model,
        "max_tokens": int(config.get("judge_max_tokens", 80)),
        "temperature": float(config.get("judge_temperature", 0)),
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    url = "https://api.anthropic.com/v1/messages"

    def once(current: dict) -> dict:
        return _with_backoff(
            lambda: _post_json(
                url,
                current,
                {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                [api_key],
                timeout=120,
            ),
            int(config.get("max_attempts", 5)),
            float(config.get("retry_initial_seconds", 2)),
            float(config.get("retry_multiplier", 2)),
        )

    payload = None
    for _try in range(4):
        try:
            payload = once(body)
            break
        except APIError as exc:
            message = str(exc).lower()
            if exc.status != 400:
                raise RuntimeError(str(exc)) from None
            thinking = body.get("thinking") or {}
            if "between_tools" in message and thinking.get("type") != "between_tools":
                body["thinking"] = {"type": "between_tools"}
                continue
            if "temperature" in message and "temperature" in body:
                body.pop("temperature", None)
                continue
            if "effort" in message and "output_config" in body:
                body.pop("output_config", None)
                continue
            if "disabled" in message and thinking.get("type") == "disabled":
                body.pop("thinking", None)
                continue
            raise RuntimeError(str(exc)) from None
    if payload is None:
        raise RuntimeError("Claude judge request failed")
    usage = payload.get("usage") or {}
    _add_usage(
        config,
        {
            "usage": {
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
            }
        },
    )
    return _anthropic_text(payload)


def _call_local_judge(config: dict, system: str, user: str, model: str) -> str:
    base = str(config.get("lmstudio_base_url", "http://localhost:1234/v1")).rstrip("/")
    body = {
        "model": model,
        "temperature": float(config.get("judge_temperature", 0)),
        "max_tokens": int(config.get("judge_max_tokens", 80)),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error: Exception | None = None
    for _try in range(3):
        try:
            payload = _post(f"{base}/chat/completions", body)
            _add_usage(config, payload)
            return _message_text(payload).strip()
        except RuntimeError as exc:
            last_error = exc
            message = str(exc)
            if "HTTP 400" in message and "chat_template_kwargs" in body:
                body.pop("chat_template_kwargs", None)
                continue
            if message.startswith("connection failed") or message.startswith("HTTP 5"):
                time.sleep(2)
                continue
            raise
    raise last_error if last_error else RuntimeError("judge request failed")


def judge_slot_for_speaker(model_id: str) -> str:
    """Claude replies are coded by GPT-4.1. Every other family is coded by Claude."""
    if "claude" in model_id.lower():
        return "other_replies"
    return "gpt_replies"


def speaker_model_of(row: dict) -> str:
    """Listener model on a coding row. Older GPT rows did not store one."""
    if row.get("speaker_model"):
        return str(row["speaker_model"])
    if row.get("listener") == "human":
        return "human"
    return "gpt-4.1-2025-04-14"


def judge_spec(config: dict, task: dict) -> dict:
    """GPT replies go to Claude. Human replies are sent to the slot named on the task."""
    judges = config.get("judges") or {}
    slot = task.get("judge_slot")
    if slot not in {"gpt_replies", "other_replies"}:
        if task.get("task") in {"reply", "correct"} and task.get("listener") in GPT_REPLY_LISTENERS:
            slot = "gpt_replies"
        else:
            slot = "other_replies"
    spec = judges.get(slot) or {}
    if not spec.get("model") or not spec.get("provider"):
        raise SystemExit(f"Set judges.{slot}.model before coding.")
    return spec


def claude_judge(config: dict) -> dict:
    spec = (config.get("judges") or {}).get("gpt_replies") or {}
    if not spec.get("model") or spec.get("provider") != "anthropic":
        raise SystemExit(
            "Set judges.gpt_replies.model to a Claude model id. The gold set has not been run."
        )
    return spec


def call_judge(config: dict, system: str, user: str, spec: dict | None = None) -> str:
    chosen = spec or {
        "provider": config.get("judge_provider") or "lmstudio",
        "model": config.get("judge_model"),
    }
    provider = str(chosen.get("provider") or "")
    model = str(chosen.get("model") or "")
    if provider == "openai":
        return _call_openai_judge(config, system, user, model)
    if provider == "anthropic":
        return _call_anthropic_judge(config, system, user, model)
    return _call_local_judge(config, system, user, model)


def _load_jsonl(path: Path) -> list[dict]:
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


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def build_tasks(items: list[dict], generations: list[dict], model_id: str | list[str]) -> list[dict]:
    model_ids = [model_id] if isinstance(model_id, str) else list(model_id)
    replies: dict[tuple[str, str, str], str] = {}
    for record in generations:
        speaker = str(record.get("model") or "")
        if speaker not in model_ids:
            continue
        replies[(record["item_id"], record["prompt_condition"], speaker)] = record.get("reply") or ""
    tasks = []
    for item in items:
        item_id = item["item_id"]
        for slot in ("gpt_replies", "other_replies"):
            tasks.append(
                {
                    "task": "reply",
                    "item_id": item_id,
                    "listener": "human",
                    "speaker_model": "human",
                    "judge_slot": slot,
                    "context": item["context"],
                    "reply": item.get("human_next_turn") or "",
                    "trouble_class": item["trouble_class"],
                    "gold_target": item.get("gold_target"),
                }
            )
        for speaker in model_ids:
            for listener in GPT_REPLY_LISTENERS:
                key = (item_id, listener, speaker)
                if key not in replies:
                    continue
                tasks.append(
                    {
                        "task": "reply",
                        "item_id": item_id,
                        "listener": listener,
                        "speaker_model": speaker,
                        "judge_slot": judge_slot_for_speaker(speaker),
                        "context": item["context"],
                        "reply": replies[key],
                        "trouble_class": item["trouble_class"],
                        "gold_target": item.get("gold_target"),
                    }
                )
        if item["trouble_class"] == "error":
            tasks.append(
                {
                    "task": "notice",
                    "item_id": item_id,
                    "context": item["context"],
                    "trouble_class": item["trouble_class"],
                }
            )
        if item["trouble_class"] == "none":
            tasks.append(
                {
                    "task": "visible",
                    "item_id": item_id,
                    "context": item["context"],
                    "trouble_class": item["trouble_class"],
                    "prompt_version": "v2",
                }
            )
    return tasks


def _task_key(config: dict, task: dict) -> tuple:
    if task["task"] in {"reply", "correct"}:
        model = task.get("judge_model") or judge_spec(config, task)["model"]
        return (task["task"], task["item_id"], task.get("listener"), model, speaker_model_of(task))
    if task["task"] == "visible":
        return ("visible", task["item_id"], task.get("prompt_version") or "v1")
    return (task["task"], task["item_id"])


def _ask(config: dict, spec: dict, system: str, user: str, parse) -> tuple[str, dict | None]:
    attempts = int(config.get("judge_max_attempts", 3))
    prompt = user
    raw = ""
    parsed = None
    for _try in range(attempts):
        raw = call_judge(config, system, prompt, spec)
        parsed = parse(raw)
        if parsed is not None:
            break
        prompt = user + "\n\nThat was not the required JSON object. Output only the JSON object."
        time.sleep(0.5)
    return raw, parsed


def code_tasks(config: dict, tasks: list[dict], path: Path) -> list[dict]:
    done = {_task_key(config, row): row for row in _load_jsonl(path)}
    pending = [task for task in tasks if _task_key(config, task) not in done]
    rng = random.Random(SEED)
    rng.shuffle(pending)
    for task in pending:
        spec = judge_spec(config, task)
        qwen = "qwen" in str(spec.get("model") or "").lower()
        if task["task"] == "reply":
            user = judge_user(task["context"], task["reply"], qwen=qwen)
            system = SYSTEM_PROMPT
            parse = parse_reply_judgment
        elif task["task"] == "visible":
            user = visible_user(task["context"], qwen=qwen)
            system = NOTICE_SYSTEM
            parse = lambda text: parse_flag(text, "visible_trouble")
        else:
            user = notice_user(task["context"], qwen=qwen)
            system = NOTICE_SYSTEM
            parse = parse_notice_judgment
        raw, parsed = _ask(config, spec, system, user, parse)
        record = {
            "task": task["task"],
            "item_id": task["item_id"],
            "trouble_class": task["trouble_class"],
            "raw_judge": raw,
            "failed": parsed is None,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "judge_model": spec.get("model"),
            "judge_provider": spec.get("provider"),
        }
        if task["task"] == "reply":
            corrects = None
            corrects_raw = None
            if task["trouble_class"] == "error":
                corrects_raw, corrects_parsed = _ask(
                    config,
                    spec,
                    SYSTEM_PROMPT,
                    corrects_user(task["context"], task["reply"], qwen=qwen),
                    lambda text: parse_flag(text, "corrects_error"),
                )
                corrects = None if corrects_parsed is None else corrects_parsed["corrects_error"]
            record.update(
                {
                    "listener": task["listener"],
                    "speaker_model": speaker_model_of(task),
                    "context": task["context"],
                    "reply": task["reply"],
                    "category": None if parsed is None else parsed["category"],
                    "wrong_role": None if parsed is None else parsed["wrong_role"],
                    "n_words": n_words(task["reply"]),
                    "contains_target": contains_target(task["reply"], task.get("gold_target")),
                    "corrects_error": corrects,
                    "corrects_raw": corrects_raw,
                }
            )
        elif task["task"] == "visible":
            record["context"] = task["context"]
            record["visible_trouble"] = None if parsed is None else parsed["visible_trouble"]
            record["prompt_version"] = task.get("prompt_version") or "v1"
        else:
            record["noticeable_error"] = None if parsed is None else parsed["noticeable_error"]
        _append(path, record)
        done[_task_key(config, record)] = record
        label = task["listener"] if task["task"] == "reply" else task["task"]
        print(f"{label} {spec.get('model')} {task['item_id']} {'fail' if parsed is None else 'ok'}", flush=True)
    return list(done.values())


def _rate(count: int, denominator: int) -> str:
    if denominator == 0:
        return "—"
    return f"{count / denominator:.1%} ({count}/{denominator})"


def clarification_counts(records: list[dict]) -> dict[str, dict[str, tuple[int, int]]]:
    replies = [row for row in records if row.get("task") == "reply" and not row.get("failed")]
    counts: dict[str, dict[str, tuple[int, int]]] = {}
    for listener in LISTENERS:
        counts[listener] = {}
        for trouble in TROUBLE_CLASSES:
            rows = [
                row
                for row in replies
                if row.get("listener") == listener and row.get("trouble_class") == trouble
            ]
            hit = sum(1 for row in rows if row.get("category") in CLARIFICATION)
            counts[listener][trouble] = (hit, len(rows))
    return counts


def paired_binary(
    left: dict[str, dict],
    right: dict[str, dict],
    field: str,
    positive,
) -> list[tuple[str, str]]:
    """Yes/no pairs for one field. A missing value drops that item."""
    pairs: list[tuple[str, str]] = []
    for item_id in sorted(set(left) & set(right)):
        labels = []
        for row in (left[item_id], right[item_id]):
            value = row.get(field)
            if value is None:
                labels = []
                break
            if isinstance(positive, (set, frozenset)):
                labels.append("yes" if value in positive else "no")
            else:
                labels.append("yes" if value == positive else "no")
        if labels:
            pairs.append((labels[0], labels[1]))
    return pairs


def _kappa_line(what: str, pairs: list[tuple[str, str]]) -> str:
    kappa, agree, count = cohens_kappa(pairs)
    text = "—" if kappa is None else f"{kappa:.3f}"
    return f"Cohen's kappa on {what}: {text} ({agree} agreements in {count} replies)."


def cohens_kappa(pairs: list[tuple[str, str]]) -> tuple[float | None, int, int]:
    """Unweighted Cohen's kappa. Returns kappa, agreements, and the number of pairs."""
    count = len(pairs)
    if count == 0:
        return None, 0, 0
    agree = sum(left == right for left, right in pairs)
    labels = {left for left, _ in pairs} | {right for _, right in pairs}
    chance = 0.0
    for label in labels:
        left_share = sum(left == label for left, _ in pairs) / count
        right_share = sum(right == label for _, right in pairs) / count
        chance += left_share * right_share
    if chance == 1:
        return 1.0, agree, count
    return (agree / count - chance) / (1 - chance), agree, count


def _points(new: tuple[int, int], old: tuple[int, int]) -> str:
    if new[1] == 0 or old[1] == 0:
        return "—"
    delta = (new[0] / new[1] - old[0] / old[1]) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f} pp"


def _listener_rows(
    replies: list[dict],
    listener: str,
    judge: str,
    speaker: str | None = None,
) -> list[dict]:
    return [
        row
        for row in replies
        if row.get("listener") == listener
        and row.get("judge_model") == judge
        and not row.get("failed")
        and (speaker is None or speaker_model_of(row) == speaker)
    ]


def _metric_cells(rows: list[dict]) -> dict[str, str]:
    grouped: dict[str, list[dict]] = {trouble: [] for trouble in TROUBLE_CLASSES}
    for row in rows:
        if row.get("trouble_class") in grouped:
            grouped[row["trouble_class"]].append(row)
    cells = {}
    for trouble, group in grouped.items():
        n = len(group)
        def count(name: str, group: list[dict] = group) -> int:
            return sum(1 for row in group if row.get("category") == name)
        targeted = [row for row in group if row.get("contains_target") is not None]
        hits = sum(1 for row in targeted if row.get("contains_target"))
        corrected = [row for row in group if row.get("corrects_error") is not None]
        correct_hits = sum(1 for row in corrected if row.get("corrects_error"))
        cells[trouble] = {
            "n": str(n),
            "clar": _rate(sum(1 for row in group if row.get("category") in CLARIFICATION), n),
            "reform": _rate(count("reformulation"), n),
            "target": _rate(hits, len(targeted)),
            "corrects": _rate(correct_hits, len(corrected)),
            "echo": _rate(count("echo_error"), n),
            "ack": _rate(count("acknowledge"), n),
            "topic": _rate(count("topic_question"), n),
            "guess": _rate(count("guess_proceed"), n),
            "ignore": _rate(count("ignore_shift"), n),
            "wrong": str(sum(1 for row in group if row.get("wrong_role"))),
            "mean": "—" if not group else f"{sum(row.get('n_words') or 0 for row in group) / n:.1f}",
        }
    return cells


def _append_rate_table(lines: list[str], title: str, rows: list[dict]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        "| Trouble class | N | Clarification | Reformulation | contains_target | corrects_error | "
        "Echo error | Acknowledge | Topic question | Guess and proceed | "
        "Ignore or shift | Wrong role | Mean words |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    cells = _metric_cells(rows)
    for trouble in TROUBLE_CLASSES:
        cell = cells[trouble]
        lines.append(
            "| {trouble} | {n} | {clar} | {reform} | {target} | {corrects} | {echo} | {ack} | {topic} | {guess} | {ignore} | {wrong} | {mean} |".format(
                trouble=trouble, **cell
            )
        )
    lines.append("")


def write_report(
    path: Path,
    records: list[dict],
    *,
    gold_note: str,
    provisional: bool = False,
    previous: list[dict] | None = None,
    claude_model: str = "claude-sonnet-5-5",
    gpt_model: str = "gpt-4.1-2025-04-14",
    recent_trouble: dict[str, bool] | None = None,
    speaker_model: str = "gpt-4.1-2025-04-14",
) -> None:
    replies = [row for row in records if row.get("task") == "reply"]
    notices = {row["item_id"]: row for row in records if row.get("task") == "notice"}
    failures = [row for row in replies if row.get("failed")]
    notice_failures = [row for row in notices.values() if row.get("failed")]
    model = claude_model
    other = gpt_model
    lines = [
        "# Coding the GPT pilot",
        "",
        f"GPT replies (`plain`, `child`, `repair`) are labelled by `{model}`. "
        f"The same human replies are labelled by both `{model}` and `{other}`. "
        "A comparison uses one judge for both sides: GPT replies against humans "
        f"use `{model}`; Claude replies, when they exist, against humans use `{other}`. "
        + (
            f"`{other}` remains a tested listener, so its human labels are provisional. "
            if provisional
            else ""
        )
        + "No model codes replies from its own family. "
        "Up-front thinking is turned off. Temperature 0. "
        "The judge saw the context and one reply, never who wrote it. "
        f"Reply order was shuffled with seed {SEED}.",
        "",
        gold_note,
        "",
        f"Reply judgments: {len(replies)}. Failures: {len(failures)}. "
        f"Error items judged for a noticeable error: {len(notices)}. "
        f"Notice failures: {len(notice_failures)}.",
        "",
        "Rates use successful judgments only. Clarification is open_request + "
        "restricted_request + candidate_check. contains_target is among error items "
        "that have a gold target, matched on words with case and punctuation ignored.",
        "",
    ]
    human_claude = _listener_rows(replies, "human", claude_model, "human")
    human_gpt = _listener_rows(replies, "human", gpt_model, "human")
    claude_cells = _metric_cells(human_claude)
    gpt_cells = _metric_cells(human_gpt)
    lines.extend(
        [
            "## Human rates under each judge",
            "",
            f"Each human reply has two stored labels. `{claude_model}` is the judge used "
            f"beside the GPT replies. `{gpt_model}` is the judge to use beside Claude replies later.",
            "",
            "| Trouble class | Judge | N | Clarification | Reformulation | contains_target | corrects_error | "
            "Echo error | Acknowledge | Topic question |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trouble in TROUBLE_CLASSES:
        for judge_name, cells in ((claude_model, claude_cells), (gpt_model, gpt_cells)):
            cell = cells[trouble]
            lines.append(
                f"| {trouble} | {judge_name} | {cell['n']} | {cell['clar']} | {cell['reform']} | "
                f"{cell['target']} | {cell['corrects']} | {cell['echo']} | {cell['ack']} | {cell['topic']} |"
            )
    lines.append("")
    by_item_claude = {row["item_id"]: row for row in human_claude}
    by_item_gpt = {row["item_id"]: row for row in human_gpt}
    pairs = [
        (by_item_claude[item_id]["category"], by_item_gpt[item_id]["category"])
        for item_id in sorted(set(by_item_claude) & set(by_item_gpt))
        if by_item_claude[item_id].get("category") and by_item_gpt[item_id].get("category")
    ]
    kappa, agree, pair_n = cohens_kappa(pairs)
    kappa_text = "—" if kappa is None else f"{kappa:.3f}"
    splits: dict[tuple[str, str], int] = {}
    for item_id in sorted(set(by_item_claude) & set(by_item_gpt)):
        left = by_item_claude[item_id].get("category")
        right = by_item_gpt[item_id].get("category")
        if left and right and left != right:
            splits[(left, right)] = splits.get((left, right), 0) + 1
    split_text = ", ".join(
        f"{left} vs {right} ({count})"
        for (left, right), count in sorted(splits.items(), key=lambda item: (-item[1], item[0]))
    )
    lines.extend(
        [
            f"Cohen's kappa between `{claude_model}` and `{gpt_model}` on the human-reply "
            f"categories: {kappa_text} ({agree} agreements in {pair_n} replies).",
            "",
            _kappa_line(
                "clarification (yes/no)",
                paired_binary(by_item_claude, by_item_gpt, "category", CLARIFICATION),
            ),
            "",
            _kappa_line(
                "corrects_error (yes/no, error items only)",
                paired_binary(
                    {item_id: row for item_id, row in by_item_claude.items() if row.get("trouble_class") == "error"},
                    {item_id: row for item_id, row in by_item_gpt.items() if row.get("trouble_class") == "error"},
                    "corrects_error",
                    True,
                ),
            ),
            "",
            "Where the categories differ, the first label is Claude and the second is GPT-4.1: "
            + (split_text or "none")
            + ".",
            "",
            f"## GPT replies compared with humans, both labelled by {claude_model}",
            "",
        ]
    )
    _append_rate_table(lines, f"human ({claude_model})", human_claude)
    for listener in GPT_REPLY_LISTENERS:
        _append_rate_table(
            lines,
            f"{LISTENER_LABEL[listener]} ({claude_model})",
            _listener_rows(replies, listener, claude_model, speaker_model),
        )
    lines.extend(
        [
            f"## Human replies labelled by {gpt_model}",
            "",
            "These labels are stored for a later comparison with Claude replies. "
            "This pilot has no Claude replies, so there is no model column beside them yet.",
            "",
        ]
    )
    _append_rate_table(lines, f"human ({gpt_model})", human_gpt)

    if previous:
        before = clarification_counts(previous)
        after = clarification_counts(records)
        lines.extend(
            [
                "## Change in clarification rate",
                "",
                "Compared with the previous GPT-4.1 coding, before the clarification rule was tightened.",
                "",
                "| Listener | Trouble class | Previous | This coding | Change |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for listener in LISTENERS:
            for trouble in TROUBLE_CLASSES:
                old = before[listener][trouble]
                new = after[listener][trouble]
                lines.append(
                    f"| {LISTENER_LABEL[listener]} | {trouble} | {_rate(*old)} | {_rate(*new)} | {_points(new, old)} |"
                )
        lines.append("")

    noticed = [row for row in notices.values() if not row.get("failed")]
    yes = sum(1 for row in noticed if row.get("noticeable_error"))
    lines.extend(
        [
            "## Noticeable errors",
            "",
            "Once per error item, not per reply. "
            f"{yes} of {len(noticed)} successful judgments say a typical adult would notice an error.",
            "",
        ]
    )
    lines.extend(
        [
            "## Baseline items: human clarification beside the repair false-alarm rate",
            "",
            f"Both columns use `{claude_model}`, the judge of the GPT repair replies. "
            "The items are the trouble-free baseline. `recent_trouble` is not a judge: "
            "it is true when any of the previous three target-child turns in the context "
            "has a trouble code in the original CHAT. "
            "These lines are transcript text and must not be redistributed.",
            "",
        ]
    )
    visible_rows = [
        row
        for row in records
        if row.get("task") == "visible" and row.get("prompt_version") == "v2" and not row.get("failed")
    ]
    visible = {row["item_id"]: row.get("visible_trouble") for row in visible_rows}
    human_base = {
        row["item_id"]: row
        for row in human_claude
        if row.get("trouble_class") == "none"
    }
    repair_base = {
        row["item_id"]: row
        for row in _listener_rows(replies, "repair", claude_model, speaker_model)
        if row.get("trouble_class") == "none"
    }
    paired_ids = sorted(set(human_base) & set(repair_base))

    def _clar_rate(ids: list[str], pool: dict[str, dict]) -> str:
        rows = [pool[item_id] for item_id in ids if item_id in pool]
        hits = sum(1 for row in rows if row.get("category") in CLARIFICATION)
        return _rate(hits, len(rows))

    flags = recent_trouble or {}
    quiet_ids = [item_id for item_id in paired_ids if flags.get(item_id) is False]
    recent_true = sum(1 for item_id in paired_ids if flags.get(item_id) is True)
    missing_flags = [item_id for item_id in paired_ids if item_id not in flags]
    clean_ids = [item_id for item_id in paired_ids if visible.get(item_id) is False]
    flagged = [row for row in visible_rows if row.get("visible_trouble")]
    lines.extend(
        [
            "| Items | N | Human clarification | GPT repair clarification |",
            "| --- | ---: | ---: | ---: |",
            f"| All baseline | {len(paired_ids)} | {_clar_rate(paired_ids, human_base)} | {_clar_rate(paired_ids, repair_base)} |",
            f"| recent_trouble false | {len(quiet_ids)} | {_clar_rate(quiet_ids, human_base)} | {_clar_rate(quiet_ids, repair_base)} |",
            "",
            f"Baseline items with recent_trouble true: {recent_true} of {len(paired_ids)}.",
            "",
        ]
    )
    if missing_flags:
        lines.append(f"Baseline items with no recent_trouble flag: {len(missing_flags)}.")
        lines.append("")
    lines.extend(
        [
            f"The earlier visible_trouble check flagged {len(flagged)} of {len(visible_rows)} "
            f"baseline items. Among those with visible_trouble false, human clarification is "
            f"{_clar_rate(clean_ids, human_base)} and GPT repair clarification is "
            f"{_clar_rate(clean_ids, repair_base)}.",
            "",
        ]
    )
    lines.extend(["### Baseline items flagged by visible_trouble", ""])
    if not flagged:
        lines.append("None.")
        lines.append("")
    for row in sorted(flagged, key=lambda item: item["item_id"]):
        lines.append(f"- `{row['item_id']}`: {spoken_child_turn(row.get('context') or '')}")
    lines.append("")
    alarms = [
        row
        for row in replies
        if row.get("listener") == "repair"
        and speaker_model_of(row) == speaker_model
        and row.get("trouble_class") == "none"
        and row.get("category") in CLARIFICATION
        and not row.get("failed")
    ]
    lines.extend(["### Repair clarifications on these baseline items", ""])
    if not alarms:
        lines.append("None.")
        lines.append("")
    for row in alarms:
        lines.append(f"`{row['item_id']}` — {row['category']}")
        lines.append("")
        lines.append("Context:")
        lines.append("")
        lines.append("```")
        lines.append(row.get("context") or "")
        lines.append("```")
        lines.append("")
        lines.append("Reply:")
        lines.append("")
        lines.append("```")
        lines.append(row.get("reply") or "")
        lines.append("```")
        lines.append("")
    if failures:
        lines.append("## Failures")
        lines.append("")
        for row in failures:
            lines.append(f"- `{row['item_id']}` {row.get('listener')}: judge output was not one valid JSON object")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_validation(records: list[dict], sheet: Path, key_path: Path, n: int = 60) -> int:
    chosen = validation_sample(records, n=n, seed=SEED)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["context", "reply", "my_category"])
        writer.writeheader()
        for record in chosen:
            writer.writerow(
                {
                    "context": record.get("context") or "",
                    "reply": record.get("reply") or "",
                    "my_category": "",
                }
            )
    with key_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row", "item_id", "listener", "judge_category", "trouble_class", "wrong_role"],
        )
        writer.writeheader()
        for index, record in enumerate(chosen, start=1):
            writer.writerow(
                {
                    "row": index,
                    "item_id": record["item_id"],
                    "listener": record["listener"],
                    "judge_category": record["category"],
                    "trouble_class": record["trouble_class"],
                    "wrong_role": record.get("wrong_role"),
                }
            )
    return len(chosen)


def _reply_for(item: dict, generations: list[dict], model_id: str, listener: str) -> str:
    if listener == "human":
        return item.get("human_next_turn") or ""
    for record in generations:
        if (
            record.get("item_id") == item["item_id"]
            and record.get("model") == model_id
            and record.get("prompt_condition") == listener
        ):
            return record.get("reply") or ""
    raise SystemExit(f"no {listener} reply for {item['item_id']}")


def run_gold(config: dict, items: dict[str, dict], generations: list[dict], model_id: str, path: Path) -> tuple[int, str]:
    """Code all 19 gold replies with the Claude judge. Return how many are wrong."""
    spec = claude_judge(config)
    qwen = "qwen" in str(spec["model"]).lower()
    lines = [
        "# Judge gold set",
        "",
        f"Judge `{spec['model']}` ({spec.get('provider')}). "
        "Thinking mode off. Temperature 0. "
        "These lines quote pilot replies and must not be redistributed.",
        "",
        "| Result | Expected | Got | wrong_role | Reply |",
        "| --- | --- | --- | --- | --- |",
    ]
    wrong = 0
    for case in GOLD:
        item = items[case["item_id"]]
        reply = _reply_for(item, generations, model_id, case["listener"])
        raw = call_judge(config, SYSTEM_PROMPT, judge_user(item["context"], reply, qwen=qwen), spec)
        parsed = parse_reply_judgment(raw)
        got = None if parsed is None else parsed["category"]
        role = None if parsed is None else parsed["wrong_role"]
        ok = got == case["expected"] and role is False
        if not ok:
            wrong += 1
        mark = "ok" if ok else "WRONG"
        shown = reply.replace("|", "/")
        lines.append(
            f"| {mark} | {case['expected']} | {got} | {role} | {case['listener']} {shown} |"
        )
        print(f"gold {mark} {case['listener']} {case['item_id']} -> {got} role={role}", flush=True)
    lines.extend(["", f"Wrong: {wrong} of {len(GOLD)}.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    note = (
        f"Gold set before the pilot coding: {len(GOLD) - wrong} of {len(GOLD)} matched. "
        f"Wrong: {wrong}. Detail is in `reports/judge_gold.md`."
    )
    return wrong, note


def _slice_rate(rows: list[dict], trouble: str) -> str:
    chosen = [row for row in rows if row.get("trouble_class") == trouble and not row.get("failed")]
    hits = sum(1 for row in chosen if row.get("category") in CLARIFICATION)
    return _rate(hits, len(chosen))


def _corrects_rate(rows: list[dict]) -> str:
    chosen = [
        row
        for row in rows
        if row.get("trouble_class") == "error" and row.get("corrects_error") is not None and not row.get("failed")
    ]
    hits = sum(1 for row in chosen if row.get("corrects_error"))
    return _rate(hits, len(chosen))


def _quiet_rate(rows: list[dict], recent_trouble: dict[str, bool]) -> str:
    chosen = [
        row
        for row in rows
        if row.get("trouble_class") == "none"
        and recent_trouble.get(row["item_id"]) is False
        and not row.get("failed")
    ]
    hits = sum(1 for row in chosen if row.get("category") in CLARIFICATION)
    return _rate(hits, len(chosen))


def comparison_judge(speaker: str, claude_model: str, gpt_model: str) -> str:
    if "claude" in speaker.lower():
        return gpt_model
    return claude_model


def write_fleet_report(
    path: Path,
    records: list[dict],
    *,
    speakers: list[str],
    claude_model: str,
    gpt_model: str,
    recent_trouble: dict[str, bool],
) -> None:
    replies = [row for row in records if row.get("task") == "reply" and not row.get("failed")]
    lines = [
        "# Pilot coding, all listeners",
        "",
        "All three local models are 3B to 4B parameters because of an 8 GB memory limit.",
        "",
        f"Claude replies are labelled by `{gpt_model}` and compared with the human replies "
        f"labelled by `{gpt_model}`. Gemini, Llama, Qwen, Gemma, and GPT replies are labelled "
        f"by `{claude_model}` and compared with the human replies labelled by `{claude_model}`.",
        "",
        "## Summary",
        "",
        "| Listener | Condition | Judge | Clarification at unintelligible | "
        "corrects_error at errors | Clarification at baseline, recent_trouble false |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for judge in (claude_model, gpt_model):
        rows = _listener_rows(replies, "human", judge, "human")
        lines.append(
            f"| human | human reply | {judge} | {_slice_rate(rows, 'unintelligible')} | "
            f"{_corrects_rate(rows)} | {_quiet_rate(rows, recent_trouble)} |"
        )
    for speaker in speakers:
        judge = comparison_judge(speaker, claude_model, gpt_model)
        for condition in GPT_REPLY_LISTENERS:
            rows = _listener_rows(replies, condition, judge, speaker)
            if not rows:
                continue
            lines.append(
                f"| {speaker} | {condition} | {judge} | {_slice_rate(rows, 'unintelligible')} | "
                f"{_corrects_rate(rows)} | {_quiet_rate(rows, recent_trouble)} |"
            )
    lines.append("")
    for speaker in speakers:
        judge = comparison_judge(speaker, claude_model, gpt_model)
        lines.append(f"## {speaker}")
        lines.append("")
        lines.append(f"Compared with humans labelled by `{judge}`.")
        lines.append("")
        _append_rate_table(lines, f"human ({judge})", _listener_rows(replies, "human", judge, "human"))
        for condition in GPT_REPLY_LISTENERS:
            rows = _listener_rows(replies, condition, judge, speaker)
            if rows:
                _append_rate_table(lines, f"{condition} ({judge})", rows)
        human_base = {
            row["item_id"]: row
            for row in _listener_rows(replies, "human", judge, "human")
            if row.get("trouble_class") == "none"
        }
        repair_base = {
            row["item_id"]: row
            for row in _listener_rows(replies, "repair", judge, speaker)
            if row.get("trouble_class") == "none"
        }
        paired = sorted(set(human_base) & set(repair_base))
        quiet = [item_id for item_id in paired if recent_trouble.get(item_id) is False]

        def rate(ids: list[str], pool: dict[str, dict]) -> str:
            chosen = [pool[item_id] for item_id in ids if item_id in pool]
            hits = sum(1 for row in chosen if row.get("category") in CLARIFICATION)
            return _rate(hits, len(chosen))

        lines.extend(
            [
                "| Items | N | Human clarification | Repair clarification |",
                "| --- | ---: | ---: | ---: |",
                f"| All baseline | {len(paired)} | {rate(paired, human_base)} | {rate(paired, repair_base)} |",
                f"| recent_trouble false | {len(quiet)} | {rate(quiet, human_base)} | {rate(quiet, repair_base)} |",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Code pilot replies with the judge in config.yaml.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if not config.get("judge_model"):
        raise SystemExit("Set judge_model in config.yaml.")
    if config.get("judge_reasoning"):
        raise SystemExit("judge_reasoning must be false. Thinking mode stays off.")
    speaker_ids = {model["id"] for model in config["models"]}
    if config["judge_model"] in speaker_ids:
        print(
            "Note: the labeller is one of the tested speaker models. "
            "It will score its own replies.",
            flush=True,
        )
    items = load_items(PROJECT_ROOT / str(config.get("items_path", "data/processed/items.jsonl")))
    pilot_path = PROJECT_ROOT / str(config.get("pilot_items_path", "data/processed/pilot_items.txt"))
    pilot_ids = [line.strip() for line in pilot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {item["item_id"]: item for item in items}
    missing = [item_id for item_id in pilot_ids if item_id not in by_id or by_id[item_id].get("exclude")]
    if missing:
        raise SystemExit("pilot file contains unknown or excluded items: " + ", ".join(missing))
    pilot_items = [by_id[item_id] for item_id in pilot_ids]
    generations = _load_jsonl(PROJECT_ROOT / str(config.get("generations_path", "outputs/generations.jsonl")))
    pilot_set = set(pilot_ids)
    present: list[str] = []
    seen: set[str] = set()
    for model in config["models"]:
        if any(row.get("model") == model["id"] and row.get("item_id") in pilot_set for row in generations):
            present.append(model["id"])
            seen.add(model["id"])
    for row in generations:
        speaker = row.get("model")
        if row.get("item_id") in pilot_set and speaker and speaker not in seen:
            present.append(str(speaker))
            seen.add(str(speaker))
    model_id = next(model["id"] for model in config["models"] if model["provider"] == "openai")
    gold_path = PROJECT_ROOT / "reports/judge_gold.md"
    if args.skip_gold:
        wrong = 0
        gold_note = (
            "The 19-case gold set was run earlier on claude-sonnet-5-5 and matched 19 of 19. "
            "It was not repeated for this coding."
        )
    else:
        wrong, gold_note = run_gold(config, by_id, generations, model_id, gold_path)
    print(gold_note, flush=True)
    if wrong > 2 or args.gold_only:
        usage = config.get("_judge_usage") or {}
        print(
            f"judge tokens in {usage.get('input', 0)} out {usage.get('output', 0)}",
            flush=True,
        )
        print(f"wrote {gold_path}", flush=True)
        return 2 if wrong > 2 else 0
    tasks = build_tasks(pilot_items, generations, present or [model_id])
    coding_path = PROJECT_ROOT / str(config.get("codings_path", "outputs/codings.jsonl"))
    records = code_tasks(config, tasks, coding_path)
    report = PROJECT_ROOT / str(config.get("coding_report_path", "reports/coding_gpt_pilot.md"))
    judges = config.get("judges") or {}
    write_report(
        report,
        records,
        gold_note=gold_note,
        provisional=bool(config.get("judge_provisional")),
        previous=None,
        claude_model=str((judges.get("gpt_replies") or {}).get("model") or "claude-sonnet-5-5"),
        gpt_model=str((judges.get("other_replies") or {}).get("model") or "gpt-4.1-2025-04-14"),
        recent_trouble=(
            recent := {
                item_id: item["recent_trouble"]
                for item_id, item in by_id.items()
                if item.get("trouble_class") == "none" and isinstance(item.get("recent_trouble"), bool)
            }
        ),
        speaker_model=model_id,
    )
    fleet = PROJECT_ROOT / "reports" / "coding_pilot_all.md"
    write_fleet_report(
        fleet,
        records,
        speakers=present or [model_id],
        claude_model=str((judges.get("gpt_replies") or {}).get("model") or "claude-sonnet-5-5"),
        gpt_model=str((judges.get("other_replies") or {}).get("model") or "gpt-4.1-2025-04-14"),
        recent_trouble=recent,
    )
    print(f"wrote {fleet}", flush=True)
    sheet = PROJECT_ROOT / str(config.get("validation_sheet_path", "results/validation_pilot.csv"))
    key_path = PROJECT_ROOT / str(config.get("validation_key_path", "results/validation_pilot_key.csv"))
    n_sheet = 0
    if not args.skip_validation:
        n_sheet = write_validation(records, sheet, key_path)
    usage = config.get("_judge_usage") or {}
    spent = (usage.get("input", 0) / 1_000_000) * 2 + (usage.get("output", 0) / 1_000_000) * 8
    print(
        f"judge tokens in {usage.get('input', 0)} out {usage.get('output', 0)} "
        f"calls {usage.get('calls', 0)} cost ${spent:.4f}",
        flush=True,
    )
    print(f"wrote {report}", flush=True)
    if not args.skip_validation:
        print(f"wrote {sheet} ({n_sheet} rows) and {key_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
