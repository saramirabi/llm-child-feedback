#!/usr/bin/env python3
"""Sample trouble points and render them as a listener would hear them.

Raw .cha files are read and never written. Child IDs follow the child-id
check: Feldman Don is the first session of D03, and the Rescorla Gravina
twins are gr1 (Chris) and gr2 (Andrew).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.troublepoints import (
    ADULT_ROLES,
    BULLET_RE,
    CORPUS_ORDER,
    DEFAULT_DATA_ROOT,
    SEED,
    TERMINATOR_RE,
    FileRecord,
    Token,
    _child_id,
    _corpus_files,
    _folder_group,
    _lexical_base,
    _resolve_ellisweismer,
    _skip_reason,
    classify_utterance,
    digit_age_folder,
    load_ellisweismer_status,
    parse_chat,
    read_chat,
    visibility_failure,
    _word_token,
)

K = 10
CONTEXT_TURNS = 10
TROUBLE_TYPES = ("unintelligible", "error_word", "phonological_form")
ITEM_TYPES = TROUBLE_TYPES + ("none",)
PREVIOUS_TOTALS = {
    "unintelligible": 3400,
    "error_word": 2045,
    "phonological_form": 907,
    "none": 6352,
}
PREVIOUS_TOTAL = 12704

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "items.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "render.md"

# Listener labels. Other child roles are never listeners.
CHILD_ROLES = frozenset({"Child", "Sibling", "Sister", "Brother"})
# Titles whose hidden name is coded Title_xxx. Mrs_xxx is the stated case;
# the same files also use Mr_xxx.
NAME_TITLES = {"mrs": "Mrs", "mr": "Mr", "miss": "Miss", "ms": "Ms"}

# Roster assignment from reports/child_id_check.md. Filename digits at 48
# months do not follow the child: gr248 is Chris (gr1) and gr148 is Drew (gr2).
RESCORLA_FILE_CHILD = {
    "Rescorla/TD/36/gr36b.cha": "gr1",
    "Rescorla/TD/48/gr248.cha": "gr1",
    "Rescorla/TD/60/gr160.cha": "gr1",
    "Rescorla/TD/36/gr36a.cha": "gr2",
    "Rescorla/TD/48/gr148.cha": "gr2",
    "Rescorla/TD/60/0nomedia/gr260.cha": "gr2",
}

ITEM_FIELDS = (
    "item_id",
    "session_id",
    "corpus",
    "group",
    "child_id",
    "child_age_months",
    "trouble_type",
    "trouble_class",
    "all_trouble_types",
    "in_group_contrast",
    "gold_target",
    "target_in_context",
    "context",
    "human_next_turn",
    "adult_role",
    "exclude",
    "exclude_reason",
    "recent_trouble",
)


@dataclass
class Session:
    corpus: str
    path: Path
    rel: str
    session_id: str
    child_id: str
    parser_child_id: str
    group: str
    age_months: float | None
    turns: list = field(default_factory=list)


@dataclass
class Item:
    item_id: str
    session_id: str
    corpus: str
    group: str
    child_id: str
    child_age_months: float | None
    trouble_type: str
    trouble_class: str
    all_trouble_types: list[str]
    in_group_contrast: bool
    gold_target: list[str] | None
    target_in_context: bool
    context: str
    human_next_turn: str
    adult_role: str
    exclude: bool = False
    exclude_reason: str | None = None
    recent_trouble: bool | None = None

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "session_id": self.session_id,
            "corpus": self.corpus,
            "group": self.group,
            "child_id": self.child_id,
            "child_age_months": self.child_age_months,
            "trouble_type": self.trouble_type,
            "trouble_class": self.trouble_class,
            "all_trouble_types": self.all_trouble_types,
            "in_group_contrast": self.in_group_contrast,
            "gold_target": self.gold_target,
            "target_in_context": self.target_in_context,
            "context": self.context,
            "human_next_turn": self.human_next_turn,
            "adult_role": self.adult_role,
            "exclude": self.exclude,
            "exclude_reason": self.exclude_reason,
            "recent_trouble": self.recent_trouble,
        }


def canonical_child_id(corpus: str, rel: str, child_id: str) -> str:
    """One ID per child. Feldman Don is D03. Gravina files follow the roster."""
    if corpus == "Feldman" and child_id == "DON":
        return "D03"
    if corpus == "Rescorla":
        return RESCORLA_FILE_CHILD.get(rel.replace("\\", "/"), child_id)
    return child_id


def sampling_unit(corpus: str, child_id: str, session_id: str) -> str:
    """Conti3 is capped per session. Every other corpus is capped per child."""
    if corpus == "Conti3":
        return f"{child_id}|{session_id}"
    return child_id


def unit_rng(unit: str) -> random.Random:
    """Stable RNG. The seed string is exactly 42:<unit>."""
    return random.Random(f"42:{unit}")


# The [*] on these turns sits on an omitted word or a stray bracket, so the
# listener string has no error word. Each one takes its matched baseline slot
# with it. sample_unit lays trouble turns down first and then the same number
# of baseline turns, so slot i of the baseline list matches trouble turn i.
NO_VISIBLE_ERROR_IDS = frozenset(
    {
        "Feldman/ParentChild/SLI/PC-ac/bot33.cha:45",
        "Feldman/ParentChild/SLI/PC-g/gla21.cha:1013",
        "Feldman/ParentChild/SLI/PC-sz/shn39.cha:469",
        "EllisWeismer/LT/30pc/11068.cha:156",
    }
)
NO_VISIBLE_ERROR_REASON = (
    "no visible error word: the [*] marks an omitted word or a stray bracket"
)


def mark_matched_exclusions(chosen: list[Item]) -> list[str]:
    """Set exclude on the invisible error items and the baseline slot of each.

    Returns error item ids that had no baseline slot left to match.
    """
    for item in chosen:
        item.exclude = False
        item.exclude_reason = None
    trouble = [item for item in chosen if item.trouble_type != "none"]
    baseline = [item for item in chosen if item.trouble_type == "none"]
    missing: list[str] = []
    for index, trouble_item in enumerate(trouble):
        if trouble_item.item_id not in NO_VISIBLE_ERROR_IDS:
            continue
        trouble_item.exclude = True
        trouble_item.exclude_reason = NO_VISIBLE_ERROR_REASON
        if index >= len(baseline):
            missing.append(trouble_item.item_id)
            continue
        partner = baseline[index]
        partner.exclude = True
        partner.exclude_reason = f"matched baseline of {trouble_item.item_id}"
    return missing


def speaker_label(role: str) -> str:
    if role == "Target_Child":
        return "Child"
    if role in ADULT_ROLES:
        return "Adult"
    if role in CHILD_ROLES:
        return "Other child"
    return "Other"


def _name_placeholder(form: str) -> str | None:
    match = re.fullmatch(r"(mrs|mr|miss|ms)_xxx", form, flags=re.IGNORECASE)
    if not match:
        return None
    return f"{NAME_TITLES[match.group(1).lower()]} (name)"


# (i)s, (u)s, (doe)s, (a)m: the parenthetical was not spoken, and the
# letter that remains is a contraction. (a)s_well keeps the rest of that
# word ("'s well"). s(ome) and m(om) omit an ending and stay a fragment.
_OMITTED_CLITIC = re.compile(
    r"^(?P<host>.*?)\((?P<omitted>[^)]+)\)(?P<letter>[sm])(?:_(?P<rest>[A-Za-z0-9_']+))?$",
    re.IGNORECASE,
)


def _omitted_clitic(token_text: str) -> tuple[str, str, str] | None:
    """Return (host, letter, rest) for an omitted-sound contraction, or None.

    The host is empty when the token is only the clitic, as in `(i)s`.
    `rest` is the remainder of an underscore compound, as in `(a)s_well`.
    """
    base = _lexical_base(token_text).strip(".,?!;:\"")
    match = _OMITTED_CLITIC.fullmatch(base)
    if not match:
        return None
    host = match.group("host").replace("_", " ").strip(".,?!;:\"")
    rest = (match.group("rest") or "").replace("_", " ")
    return host, match.group("letter").lower(), rest


def _attach_clitic(previous: str, letter: str) -> str:
    head, sep, tail = previous.rpartition(" ")
    tail = f"{tail}'{letter}"
    return f"{head}{sep}{tail}" if sep else tail


def _expand_repetitions(form: str) -> str:
    """Turn ↫sh-sh↫she into sh-sh-she. The marker itself is never spoken."""
    marker = "\u21ab"
    while marker in form:
        match = re.search(r"\u21ab([^\u21ab]*)\u21ab", form)
        if not match:
            form = form.replace(marker, "")
            break
        fragment = match.group(1).strip()
        rest = form[match.end() :]
        if not fragment:
            spoken = rest
        elif fragment.endswith("-"):
            spoken = fragment + rest
        elif rest:
            spoken = f"{fragment}-{rest}"
        else:
            spoken = f"{fragment}-"
        form = form[: match.start()] + spoken
    return form


def _matching_closer(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index of the closer that matches text[start], or -1.

    Angle brackets nest. A `[...]` code is opaque, so a `>` inside `[>]`
    does not close a group.
    """
    depth = 1
    i = start + 1
    n = len(text)
    in_code = False
    while i < n:
        ch = text[i]
        if in_code:
            if ch == "]":
                in_code = False
            i += 1
            continue
        if ch == "[":
            in_code = True
            i += 1
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def tokenize_spoken(text: str) -> list[Token]:
    """Listener tokens. Groups nest, and every `[...]` code is one token."""
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace() or text[i] in ">]":
            i += 1
            continue
        if text[i] == "<":
            end = _matching_closer(text, i, "<", ">")
            if end < 0:
                i += 1
                continue
            tokens.append(Token("group", text[i : end + 1]))
            i = end + 1
            continue
        if text[i] == "[":
            end = text.find("]", i + 1)
            if end < 0:
                while i < n and not text[i].isspace():
                    i += 1
                continue
            tokens.append(Token("code", text[i : end + 1]))
            i = end + 1
            continue
        end = i
        while end < n and not text[end].isspace() and text[end] not in "<[>]":
            end += 1
        tokens.append(_word_token(text[i:end]))
        i = end
    return tokens


def _render_word(form: str) -> str:
    form = _expand_repetitions(form)
    named = _name_placeholder(_lexical_base(form))
    if named:
        return named
    spoken = re.sub(r"\([^)]*\)", "", _lexical_base(form))
    spoken = spoken.replace("_", " ")
    pieces: list[str] = []
    for bit in spoken.split():
        lowered = bit.lower()
        if lowered in {"xxx", "yyy"}:
            pieces.append("(unintelligible)")
        elif lowered == "www":
            pieces.append("(untranscribed)")
        else:
            cleaned = _drop_lengthening(bit)
            if cleaned:
                pieces.append(cleaned)
    return " ".join(pieces)


def _drop_lengthening(text: str) -> str:
    """CHAT writes a lengthened sound with a colon inside the word: a:h is ah."""
    return text.strip(".,?!;:\"").replace(":", "")


def _render_nonlex(form: str) -> str:
    if form.startswith("&="):
        return ""
    if form.startswith("&-"):
        return _drop_lengthening(form[2:])
    if form.startswith("&+"):
        fragment = _drop_lengthening(form[2:])
        if not fragment:
            return ""
        return fragment if fragment.endswith("-") else f"{fragment}-"
    if form.startswith("&~"):
        return _drop_lengthening(form[2:])
    if form.startswith("&"):
        return _drop_lengthening(form[1:])
    return ""


def _render_token(token: Token) -> str:
    if token.kind in {"pause", "code", "omission", "bare_zero"}:
        return ""
    if token.kind == "nonlex":
        return _render_nonlex(token.text)
    if token.kind == "group":
        inner = token.text[1:-1].strip()
        if not inner:
            return ""
        return render_utterance(inner)
    if token.text.startswith("+"):
        return ""
    return _render_word(token.text)


def render_utterance(text: str) -> str:
    """Surface form a listener would hear. Coding, targets, and events are gone.

    A lone omitted-sound clitic joins the previous word: `that (i)s` is
    "that's", `I (a)m` is "I'm". It is not left as a separate "s" or "m".
    """
    cleaned = BULLET_RE.sub(" ", text or "")
    cleaned = cleaned.replace("\x15", " ")
    cleaned = TERMINATOR_RE.sub("", cleaned.strip())
    pieces: list[str] = []
    for token in tokenize_spoken(cleaned):
        if token.kind == "word":
            clitic = _omitted_clitic(token.text)
            if clitic is not None:
                host, letter, rest = clitic
                if rest:
                    head = f"{host}'{letter}" if host else f"'{letter}"
                    pieces.append(f"{head} {rest}")
                elif host:
                    pieces.append(f"{host}'{letter}")
                elif pieces:
                    pieces[-1] = _attach_clitic(pieces[-1], letter)
                else:
                    pieces.append(f"'{letter}")
                continue
        piece = _render_token(token)
        if piece:
            pieces.append(piece)
    # CHAT „ (U+201E) marks a tag question. It is a comma. Curly quotes stay.
    text = " ".join(pieces).replace("\u201e", ",")
    text = re.sub(r"\s+,", ",", text)
    return re.sub(r"\s+", " ", text).strip()


def trouble_class(trouble_type: str) -> str:
    if trouble_type == "unintelligible":
        return "unintelligible"
    if trouble_type in {"error_word", "phonological_form"}:
        return "error"
    return "none"


def _phrase_in_text(phrase: str, text: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def target_appears_earlier(gold: list[str], earlier_context: str) -> bool:
    return any(target and _phrase_in_text(target, earlier_context) for target in gold)


def render_context(turns, index: int) -> tuple[str, str]:
    """Labeled context ending at the child turn, plus the text before that turn."""
    start = max(0, index - (CONTEXT_TURNS - 1))
    lines: list[str] = []
    earlier: list[str] = []
    for turn in turns[start:index]:
        rendered = render_utterance(turn.text)
        if not rendered:
            continue
        line = f"{speaker_label(turn.role)}: {rendered}"
        lines.append(line)
        earlier.append(line)
    child = render_utterance(turns[index].text)
    lines.append(f"Child: {child}")
    return "\n".join(lines), "\n".join(earlier)


def recent_trouble_in_context(turns, index: int) -> bool:
    """True when one of the previous three target-child turns in the context has trouble.

    The window is the same ten turns render_context shows. A turn counts only
    when the listener string is not empty. Trouble is read from the original
    CHAT: xxx or yyy, a [*] on a word the child produced, or a [: target]
    phonological form. Other-child turns are not counted.
    """
    start = max(0, index - (CONTEXT_TURNS - 1))
    shown = []
    for turn in turns[start:index]:
        if turn.role != "Target_Child":
            continue
        if not render_utterance(turn.text):
            continue
        shown.append(turn)
    for turn in shown[-3:]:
        if classify_utterance(turn.text).trouble is not None:
            return True
    return False


def load_sessions(root: Path) -> tuple[list[Session], list[str]]:
    """Sessions kept by the same file rules as the trouble-point extractor."""
    status_path = root / "EllisWeismer" / "0status.xlsx"
    status = load_ellisweismer_status(status_path) if status_path.exists() else {}
    pending: list[tuple[Session, FileRecord]] = []
    notes: list[str] = []
    records: list[FileRecord] = []

    for corpus, path in _corpus_files(root):
        rel = path.relative_to(root).as_posix()
        reason = _skip_reason(corpus, path, include_unrelated=False)
        if reason:
            continue
        text, _replaced = read_chat(path)
        parsed = parse_chat(text)
        targets = parsed.target_children()
        if len(targets) != 1:
            continue
        age_folder = digit_age_folder(path, root)
        parser_id = _child_id(corpus, path, parsed, age_folder)
        if not parser_id:
            continue
        header_age = targets[0].age_months
        folder_age = float(age_folder) if age_folder else None
        age = folder_age if header_age is None else header_age
        group = _folder_group(corpus, path)
        record = FileRecord(
            corpus=corpus,
            path=path,
            rel=rel,
            child_id=parser_id,
            folder_group=group,
            group=group,
            age_months=age,
            folder_age=folder_age,
        )
        session = Session(
            corpus=corpus,
            path=path,
            rel=rel,
            session_id=Path(rel).with_suffix("").as_posix(),
            child_id=parser_id,
            parser_child_id=parser_id,
            group=group,
            age_months=age,
            turns=parsed.turns,
        )
        pending.append((session, record))
        records.append(record)

    _resolve_ellisweismer(records, status)
    sessions: list[Session] = []
    for session, record in pending:
        if record not in records:
            notes.append(f"{session.rel}: dropped during EllisWeismer status resolution")
            continue
        session.group = record.group
        session.child_id = canonical_child_id(session.corpus, session.rel, session.parser_child_id)
        sessions.append(session)
    return sessions, notes


def _eligible_turn(turn, follower_role: str) -> bool:
    if turn.role != "Target_Child" or follower_role not in ADULT_ROLES:
        return False
    classified = classify_utterance(turn.text)
    if classified.exclusion:
        return False
    if classified.trouble is None and turn.has_err_tier:
        return False
    if visibility_failure(turn.text, classified, turn.err_text):
        return False
    return True


def build_candidates(sessions: list[Session]) -> tuple[list[Item], dict[str, int]]:
    """Render every eligible turn. Empty child turns and empty replies are dropped."""
    items: list[Item] = []
    dropped = {
        "empty_child": 0,
        "empty_adult": 0,
        "empty_child_trouble": 0,
        "empty_adult_trouble": 0,
    }
    for session in sessions:
        turns = session.turns
        contrast = session.corpus != "Conti3"
        for index, turn in enumerate(turns):
            nxt = turns[index + 1] if index + 1 < len(turns) else None
            follower = nxt.role if nxt is not None else ""
            if not _eligible_turn(turn, follower):
                continue
            classified = classify_utterance(turn.text)
            is_trouble = classified.trouble is not None
            child_text = render_utterance(turn.text)
            if not child_text:
                dropped["empty_child"] += 1
                if is_trouble:
                    dropped["empty_child_trouble"] += 1
                continue
            adult_text = render_utterance(nxt.text)
            if not adult_text:
                dropped["empty_adult"] += 1
                if is_trouble:
                    dropped["empty_adult_trouble"] += 1
                continue
            trouble = classified.trouble or "none"
            context, earlier = render_context(turns, index)
            recent = recent_trouble_in_context(turns, index) if trouble == "none" else None
            gold = list(classified.gold_targets) or None
            items.append(
                Item(
                    item_id=f"{session.rel}:{index}",
                    session_id=session.session_id,
                    corpus=session.corpus,
                    group=session.group,
                    child_id=session.child_id,
                    child_age_months=session.age_months,
                    trouble_type=trouble,
                    trouble_class=trouble_class(trouble),
                    all_trouble_types=list(classified.present_types),
                    in_group_contrast=contrast,
                    gold_target=gold,
                    target_in_context=target_appears_earlier(gold or [], earlier),
                    context=context,
                    human_next_turn=adult_text,
                    adult_role=follower,
                    recent_trouble=recent,
                )
            )
    items.sort(key=lambda item: item.item_id)
    return items, dropped


def sample_unit(items: list[Item], rng: random.Random, k: int) -> tuple[list[Item], int]:
    """Up to k of each trouble type, then that many baseline turns.

    Returns the sample and how many baseline turns were missing.
    """
    by_type: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_type[item.trouble_type].append(item)
    chosen: list[Item] = []
    for trouble in TROUBLE_TYPES:
        pool = list(by_type.get(trouble, []))
        rng.shuffle(pool)
        chosen.extend(pool[:k])
    baseline = list(by_type.get("none", []))
    rng.shuffle(baseline)
    need = len(chosen)
    chosen.extend(baseline[:need])
    short = max(0, need - len(baseline))
    return chosen, short


def subsample_child(items: list[Item], frac: float, rng: random.Random) -> list[Item]:
    """Keep frac of each trouble type. A child who had items keeps at least one."""
    if frac >= 1 or not items:
        return list(items)
    by_type: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_type[item.trouble_type].append(item)
    kept: list[Item] = []
    for trouble in ITEM_TYPES:
        pool = list(by_type.get(trouble, []))
        if not pool:
            continue
        rng.shuffle(pool)
        n_keep = min(len(pool), int(math.floor(len(pool) * frac + 0.5)))
        kept.extend(pool[:n_keep])
    if not kept:
        kept.append(min(items, key=lambda item: item.item_id))
    return kept


def sample_items(
    candidates: list[Item],
    *,
    k: int = K,
    subsample_frac: float = 1.0,
) -> tuple[list[Item], dict[str, int]]:
    grouped: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in candidates:
        unit = sampling_unit(item.corpus, item.child_id, item.session_id)
        grouped[(item.corpus, unit)].append(item)

    selected: list[Item] = []
    short_baseline = 0
    for (corpus, unit), pool in sorted(grouped.items()):
        pool = sorted(pool, key=lambda item: item.item_id)
        chosen, short = sample_unit(pool, unit_rng(unit), k)
        short_baseline += short
        unmatched = mark_matched_exclusions(chosen)
        if unmatched:
            raise RuntimeError(
                "no baseline slot for excluded error items: " + ", ".join(unmatched)
            )
        selected.extend(chosen)

    found = {item.item_id for item in selected if item.exclude and item.trouble_type != "none"}
    if found != NO_VISIBLE_ERROR_IDS:
        missing = NO_VISIBLE_ERROR_IDS - found
        extra = found - NO_VISIBLE_ERROR_IDS
        raise RuntimeError(
            "excluded error items do not match the invisible-error list: "
            f"missing {sorted(missing)}; extra {sorted(extra)}"
        )

    if subsample_frac < 1:
        by_child: dict[tuple[str, str], list[Item]] = defaultdict(list)
        for item in selected:
            by_child[(item.corpus, item.child_id)].append(item)
        reduced: list[Item] = []
        for (corpus, child_id), pool in sorted(by_child.items()):
            rng = random.Random(f"42:{corpus}|{child_id}|subsample")
            reduced.extend(subsample_child(pool, subsample_frac, rng))
        selected = reduced

    selected.sort(key=lambda item: (CORPUS_ORDER.index(item.corpus), item.child_id, item.item_id))
    return selected, {"short_baseline": short_baseline}


def children_in(sessions_or_items, *, trouble_only: bool = False) -> dict[tuple[str, str], set[str]]:
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in sessions_or_items:
        if trouble_only and getattr(row, "trouble_type", None) in {None, "none"}:
            continue
        found[(row.corpus, row.group)].add(row.child_id)
    return found


def _group_names(corpus: str) -> list[str]:
    if corpus == "Feldman":
        return ["focal_lesion", "td"]
    if corpus in {"Conti1", "Conti2"}:
        return ["dld", "td_sibling"]
    if corpus == "Conti3":
        return ["dld"]
    return ["late_talker", "td"]


def _exclusion_note(items: list[Item]) -> str:
    excluded = [item for item in items if item.exclude]
    usable = len(items) - len(excluded)
    lines = [
        f"Usable items: {usable}. Excluded: {len(excluded)}. "
        "Later generation, coding, and analysis skip `exclude: true`.",
    ]
    for item in excluded:
        lines.append(f"- `{item.item_id}` ({item.trouble_type}): {item.exclude_reason}")
    return "\n".join(lines)


def write_jsonl(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")


def _count_table(items: list[Item]) -> list[str]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        counts[(item.corpus, item.group)][item.trouble_type] += 1
    lines = [
        "| Corpus | Group | Unintelligible | Error word | Phonological form | Baseline | Total |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    grand = defaultdict(int)
    for corpus in CORPUS_ORDER:
        for group in _group_names(corpus):
            row = counts.get((corpus, group), {})
            total = sum(row.get(kind, 0) for kind in ITEM_TYPES)
            for kind in ITEM_TYPES:
                grand[kind] += row.get(kind, 0)
            grand["total"] += total
            lines.append(
                f"| {corpus} | {group} | {row.get('unintelligible', 0)} | "
                f"{row.get('error_word', 0)} | {row.get('phonological_form', 0)} | "
                f"{row.get('none', 0)} | {total} |"
            )
    lines.append(
        f"| Total |  | {grand['unintelligible']} | {grand['error_word']} | "
        f"{grand['phonological_form']} | {grand['none']} | {grand['total']} |"
    )
    return lines


def _gold_table(items: list[Item]) -> list[str]:
    items_with = defaultdict(int)
    strings = defaultdict(int)
    for item in items:
        if not item.gold_target:
            continue
        items_with[item.corpus] += 1
        strings[item.corpus] += len(item.gold_target)
    lines = [
        "| Corpus | Items with a gold target | Gold target strings |",
        "| --- | ---: | ---: |",
    ]
    for corpus in CORPUS_ORDER:
        lines.append(f"| {corpus} | {items_with[corpus]} | {strings[corpus]} |")
    lines.append(f"| Total | {sum(items_with.values())} | {sum(strings.values())} |")
    return lines


def _child_table(
    sessions: list[Session],
    sampled_trouble: list[Item],
    final_items: list[Item],
) -> list[str]:
    transcripts = children_in(sessions)
    trouble = children_in(sampled_trouble, trouble_only=True)
    final = children_in(final_items)
    lines = [
        "| Corpus | Group | Children in the transcripts | Children with at least one sampled trouble item | Children in the final items.jsonl |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for corpus in CORPUS_ORDER:
        for group in _group_names(corpus):
            key = (corpus, group)
            lines.append(
                f"| {corpus} | {group} | {len(transcripts.get(key, ()))} | "
                f"{len(trouble.get(key, ()))} | {len(final.get(key, ()))} |"
            )
    return lines


def _preview(items: list[Item], n: int = 15, seed: int = SEED) -> list[Item]:
    if len(items) <= n:
        return list(items)
    rng = random.Random(seed)
    return rng.sample(list(items), n)


def _format_preview(items: list[Item]) -> list[str]:
    blocks: list[str] = []
    for number, item in enumerate(items, start=1):
        gold = item.gold_target if item.gold_target else "null"
        blocks.append(
            "\n".join(
                [
                    f"### {number}. `{item.item_id}`",
                    "",
                    f"- {item.corpus} / {item.group} / {item.child_id} / {item.trouble_type} / {item.adult_role}",
                    f"- gold_target: {gold}; target_in_context: {str(item.target_in_context).lower()}; "
                    f"in_group_contrast: {str(item.in_group_contrast).lower()}",
                    "",
                    "```",
                    item.context,
                    f"Adult: {item.human_next_turn}",
                    "```",
                ]
            )
        )
    return blocks


def explain_previous(
    current: dict[str, int],
    uncorrected: dict[str, int] | None,
    dropped: dict[str, int],
    short_baseline: int,
) -> list[str]:
    total = sum(current.values())
    lines = [
        f"The previous build at K = {K} had {PREVIOUS_TOTAL} items "
        f"(unintelligible {PREVIOUS_TOTALS['unintelligible']}, "
        f"error_word {PREVIOUS_TOTALS['error_word']}, "
        f"phonological_form {PREVIOUS_TOTALS['phonological_form']}, "
        f"none {PREVIOUS_TOTALS['none']}).",
        f"This build has {total} items "
        f"(unintelligible {current.get('unintelligible', 0)}, "
        f"error_word {current.get('error_word', 0)}, "
        f"phonological_form {current.get('phonological_form', 0)}, "
        f"none {current.get('none', 0)}).",
        f"The difference is {total - PREVIOUS_TOTAL:+d} items.",
    ]
    if total == PREVIOUS_TOTAL and all(current.get(kind, 0) == PREVIOUS_TOTALS[kind] for kind in ITEM_TYPES):
        lines.append("The totals match the previous build.")
        return lines
    lines.append(
        "Baseline still matches the trouble sample one for one "
        f"({short_baseline} units were short of baseline turns). "
        "The gap is therefore fewer trouble slots, plus the baseline slot matched to each one."
    )
    if uncorrected is not None:
        old_total = sum(uncorrected.values())
        lines.append(
            "With the extractor's old IDs (Don separate from D03, and gr36a / gr36b "
            f"separate from gr1 / gr2) the same render produces {old_total} items "
            f"(unintelligible {uncorrected.get('unintelligible', 0)}, "
            f"error_word {uncorrected.get('error_word', 0)}, "
            f"phonological_form {uncorrected.get('phonological_form', 0)}, "
            f"none {uncorrected.get('none', 0)}). "
            f"Joining those files into one child removes {old_total - total} items, "
            f"because the joined files share one cap of {K} per trouble type."
        )
        remain = PREVIOUS_TOTAL - old_total
        lines.append(
            f"The other {remain} items are {remain // 2} trouble slots and the baseline slots matched to them. "
            f"Against the previous mix, unintelligible is higher "
            f"({uncorrected.get('unintelligible', 0)} versus {PREVIOUS_TOTALS['unintelligible']}) "
            f"and error_word is lower "
            f"({uncorrected.get('error_word', 0)} versus {PREVIOUS_TOTALS['error_word']}). "
            f"Phonological form is {uncorrected.get('phonological_form', 0)} versus "
            f"{PREVIOUS_TOTALS['phonological_form']}."
        )
    trouble_drops = dropped.get("empty_child_trouble", 0) + dropped.get("empty_adult_trouble", 0)
    lines.append(
        f"{dropped.get('empty_child', 0)} child turns rendered empty and "
        f"{dropped.get('empty_adult', 0)} adult replies rendered empty, so those turns were not sampled. "
        f"{trouble_drops} of the dropped turns were trouble turns "
        f"({dropped.get('empty_child_trouble', 0)} empty child lines, "
        f"{dropped.get('empty_adult_trouble', 0)} empty adult replies). "
        "A turn that is only xxx is kept, as \"(unintelligible)\". "
        "Most units still had other trouble points under the cap, so the empty-reply drops "
        "do not account for the whole gap. "
        "Error coding now counts every [*] on a word the child produced, including "
        "[* m:+ed] and the other subcodes, and a baseline turn cannot carry a %err tier."
    )
    return lines


def _recent_trouble_note(items: list[Item]) -> str:
    baseline = [item for item in items if item.trouble_class == "none"]
    usable = [item for item in baseline if not item.exclude]
    flagged = sum(1 for item in usable if item.recent_trouble)
    return (
        f"Usable baseline items with `recent_trouble` true: {flagged} of {len(usable)}. "
        "The flag is true when any of the previous three target-child turns in the "
        "ten-turn context has `xxx`, `yyy`, a `[*]` on a word the child produced, or "
        "a `[: target]` phonological form in the original CHAT. Other-child turns "
        "are not counted. Trouble items have the field empty."
    )


def write_report(
    path: Path,
    *,
    items: list[Item],
    sessions: list[Session],
    sampled_trouble: list[Item],
    dropped: dict[str, int],
    short_baseline: int,
    k: int,
    subsample_frac: float,
    output_path: Path,
    uncorrected: dict[str, int] | None,
    notes: list[str],
) -> None:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    current = defaultdict(int)
    for item in items:
        current[item.trouble_type] += 1
    flagged = sum(1 for item in items if item.target_in_context)
    lines = [
        "# Rendered items",
        "",
        f"Run at {now}. K = {k}. Subsample fraction = {subsample_frac}. "
        f"Seed string per sampling unit is `42:<unit>`.",
        "",
        "Each unit keeps up to K trouble points of each trouble type, then the same "
        "number of trouble-free child turns that are followed by an adult. "
        "Conti3 is capped per session (`unit = child_id|session_id`) and every Conti3 "
        "item has `in_group_contrast` false. Other items have `in_group_contrast` true. "
        "The subsample, when the fraction is below 1, keeps that fraction of each "
        "trouble type within a child and does not drop a child who had an item.",
        "",
        f"Output: `{output_path.relative_to(PROJECT_ROOT)}`.",
        "",
        _exclusion_note(items),
        "",
        "## Counts by corpus, group, and trouble type",
        "",
        *_count_table(items),
        "",
        "## Gold targets",
        "",
        "A gold target is the `[: target]` text on the child turn, excluding casual "
        "reductions. It is removed from the rendered turn. `target_in_context` flags "
        "items where that word already appears in an earlier context turn.",
        "",
        *_gold_table(items),
        "",
        f"Items flagged `target_in_context`: {flagged}. They are kept.",
        "",
        "## Recent trouble on baseline items",
        "",
        _recent_trouble_note(items),
        "",
        "## Difference from the previous build",
        "",
        *explain_previous(current, uncorrected, dropped, short_baseline),
        "",
        "## Children",
        "",
        *_child_table(sessions, sampled_trouble, items),
        "",
        "Feldman (33 controls) and Rescorla (25 controls) differ from the TalkBank "
        "index pages (50 and 22) because of what the archive contains, not because "
        "of the parser. The Feldman page's 50 is the number of rows in the control "
        "table: 57 files of 33 children. Rescorla's own file list has 24 controls "
        "with a 36-, 48-, or 60-month transcript, plus Briggs (`bri60`), whose row "
        "is marked as having no transcript. The index's 22 is not that list.",
        "",
        "The transcript column uses one ID per child. Feldman Don is the 1;4 session "
        "of D03. The two Gravina boys are `gr1` (Chris: `gr36b`, `gr248`, `gr160`) "
        "and `gr2` (Andrew: `gr36a`, `gr148`, `gr260`). The 36-month files have no "
        "names; `gr36b` shares SES 35 with Chris, and `gr36a` is the other twin. "
        "Those two files also disagree on maternal education (2 and 4).",
        "",
        "## Fifteen random rendered items",
        "",
        "Seed 42, drawn from the written items. These lines are rendered transcript "
        "text and must not be committed or redistributed.",
        "",
        *_format_preview(_preview(items)),
        "",
    ]
    if notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _totals_for(candidates: list[Item], child_of, k: int) -> dict[str, int]:
    """Resample with a different child-id function. Used only to explain the old total."""
    regrouped: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in candidates:
        child_id = child_of(item)
        unit = sampling_unit(item.corpus, child_id, item.session_id)
        regrouped[(item.corpus, unit)].append(item)
    totals: dict[str, int] = defaultdict(int)
    for (corpus, unit), pool in sorted(regrouped.items()):
        pool = sorted(pool, key=lambda item: item.item_id)
        chosen, _short = sample_unit(pool, unit_rng(unit), k)
        for item in chosen:
            totals[item.trouble_type] += 1
    return totals


def render(
    root: Path,
    *,
    k: int = K,
    subsample_frac: float = 1.0,
) -> dict:
    if not 0 <= subsample_frac <= 1:
        raise ValueError("subsample_frac must be between 0 and 1")
    sessions, notes = load_sessions(root)
    candidates, dropped = build_candidates(sessions)
    items, sample_info = sample_items(candidates, k=k, subsample_frac=subsample_frac)
    # Trouble items actually drawn, before the optional subsample removes some.
    full, _info = sample_items(candidates, k=k, subsample_frac=1.0)
    sampled_trouble = [item for item in full if item.trouble_type != "none"]
    uncorrected = None
    if subsample_frac == 1:
        parser_ids = {session.rel: session.parser_child_id for session in sessions}

        def old_child(item: Item) -> str:
            if item.corpus not in {"Feldman", "Rescorla"}:
                return item.child_id
            rel = item.item_id.rsplit(":", 1)[0]
            return parser_ids.get(rel, item.child_id)

        uncorrected = _totals_for(candidates, old_child, k)
    return {
        "sessions": sessions,
        "items": items,
        "sampled_trouble": sampled_trouble,
        "dropped": dropped,
        "short_baseline": sample_info["short_baseline"],
        "uncorrected": uncorrected,
        "notes": notes,
        "k": k,
        "subsample_frac": subsample_frac,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample and render listener items.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--subsample-frac", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = render(args.data_root, k=args.k, subsample_frac=args.subsample_frac)
    write_jsonl(args.output, result["items"])
    write_report(
        args.report,
        items=result["items"],
        sessions=result["sessions"],
        sampled_trouble=result["sampled_trouble"],
        dropped=result["dropped"],
        short_baseline=result["short_baseline"],
        k=result["k"],
        subsample_frac=result["subsample_frac"],
        output_path=args.output,
        uncorrected=result["uncorrected"],
        notes=result["notes"],
    )
    print(f"wrote {len(result['items'])} items to {args.output}")
    print(f"wrote {args.report}")
    print()
    for block in _format_preview(_preview(result["items"])):
        print(block)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
