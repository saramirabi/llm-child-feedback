#!/usr/bin/env python3
"""CHAT parser and trouble-point extractor for the in-scope Clinical-Eng corpora.

Raw .cha files are read and never written. Group labels come from the folder
path. Trouble is classified only on the main tier of a Target_Child turn, and
a trouble point is counted only when the next speaker turn is an adult.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SEED = 42
INCLUDE_CONTI1_UNRELATED_CONTROLS = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "childes_clinical_eng" / "Clinical-Eng"
)
DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "trouble_points_per_child.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "troublepoints.md"
DEFAULT_SAMPLE = PROJECT_ROOT / "reports" / "regex_check_sample.txt"

ADULT_ROLES = frozenset(
    {
        "Mother",
        "Father",
        "Investigator",
        "Grandmother",
        "Grandfather",
        "Adult",
        "Relative",
    }
)
# Siblings and other children are never listeners. These roles are also not
# the Target_Child whose turns are scored, except Target_Child itself.
NON_LISTENER_ROLES = frozenset(
    {
        "Target_Child",
        "Child",
        "Sibling",
        "Sister",
        "Brother",
    }
)
# Coded, but not an adult role and not a child role. Turns from these speakers
# do not count as the following listener. They are reported, not guessed.
AMBIGUOUS_ROLES = frozenset({"Participant", "Unidentified", "Speaker"})

CASUAL_REDUCTIONS = frozenset(
    {
        "dunno",
        "gonna",
        "wanna",
        "gotta",
        "hafta",
        "hasta",
        "lemme",
        "gimme",
        "kinda",
        "sorta",
        "outta",
        "lotsa",
        "dyou",
        "d'you",
        "whatcha",
        "gotcha",
        "dontcha",
        "coulda",
        "shoulda",
        "woulda",
        "oughta",
        "cuz",
        "'cause",
        "ya",
        "yeah",
        "yep",
        "nope",
        "em",
        "'em",
        "lookit",
        "c'mon",
        "y'know",
        "yknow",
    }
)

# Name table on https://talkbank.org/childes/access/Clinical-Eng/Feldman/ParentChild.html
# The page header says 56 target children; the table itself lists these 55 codes.
FELDMAN_PAGE_TARGET_CODES = frozenset(
    """
    BAR BEA BEY BLA BOT BRA CAL CAS CES CHA CIC CIO DAN DEB DOR DUP EDW FEI FLO
    FRI GAL GAM GAR GIG GLA GRA HAL HAP HAT HIN LEO LIN MAC MAG MAR MAT MEC MED
    MEY MYE NEA NOW PLA POP ROM SCO SHN SNO SUL TRU WAL WIX YUC YUZ ZIC
    """.split()
)

CORPUS_ORDER = (
    "Feldman",
    "Conti1",
    "Conti2",
    "Conti3",
    "EllisWeismer",
    "Rescorla",
)

BULLET_RE = re.compile(r"\x15[^\x15]*\x15")
TERMINATOR_RE = re.compile(
    r"(?:\+\.\.\.|\+//\.|\+//\?|\+/\.|\+/\?|\+\"/?\.?|[.?!])\s*$"
)
PAUSE_RE = re.compile(r"^(?:\(\.{1,3}\)|\(\d+(?:\.\d+)?\)|#)$")
AGE_RE = re.compile(r"^(\d+);(\d+)(?:\.(\d+))?$")
OMISSION_WORD_RE = re.compile(r"^0.+")
ID_RE = re.compile(r"^@ID:\s*(.*)$")
PART_RE = re.compile(r"^@Participants:\s*(.*)$")
SPEAKER_RE = re.compile(r"^\*([^:\s]+):(.*)$")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

CSV_FIELDS = [
    "row_type",
    "corpus",
    "child_id",
    "group",
    "n_files",
    "age_months_min",
    "age_months_max",
    "n_child_turns",
    "n_followed_by_adult",
    "n_trouble_points",
    "n_unintelligible",
    "n_error_word",
    "n_phonological_form",
    "n_baseline",
    "n_excluded_omission",
        "n_excluded_question",
        "n_excluded_other_error",
        "n_excluded_err_tier",
        "n_excluded_invisible",
        "n_gold_targets",
    ]


@dataclass
class Speaker:
    code: str
    name: str | None
    role: str
    age_raw: str
    age_months: float | None


@dataclass
class Turn:
    speaker: str
    role: str
    text: str
    lineno: int
    has_err_tier: bool = False
    err_text: str = ""


@dataclass
class ParsedChat:
    speakers: dict[str, Speaker]
    turns: list[Turn]
    warnings: list[str] = field(default_factory=list)

    def target_children(self) -> list[Speaker]:
        return [s for s in self.speakers.values() if s.role == "Target_Child"]


@dataclass
class Token:
    kind: str
    text: str


@dataclass
class Unit:
    kind: str
    form: str
    codes: list[str] = field(default_factory=list)


@dataclass
class Classification:
    trouble: str | None
    exclusion: str | None
    gold_targets: list[str]
    omission_word: bool = False
    omission_morpheme: bool = False
    unscoped_bare_star: bool = False
    odd_unintelligible: list[str] = field(default_factory=list)
    phonological_forms: list[str] = field(default_factory=list)
    baseline_contains_question: bool = False
    present_types: list[str] = field(default_factory=list)
    error_forms: list[str] = field(default_factory=list)
    bare_error_only: bool = False


@dataclass
class FileRecord:
    corpus: str
    path: Path
    rel: str
    child_id: str
    folder_group: str
    group: str
    age_months: float | None
    folder_age: float | None
    n_child_turns: int = 0
    n_followed_by_adult: int = 0
    n_trouble_points: int = 0
    n_unintelligible: int = 0
    n_error_word: int = 0
    n_phonological_form: int = 0
    n_baseline: int = 0
    n_excluded_omission: int = 0
    n_excluded_question: int = 0
    n_excluded_other_error: int = 0
    n_excluded_err_tier: int = 0
    n_excluded_invisible: int = 0
    n_gold_targets: int = 0
    n_gold_on_excluded: int = 0
    n_bare_zero_baseline: int = 0
    n_baseline_with_err: int = 0
    n_err_tier_not_trouble: int = 0
    n_baseline_with_question: int = 0
    n_odd_unintelligible: int = 0
    n_baseline_only_www: int = 0
    n_trouble_with_omission: int = 0
    n_unscoped_bare_star: int = 0
    phon_forms: Counter = field(default_factory=Counter)
    follower_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class SampleItem:
    corpus: str
    kind: str
    group: str
    child_id: str
    rel: str
    lineno: int
    text: str
    next_speaker: str
    next_role: str


@dataclass
class Extraction:
    records: list[FileRecord]
    skipped: list[tuple[str, str]]
    warnings: list[str]
    samples: dict[str, dict[str, list[SampleItem]]]
    ew_status: dict[str, str]
    ew_reassigned: list[str]
    ew_in_both: list[tuple[str, str]]
    ew_missing_from_sheet: list[str]
    ew_sheet_not_in_files: list[str]
    ew_status_disagreements: list[str]
    age_mismatches: list[str]
    no_target_child: list[str]
    include_conti1_unrelated_controls: bool


def age_in_months(age_raw: str) -> float | None:
    """Convert a CHAT @ID age (years;months.days) to months."""
    text = (age_raw or "").strip().rstrip(".")
    if not text:
        return None
    match = AGE_RE.fullmatch(text)
    if not match:
        return None
    years = int(match.group(1))
    months = int(match.group(2))
    days = int(match.group(3)) if match.group(3) else 0
    return round(years * 12 + months + days / 30.0, 2)


def _normalize_form(form: str) -> str:
    text = form.strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,?!\"")
    return text


def is_casual_reduction(form: str) -> bool:
    return _normalize_form(form) in CASUAL_REDUCTIONS


def _code_body(code: str) -> str:
    if code.startswith("[") and code.endswith("]"):
        return code[1:-1].strip()
    return code.strip()


def _is_bare_star(code: str) -> bool:
    return _code_body(code) == "*"


def _is_star(code: str) -> bool:
    return _code_body(code).startswith("*")


def _is_omission_subcode(code: str) -> bool:
    """A missing form written in the code itself, such as [* 0es] or [* 0's].

    [* m:0ing] and [* m:03s] are not this. The 0 there is part of a
    morphological feature on a word the child produced.
    """
    body = _code_body(code)
    if not body.startswith("*") or body == "*":
        return False
    return body[1:].strip().startswith("0")


def _is_morpheme_omission(code: str) -> bool:
    return _is_omission_subcode(code)


def _replacement_target(code: str) -> str | None:
    body = _code_body(code)
    if body.startswith("::") or not body.startswith(":"):
        return None
    return body[1:].strip()


def _is_omission_word(text: str) -> bool:
    return OMISSION_WORD_RE.match(text) is not None


def _lexical_base(word: str) -> str:
    return word.split("@", 1)[0]


def tokenize(utterance: str) -> list[Token]:
    text = BULLET_RE.sub(" ", utterance)
    text = text.replace("\x15", " ")
    text = TERMINATOR_RE.sub("", text.strip())
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        if text[i] == "<":
            end = text.find(">", i + 1)
            if end < 0:
                tokens.append(_word_token(text[i:]))
                break
            tokens.append(Token("group", text[i : end + 1]))
            i = end + 1
            continue
        if text[i] == "[":
            end = text.find("]", i + 1)
            if end < 0:
                tokens.append(_word_token(text[i:]))
                break
            tokens.append(Token("code", text[i : end + 1]))
            i = end + 1
            continue
        end = i
        while end < n and not text[end].isspace() and text[end] not in "<[":
            end += 1
        tokens.append(_word_token(text[i:end]))
        i = end
    return tokens


def _word_token(text: str) -> Token:
    if PAUSE_RE.match(text):
        return Token("pause", text)
    if text.startswith("&"):
        return Token("nonlex", text)
    if text == "0":
        return Token("bare_zero", text)
    if _is_omission_word(text):
        return Token("omission", text)
    return Token("word", text)


def _group_kind(form: str) -> str:
    inner = form[1:-1].strip()
    if not inner:
        return "group"
    first = inner.split()[0]
    if first == "0" or _is_omission_word(first):
        return "omission"
    return "group"


def segment(utterance: str) -> tuple[list[Unit], list[str]]:
    units: list[Unit] = []
    unscoped: list[str] = []
    current: Unit | None = None
    for token in tokenize(utterance):
        if token.kind == "pause":
            continue
        if token.kind == "code":
            if current is None:
                unscoped.append(token.text)
            else:
                current.codes.append(token.text)
            continue
        kind = _group_kind(token.text) if token.kind == "group" else token.kind
        current = Unit(kind=kind, form=token.text)
        units.append(current)
    return units, unscoped


def _flatten_for_codes(units: list[Unit], unscoped: list[str]) -> list[Unit]:
    """Open <groups> so a [*] inside the brackets is attached to its word.

    Codes written after the group stay on the last inner word. [: target]
    on the group is left for the phonological pass over the original units.
    """
    flat: list[Unit] = []
    for unit in units:
        if unit.kind != "group":
            flat.append(unit)
            continue
        inner, inner_unscoped = segment(unit.form[1:-1])
        unscoped.extend(inner_unscoped)
        star_codes = [code for code in unit.codes if _is_star(code)]
        if not inner:
            if star_codes:
                flat.append(Unit(kind=unit.kind, form=unit.form, codes=list(star_codes)))
            continue
        if star_codes:
            last = inner[-1]
            if last.kind in {"word", "group"}:
                last.codes = list(last.codes) + star_codes
            else:
                inner.append(Unit(kind="word", form=unit.form, codes=list(star_codes)))
        flat.extend(inner)
    return flat


def _shifted_bare(units: list[Unit]) -> set[int]:
    """Bare [*] marks that belong to a missing word, not to the word they follow.

    Forward, in every corpus: 0word with no [*] of its own, then the next bare
    [*] on a later produced word (`0is upstairs [*]`, `how 0are you doing [*]`).
    Backward: a bare [*] on the word immediately before a 0word
    (`who [*] 0is [*] this`).
    """
    shifted: set[int] = set()
    pending = False
    for index, unit in enumerate(units):
        if unit.kind == "omission" and not any(_is_star(code) for code in unit.codes):
            pending = True
            continue
        if (
            pending
            and unit.kind in {"word", "group"}
            and any(_is_bare_star(code) for code in unit.codes)
        ):
            shifted.add(index)
            pending = False
    for index, unit in enumerate(units[:-1]):
        nxt = units[index + 1]
        if (
            unit.kind in {"word", "group"}
            and nxt.kind == "omission"
            and any(_is_bare_star(code) for code in unit.codes)
        ):
            shifted.add(index)
    return shifted


def _star_is_omission_code(code: str, subcodes: bool) -> bool:
    if subcodes:
        return _is_omission_subcode(code)
    body = _code_body(code)
    return body.startswith("*") and body != "*" and "0" in body


def _produced_error(unit: Unit, shifted: bool, subcodes: bool) -> tuple[bool, bool, bool]:
    """Return (error_word, subcode error, other_error)."""
    saw_error = False
    saw_subcode = False
    saw_other = False
    for code in unit.codes:
        if not _is_star(code) or _star_is_omission_code(code, subcodes):
            continue
        if _is_bare_star(code) and shifted:
            continue
        if not subcodes and not _is_bare_star(code):
            saw_other = True
            continue
        saw_error = True
        if not _is_bare_star(code):
            saw_subcode = True
    return saw_error, saw_subcode, saw_other


@dataclass(frozen=True)
class CodingRules:
    """Switches used to measure each fix. Defaults are the rules used for the pilot."""

    subcodes: bool = True
    flatten: bool = True
    shift_omission: bool = True
    any_question: bool = True
    any_zero: bool = True
    clean_stars: bool = True


FINAL_RULES = CodingRules()
OLD_RULES = CodingRules(
    subcodes=False,
    flatten=False,
    shift_omission=False,
    any_question=False,
    any_zero=False,
    clean_stars=False,
)


def classify_utterance(utterance: str, rules: CodingRules | None = None) -> Classification:
    """Classify one child main-tier utterance.

    Precedence is unintelligible, then any [*] on a word the child produced,
    then a [: target] with no [*] code. A [*] that marks a missing word is an
    omission in every corpus, including when it was written on a neighboring
    word. Casual reductions are not phonological forms. Any [?] keeps the
    turn out of the baseline.
    """
    rules = rules or FINAL_RULES
    units, unscoped = segment(utterance)
    flat = _flatten_for_codes(units, unscoped) if rules.flatten else list(units)
    shifted = _shifted_bare(flat) if rules.shift_omission else set()
    odd: list[str] = []
    for unit in flat:
        if unit.kind != "word":
            continue
        base = _lexical_base(unit.form).lower()
        if base in {"xx", "www"}:
            odd.append(base)

    def produced(unit: Unit) -> bool:
        return unit.kind in {"word", "group"}

    has_unintelligible = False
    has_error = False
    has_subcode_error = False
    has_other = False
    has_phon = False
    has_word_omission = False
    has_morph_omission = False
    gold: list[str] = []
    phon_forms: list[str] = []
    error_forms: list[str] = []
    code_bodies: list[str] = list(unscoped)

    for code in unscoped:
        if _star_is_omission_code(code, rules.subcodes):
            has_morph_omission = True
        elif _is_star(code) and any(produced(unit) for unit in flat):
            if rules.subcodes or _is_bare_star(code):
                has_error = True
                if not _is_bare_star(code):
                    has_subcode_error = True
            else:
                has_other = True

    unscoped_bare = any(_is_bare_star(code) for code in unscoped) and has_error

    for index, unit in enumerate(flat):
        for code in unit.codes:
            code_bodies.append(code)
            if _star_is_omission_code(code, rules.subcodes):
                has_morph_omission = True
        base = _lexical_base(unit.form).lower() if unit.kind == "word" else ""
        if unit.kind == "word" and base in {"xxx", "yyy"}:
            has_unintelligible = True
        if unit.kind == "omission" and (
            rules.any_zero or any(_is_star(code) for code in unit.codes)
        ):
            has_word_omission = True
        if not produced(unit):
            if not rules.subcodes:
                for code in unit.codes:
                    if (
                        _is_star(code)
                        and not _is_bare_star(code)
                        and not _star_is_omission_code(code, False)
                    ):
                        has_other = True
            continue
        is_error, is_subcode, is_other = _produced_error(unit, index in shifted, rules.subcodes)
        if is_other:
            has_other = True
        if is_error:
            has_error = True
            error_forms.append(unit.form)
            if is_subcode:
                has_subcode_error = True
        if index in shifted:
            has_word_omission = True

    for unit in units:
        code_bodies.extend(unit.codes)
        if not produced(unit):
            continue
        any_star = any(_is_star(code) for code in unit.codes)
        target = next(
            (
                found
                for code in unit.codes
                if (found := _replacement_target(code)) is not None
            ),
            None,
        )
        casual = is_casual_reduction(unit.form)
        if target and not casual:
            gold.append(target)
        if target and not any_star and not casual:
            has_phon = True
            phon_forms.append(_normalize_form(unit.form))
        if unit.kind == "group":
            for piece in unit.form[1:-1].split():
                if _lexical_base(piece).lower() in {"xxx", "yyy"}:
                    has_unintelligible = True

    if has_unintelligible:
        trouble = "unintelligible"
    elif has_error:
        trouble = "error_word"
    elif has_phon:
        trouble = "phonological_form"
    else:
        trouble = None

    has_question = any(_code_body(code) == "?" for code in code_bodies)
    only_question = bool(code_bodies) and all(_code_body(code) == "?" for code in code_bodies)
    if trouble:
        exclusion = None
    elif has_word_omission or has_morph_omission:
        exclusion = "omission"
    elif has_question if rules.any_question else only_question:
        exclusion = "question"
    elif has_other:
        exclusion = "other_error"
    elif (
        rules.clean_stars
        and any(_is_star(code) for code in code_bodies)
    ):
        exclusion = "other_error"
    else:
        exclusion = None

    present: list[str] = []
    if has_unintelligible:
        present.append("unintelligible")
    if has_error:
        present.append("error_word")
    if has_phon:
        present.append("phonological_form")

    return Classification(
        trouble=trouble,
        exclusion=exclusion,
        gold_targets=gold,
        omission_word=has_word_omission,
        omission_morpheme=has_morph_omission,
        unscoped_bare_star=unscoped_bare,
        odd_unintelligible=odd,
        phonological_forms=phon_forms,
        baseline_contains_question=trouble is None
        and exclusion is None
        and has_question,
        present_types=present,
        error_forms=error_forms,
        bare_error_only=has_error and not has_subcode_error and not gold,
    )


_DYSFLUENCY_PART_RE = re.compile(r"^(?:DYS\b|disfluent\b)", re.IGNORECASE)


def _only_dysfluency(err_text: str) -> bool:
    """True when every %err code is a dysfluency, not a word or grammar error."""
    parts = [part.strip() for part in re.split(r"[,|]", err_text or "") if part.strip()]
    return bool(parts) and all(_DYSFLUENCY_PART_RE.match(part) for part in parts)


def visibility_failure(
    utterance: str,
    classified: Classification,
    err_text: str = "",
) -> str | None:
    """Why an error or phonological item does not show the wrong word.

    Returns None when the listener string still contains that word.
    """
    if classified.trouble not in {"error_word", "phonological_form"}:
        return None
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.render import render_utterance

    rendered = render_utterance(utterance)

    def missing_spoken(forms: list[str]) -> tuple[list[str], bool]:
        """Forms that render as speech and are absent from the listener string.

        A form that renders empty is not a produced word: it is an omitted
        0-token, or a bracket left when a group was closed at the first `>`.
        Those forms are not something the listener string can hide.
        """
        missing: list[str] = []
        heard = False
        for form in forms:
            span = render_utterance(form)
            if not span:
                continue
            heard = True
            if not _span_in_render(span, rendered):
                missing.append(form)
        return missing, heard

    if classified.trouble == "error_word":
        forms = classified.error_forms
        if not forms:
            return "no produced word carries the [*] into the listener string"
        missing, heard = missing_spoken(forms)
        if missing:
            shown = ", ".join(missing)
            return f"rendered turn does not contain the marked word ({shown})"
        if not heard:
            return None
        if classified.bare_error_only and _only_dysfluency(err_text):
            return (
                f"the [*] marks a dysfluency ({err_text}); the repeated or broken "
                "material is not written out, so the listener string only contains the host word"
            )
        return None
    missing, _heard = missing_spoken(classified.phonological_forms)
    if missing:
        shown = ", ".join(missing)
        return f"rendered turn does not contain the produced form ({shown})"
    return None


def _span_in_render(span: str, rendered: str) -> bool:
    span = re.sub(r"\s+", " ", (span or "")).strip().lower()
    if not span:
        return False
    lowered = rendered.lower()
    if re.search(r"(?:^|\s)" + re.escape(span) + r"(?:\s|$)", lowered):
        return True
    # that (i)s renders as "that's", so the host and the clitic are one token.
    tokens = lowered.split()
    if span in {"'s", "'m"}:
        return any(token.endswith(span) for token in tokens)
    return any(token == f"{span}'s" or token == f"{span}'m" for token in tokens)


def parse_chat(text: str) -> ParsedChat:
    """Join main-tier continuation lines and drop dependent tiers."""
    speakers: dict[str, Speaker] = {}
    names: dict[str, str | None] = {}
    roles_from_participants: dict[str, str] = {}
    turns: list[Turn] = []
    warnings: list[str] = []
    current: dict | None = None
    open_tier = None

    def close() -> None:
        nonlocal current, open_tier
        if current is None:
            return
        code = current["speaker"]
        speaker = speakers.get(code)
        role = speaker.role if speaker is not None else roles_from_participants.get(code, "")
        if speaker is None and code not in roles_from_participants:
            warnings.append(f"speaker {code} has no @ID or @Participants role")
            role = "Unknown"
        turns.append(
            Turn(
                speaker=code,
                role=role or "Unknown",
                text=re.sub(r"\s+", " ", " ".join(current["parts"])).strip(),
                lineno=current["lineno"],
                has_err_tier=current["has_err"],
                err_text=current.get("err_text", ""),
            )
        )
        current = None
        open_tier = None

    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if line[0] in " \t":
            if current is not None and open_tier == "dependent" and current.get("err_open"):
                current["err_text"] = f"{current['err_text']} {line.strip()}".strip()
                continue
            if current is None or open_tier != "main":
                continue
            current["parts"].append(line.strip())
            continue
        if line.startswith("@"):
            close()
            part = PART_RE.match(line)
            if part:
                for code, name, role in _parse_participants(part.group(1)):
                    names[code] = name
                    roles_from_participants[code] = role
                    if code in speakers and name:
                        speakers[code].name = name
                continue
            id_match = ID_RE.match(line)
            if id_match:
                speaker = _parse_id(id_match.group(1), names)
                if speaker is None:
                    warnings.append(f"unparsed @ID at line {lineno}")
                else:
                    if speaker.code in names and names[speaker.code]:
                        speaker.name = names[speaker.code]
                    speakers[speaker.code] = speaker
            continue
        if line.startswith("%"):
            # A %err tier still belongs to this utterance when %mor or %wor
            # already moved the open tier off the main line.
            if current is not None:
                tier = line[1:].split(":", 1)[0].strip().lower()
                if tier == "err":
                    current["has_err"] = True
                    current["err_open"] = True
                    rest = line.split(":", 1)[1].strip() if ":" in line else ""
                    if current["err_text"]:
                        current["err_text"] = f"{current['err_text']} | {rest}"
                    else:
                        current["err_text"] = rest
                else:
                    current["err_open"] = False
            open_tier = "dependent"
            continue
        speaker_match = SPEAKER_RE.match(line)
        if speaker_match:
            close()
            current = {
                "speaker": speaker_match.group(1),
                "parts": [speaker_match.group(2).strip()],
                "lineno": lineno,
                "has_err": False,
                "err_text": "",
                "err_open": False,
            }
            open_tier = "main"
            continue
        warnings.append(f"orphan line {lineno}")
    close()
    return ParsedChat(speakers=speakers, turns=turns, warnings=warnings)


def _parse_participants(value: str) -> list[tuple[str, str | None, str]]:
    found = []
    for chunk in value.split(","):
        bits = chunk.split()
        if not bits:
            continue
        code = bits[0]
        if len(bits) == 1:
            found.append((code, None, ""))
        elif len(bits) == 2:
            found.append((code, None, bits[1]))
        else:
            found.append((code, " ".join(bits[1:-1]), bits[-1]))
    return found


def _parse_id(value: str, names: dict[str, str | None]) -> Speaker | None:
    bits = value.split("|")
    if len(bits) < 8:
        return None
    code = bits[2].strip()
    if not code:
        return None
    age_raw = bits[3].strip()
    role = bits[7].strip()
    return Speaker(
        code=code,
        name=names.get(code),
        role=role,
        age_raw=age_raw,
        age_months=age_in_months(age_raw),
    )


def read_chat(path: Path) -> tuple[str, bool]:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8"), False
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), True


def digit_age_folder(path: Path, stop: Path) -> str | None:
    for parent in path.parents:
        if parent.name.isdigit():
            return parent.name
        if parent == stop:
            break
    return None


def rescorla_child_id(stem: str, age_folder: str | None) -> str:
    """Strip the month-folder digits. Keep a trailing letter on the full stem.

    ale36 in folder 36 is ale. ly236 in folder 36 is ly2. gr36a in folder 36
    stays gr36a so the two 36-month files are not collapsed.
    """
    lowered = stem.lower()
    if not age_folder:
        return lowered
    match = re.fullmatch(rf"(.+?){re.escape(age_folder)}([a-z]?)", lowered)
    if not match:
        return lowered
    prefix, letter = match.group(1), match.group(2)
    if letter:
        return lowered
    return prefix


def load_ellisweismer_status(path: Path) -> dict[str, str]:
    """Read Status from 0status.xlsx. 1 = late talker, 0 = typical."""
    status: dict[str, str] = {}
    with zipfile.ZipFile(path) as workbook:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", NS):
                strings.append("".join(node.text or "" for node in item.findall(".//m:t", NS)))
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        for row in sheet.findall("m:sheetData/m:row", NS):
            values: dict[str, str] = {}
            for cell in row.findall("m:c", NS):
                ref = cell.get("r") or ""
                column = "".join(char for char in ref if char.isalpha())
                node = cell.find("m:v", NS)
                if node is None or node.text is None:
                    continue
                if cell.get("t") == "s":
                    values[column] = strings[int(node.text)]
                else:
                    values[column] = node.text
            child_id = values.get("A", "").strip()
            code = values.get("C", "").strip()
            if child_id.isdigit() and code in {"0", "1"}:
                status[child_id] = code
    return status


def _corpus_files(root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    specs = [
        ("Feldman", root / "Feldman" / "ParentChild"),
        ("Conti1", root / "Conti" / "Conti1"),
        ("Conti2", root / "Conti" / "Conti2"),
        ("Conti3", root / "Conti" / "Conti3"),
        ("EllisWeismer", root / "EllisWeismer"),
        ("Rescorla", root / "Rescorla"),
    ]
    for corpus, folder in specs:
        if not folder.exists():
            continue
        if corpus == "EllisWeismer":
            paths = list((folder / "LT").rglob("*.cha")) + list((folder / "TD").rglob("*.cha"))
        else:
            paths = list(folder.rglob("*.cha"))
        for path in paths:
            if path.name.startswith("._") or "__MACOSX" in path.parts:
                continue
            found.append((corpus, path))
    return found


def _skip_reason(corpus: str, path: Path, include_unrelated: bool) -> str | None:
    if corpus == "Conti1" and "0control" in path.parts:
        if path.name.startswith("concon"):
            return "Conti1 concon file (child-sibling play, no adult); always dropped"
        if path.name.startswith(("fatcon", "motcon")) and not include_unrelated:
            return "Conti1 unrelated control; INCLUDE_CONTI1_UNRELATED_CONTROLS is off"
    if corpus == "Rescorla" and any(part in {"108", "156"} for part in path.parts):
        return "Rescorla 108- or 156-month folder; dropped"
    return None


def _folder_group(corpus: str, path: Path) -> str:
    parts = path.parts
    if corpus == "Feldman":
        if "SLI" in parts:
            return "focal_lesion"
        if "TD" in parts:
            return "td"
    if corpus == "Conti1":
        if any(part in parts for part in ("fatsli", "motsli", "slisib")):
            return "dld"
        if any(part in parts for part in ("fatsib", "motsib")):
            return "td_sibling"
        if "0control" in parts:
            return "td_unrelated"
    if corpus == "Conti2":
        if "SLI" in parts:
            return "dld"
        if "TD" in parts:
            return "td_sibling"
    if corpus == "Conti3":
        return "dld"
    if corpus == "EllisWeismer":
        if "LT" in parts:
            return "late_talker"
        if "TD" in parts:
            return "td"
    if corpus == "Rescorla":
        if "LT" in parts:
            return "late_talker"
        if "TD" in parts:
            return "td"
    raise ValueError(f"no group for {corpus} {path}")


def _child_id(corpus: str, path: Path, parsed: ParsedChat, age_folder: str | None) -> str | None:
    targets = parsed.target_children()
    if corpus == "Feldman":
        if not targets:
            return None
        name = targets[0].name
        if name:
            return name.upper()
        return path.stem.upper()
    if corpus == "Conti1":
        if not targets or not targets[0].name:
            return None
        return targets[0].name
    if corpus == "Conti2":
        return path.parent.name
    if corpus == "Conti3":
        return path.parent.name
    if corpus == "EllisWeismer":
        return path.stem
    if corpus == "Rescorla":
        return rescorla_child_id(path.stem, age_folder)
    return None


def _is_bare_zero_turn(text: str) -> bool:
    units, unscoped = segment(text)
    return not unscoped and len(units) == 1 and units[0].kind == "bare_zero"


def _is_only_www(text: str) -> bool:
    """True when the utterance is only untranscribed www, not xxx/yyy."""
    units, unscoped = segment(text)
    if unscoped:
        return False
    lexical = [unit for unit in units if unit.kind != "pause"]
    return bool(lexical) and all(
        unit.kind == "word"
        and not unit.codes
        and _lexical_base(unit.form).lower() == "www"
        for unit in lexical
    )


def _header_notes(text: str) -> str:
    notes: list[str] = []
    for line in text.splitlines():
        if line.startswith("*"):
            break
        if line.startswith("@Warning:") or line.startswith("@Comment:"):
            notes.append(line.split(":", 1)[1].strip())
    if not notes:
        return ""
    return " (" + "; ".join(notes[:3]) + ")"


def extract(
    root: Path,
    *,
    include_conti1_unrelated_controls: bool = INCLUDE_CONTI1_UNRELATED_CONTROLS,
    ellisweismer_status: dict[str, str] | None = None,
) -> Extraction:
    if ellisweismer_status is None:
        status_path = root / "EllisWeismer" / "0status.xlsx"
        ellisweismer_status = load_ellisweismer_status(status_path) if status_path.exists() else {}

    records: list[FileRecord] = []
    skipped: list[tuple[str, str]] = []
    warnings: list[str] = []
    samples: dict[str, dict[str, list[SampleItem]]] = defaultdict(lambda: defaultdict(list))
    no_target: list[str] = []
    age_mismatches: list[str] = []

    for corpus, path in _corpus_files(root):
        rel = str(path.relative_to(root))
        reason = _skip_reason(corpus, path, include_conti1_unrelated_controls)
        if reason:
            skipped.append((rel, reason))
            continue
        text, replaced = read_chat(path)
        if replaced:
            warnings.append(f"{rel}: file was not valid UTF-8; replacement characters were used")
        parsed = parse_chat(text)
        for warning in parsed.warnings:
            warnings.append(f"{rel}: {warning}")
        targets = parsed.target_children()
        if len(targets) != 1:
            no_target.append(f"{rel}: {len(targets)} Target_Child speakers{_header_notes(text)}")
            continue
        age_folder = digit_age_folder(path, root)
        child_id = _child_id(corpus, path, parsed, age_folder)
        if not child_id:
            no_target.append(f"{rel}: Target_Child has no usable child_id")
            continue
        header_age = targets[0].age_months
        folder_age = float(age_folder) if age_folder else None
        if header_age is None:
            age = folder_age
        else:
            age = header_age
            if folder_age is not None and abs(header_age - folder_age) > 1:
                age_mismatches.append(
                    f"{rel}: @ID age {header_age} months, folder {age_folder}"
                )
        record = FileRecord(
            corpus=corpus,
            path=path,
            rel=rel,
            child_id=child_id,
            folder_group=_folder_group(corpus, path),
            group=_folder_group(corpus, path),
            age_months=age,
            folder_age=folder_age,
        )
        _score_file(record, parsed, samples)
        records.append(record)

    ew_reassigned, ew_in_both, missing_sheet, sheet_only, disagreements = _resolve_ellisweismer(
        records, ellisweismer_status
    )
    _sync_sample_groups(records, samples)
    return Extraction(
        records=records,
        skipped=skipped,
        warnings=warnings,
        samples=samples,
        ew_status=ellisweismer_status,
        ew_reassigned=ew_reassigned,
        ew_in_both=ew_in_both,
        ew_missing_from_sheet=missing_sheet,
        ew_sheet_not_in_files=sheet_only,
        ew_status_disagreements=disagreements,
        age_mismatches=age_mismatches,
        no_target_child=no_target,
        include_conti1_unrelated_controls=include_conti1_unrelated_controls,
    )


def _score_file(
    record: FileRecord,
    parsed: ParsedChat,
    samples: dict[str, dict[str, list[SampleItem]]],
) -> None:
    turns = parsed.turns
    for index, turn in enumerate(turns):
        if turn.role != "Target_Child":
            continue
        record.n_child_turns += 1
        nxt = turns[index + 1] if index + 1 < len(turns) else None
        if nxt is None:
            follower = "end"
        else:
            follower = nxt.role or "Unknown"
        record.follower_counts[follower] = record.follower_counts.get(follower, 0) + 1
        if follower not in ADULT_ROLES:
            continue
        record.n_followed_by_adult += 1
        classified = classify_utterance(turn.text)
        invisible = visibility_failure(turn.text, classified, turn.err_text)
        if classified.odd_unintelligible:
            record.n_odd_unintelligible += 1
        if classified.unscoped_bare_star:
            record.n_unscoped_bare_star += 1
        if invisible:
            record.n_excluded_invisible += 1
            record.n_gold_on_excluded += len(classified.gold_targets)
            item = SampleItem(
                corpus=record.corpus,
                kind="invisible",
                group=record.group,
                child_id=record.child_id,
                rel=record.rel,
                lineno=turn.lineno,
                text=turn.text,
                next_speaker=nxt.speaker if nxt else "",
                next_role=follower,
            )
            samples[record.corpus]["invisible"].append(item)
            continue
        item = SampleItem(
            corpus=record.corpus,
            kind=classified.trouble or classified.exclusion or "baseline",
            group=record.group,
            child_id=record.child_id,
            rel=record.rel,
            lineno=turn.lineno,
            text=turn.text,
            next_speaker=nxt.speaker if nxt else "",
            next_role=follower,
        )
        if classified.trouble:
            record.n_trouble_points += 1
            if classified.trouble == "unintelligible":
                record.n_unintelligible += 1
            elif classified.trouble == "error_word":
                record.n_error_word += 1
            elif classified.trouble == "phonological_form":
                record.n_phonological_form += 1
                record.phon_forms.update(classified.phonological_forms)
            if classified.omission_word or classified.omission_morpheme:
                record.n_trouble_with_omission += 1
            record.n_gold_targets += len(classified.gold_targets)
            samples[record.corpus][classified.trouble].append(item)
        elif classified.exclusion == "omission":
            record.n_excluded_omission += 1
            record.n_gold_on_excluded += len(classified.gold_targets)
            samples[record.corpus]["omission"].append(item)
        elif classified.exclusion == "question":
            record.n_excluded_question += 1
            samples[record.corpus]["question"].append(item)
        elif classified.exclusion == "other_error":
            record.n_excluded_other_error += 1
            record.n_gold_on_excluded += len(classified.gold_targets)
            samples[record.corpus]["other_error"].append(item)
        elif turn.has_err_tier:
            record.n_excluded_err_tier += 1
            samples[record.corpus]["err_tier"].append(item)
        else:
            record.n_baseline += 1
            record.n_gold_targets += len(classified.gold_targets)
            if _is_bare_zero_turn(turn.text):
                record.n_bare_zero_baseline += 1
            if _is_only_www(turn.text):
                record.n_baseline_only_www += 1
            if turn.has_err_tier:
                record.n_baseline_with_err += 1
            if classified.baseline_contains_question:
                record.n_baseline_with_question += 1
        if turn.has_err_tier and classified.trouble is None:
            record.n_err_tier_not_trouble += 1


def _resolve_ellisweismer(
    records: list[FileRecord], status: dict[str, str]
) -> tuple[list[str], list[tuple[str, str]], list[str], list[str], list[str]]:
    by_child: dict[str, list[FileRecord]] = defaultdict(list)
    for record in records:
        if record.corpus == "EllisWeismer":
            by_child[record.child_id].append(record)
    reassigned: list[str] = []
    in_both: list[tuple[str, str]] = []
    disagreements: list[str] = []
    for child_id, rows in sorted(by_child.items()):
        folders = sorted({row.folder_group for row in rows})
        sheet = status.get(child_id)
        if len(folders) > 1:
            if sheet == "1":
                resolved = "late_talker"
            elif sheet == "0":
                resolved = "td"
            else:
                resolved = ""
            in_both.append((child_id, sheet or "missing"))
            if not resolved:
                for row in rows:
                    reassigned.append(
                        f"{row.rel}: child {child_id} is in both LT and TD and has no Status; excluded"
                    )
                records[:] = [row for row in records if row not in rows]
                continue
            for row in rows:
                if row.group != resolved:
                    reassigned.append(
                        f"{row.rel}: folder {row.folder_group} reassigned to {resolved} "
                        f"because 0status.xlsx Status is {sheet}"
                    )
                    row.group = resolved
        elif sheet is not None:
            expected = "late_talker" if sheet == "1" else "td"
            if folders and folders[0] != expected:
                disagreements.append(
                    f"{child_id}: folder {folders[0]} but Status {sheet} ({expected}); folder kept"
                )
    file_ids = set(by_child)
    # Children removed above are still in by_child. Compare against original file ids
    # plus any id that was only on the sheet.
    missing_from_sheet = sorted(file_ids - set(status))
    sheet_not_in_files = sorted(set(status) - file_ids)
    return reassigned, in_both, missing_from_sheet, sheet_not_in_files, disagreements


def _sync_sample_groups(
    records: list[FileRecord],
    samples: dict[str, dict[str, list[SampleItem]]],
) -> None:
    groups = {row.rel: row.group for row in records}
    for kinds in samples.values():
        for kind, items in list(kinds.items()):
            kept: list[SampleItem] = []
            for item in items:
                group = groups.get(item.rel)
                if group is None:
                    continue
                item.group = group
                kept.append(item)
            kinds[kind] = kept


def _sum_records(rows: list[FileRecord]) -> dict[str, int | float | None]:
    ages = [row.age_months for row in rows if row.age_months is not None]
    totals: dict[str, int | float | None] = {
        "n_files": len(rows),
        "age_months_min": min(ages) if ages else None,
        "age_months_max": max(ages) if ages else None,
    }
    for name in (
        "n_child_turns",
        "n_followed_by_adult",
        "n_trouble_points",
        "n_unintelligible",
        "n_error_word",
        "n_phonological_form",
        "n_baseline",
        "n_excluded_omission",
        "n_excluded_question",
        "n_excluded_other_error",
        "n_excluded_err_tier",
        "n_excluded_invisible",
        "n_gold_targets",
    ):
        totals[name] = sum(getattr(row, name) for row in rows)
    return totals


def child_rows(extraction: Extraction) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[FileRecord]] = defaultdict(list)
    for record in extraction.records:
        grouped[(record.corpus, record.child_id, record.group)].append(record)
    rows: list[dict[str, str]] = []
    keys = sorted(grouped, key=lambda key: (CORPUS_ORDER.index(key[0]), key[2], key[1]))
    by_corpus: dict[str, list[FileRecord]] = defaultdict(list)
    for corpus, child_id, group in keys:
        members = grouped[(corpus, child_id, group)]
        by_corpus[corpus].extend(members)
        totals = _sum_records(members)
        rows.append(_csv_row("child", corpus, child_id, group, totals))
    for corpus in CORPUS_ORDER:
        if corpus not in by_corpus:
            continue
        rows.append(_csv_row("corpus", corpus, "", "", _sum_records(by_corpus[corpus])))
    return rows


def _fmt_age(value: int | float | None) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text


def _csv_row(
    row_type: str,
    corpus: str,
    child_id: str,
    group: str,
    totals: dict[str, int | float | None],
) -> dict[str, str]:
    row = {
        "row_type": row_type,
        "corpus": corpus,
        "child_id": child_id,
        "group": group,
    }
    for field_name in CSV_FIELDS[4:]:
        value = totals[field_name]
        if field_name.startswith("age_"):
            row[field_name] = _fmt_age(value)
        else:
            row[field_name] = str(value)
    return row


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_matches(
    extraction: Extraction, per_corpus: int = 20, seed: int = SEED
) -> dict[str, list[SampleItem]]:
    rng = random.Random(seed)
    chosen: dict[str, list[SampleItem]] = {}
    types = ("unintelligible", "error_word", "phonological_form")
    for corpus in CORPUS_ORDER:
        pools: dict[str, list[SampleItem]] = {}
        for kind in types:
            items = sorted(
                extraction.samples.get(corpus, {}).get(kind, []),
                key=lambda item: (item.rel, item.lineno, item.text),
            )
            rng.shuffle(items)
            pools[kind] = items
        picked: list[SampleItem] = []
        while len(picked) < per_corpus and any(pools.values()):
            for kind in types:
                if pools[kind] and len(picked) < per_corpus:
                    picked.append(pools[kind].pop())
        chosen[corpus] = picked
    return chosen


def write_sample(path: Path, extraction: Extraction, seed: int = SEED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = sample_matches(extraction, seed=seed)
    rng = random.Random(seed)
    lines = [
        "Trouble-point regex sample",
        f"Seed: {seed}",
        "Up to 20 counted trouble points per corpus, round-robin across",
        "unintelligible, error_word, and phonological_form.",
        "These lines quote TalkBank transcripts and must not be committed or redistributed.",
        "",
    ]
    for corpus in CORPUS_ORDER:
        items = chosen.get(corpus, [])
        available = sum(
            len(extraction.samples.get(corpus, {}).get(kind, []))
            for kind in ("unintelligible", "error_word", "phonological_form")
        )
        lines.append(f"## {corpus}")
        lines.append(f"{len(items)} shown of {available} counted trouble points.")
        lines.append("")
        for index, item in enumerate(items, start=1):
            lines.append(
                f"[{index}] {item.kind} group={item.group} child={item.child_id} "
                f"{item.rel}:{item.lineno}"
            )
            lines.append(f"    CHI: {_display_utterance(item.text)}")
            lines.append(f"    next: {item.next_speaker} {item.next_role}")
            lines.append("")
        lines.append("Exclusion examples (not part of the 20):")
        for kind in ("omission", "question", "other_error"):
            pool = sorted(
                extraction.samples.get(corpus, {}).get(kind, []),
                key=lambda item: (item.rel, item.lineno, item.text),
            )
            rng.shuffle(pool)
            lines.append(f"  {kind}: {len(pool)} counted turns; showing {min(5, len(pool))}")
            for item in pool[:5]:
                lines.append(f"    {item.rel}:{item.lineno} child={item.child_id}")
                lines.append(f"    CHI: {_display_utterance(item.text)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _display_utterance(text: str) -> str:
    cleaned = BULLET_RE.sub(" ", text).replace("\x15", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _corpus_records(extraction: Extraction, corpus: str) -> list[FileRecord]:
    return [row for row in extraction.records if row.corpus == corpus]


def _children(rows: list[FileRecord]) -> dict[tuple[str, str], list[FileRecord]]:
    grouped: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    for row in rows:
        grouped[(row.child_id, row.group)].append(row)
    return grouped


def write_report(path: Path, extraction: Extraction, csv_path: Path, sample_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines: list[str] = []
    add = lines.append
    add("# Trouble-point extraction")
    add("")
    add(f"Run at {now}. Seed {SEED} for the regex sample.")
    add("")
    add("## What was done")
    add("")
    add(
        "Parsed the in-scope CHAT transcripts under `data/raw/childes_clinical_eng/Clinical-Eng/`. "
        "Main-tier continuation lines were joined. Dependent tiers (`%mor`, `%gra`, `%err`, `%act`, and the rest) were not used to decide trouble. "
        "Raw `.cha` files were not modified."
    )
    add("")
    add(
        "A child turn is a main tier whose speaker role is `Target_Child`. "
        "It is a trouble point only when the next speaker turn has an adult role: "
        + ", ".join(sorted(ADULT_ROLES))
        + ". Siblings and other children (`Sibling`, `Sister`, `Brother`, `Child`) are never listeners. "
        "`Participant`, `Unidentified`, and `Speaker` are not treated as listeners."
    )
    add("")
    add(
        "Trouble precedence on the child main tier is unintelligible (`xxx` or `yyy`), "
        "then `error_word` (any `[*]` on a word the child produced, including morphological, "
        "semantic, and phonological subcodes such as `[* m:+ed]`, `[* m:03s]`, `[* m:0ing]`, "
        "`[* s:...]`, and `[* p:...]`), "
        "then `phonological_form` (a word or `<group>` with `[: target]` and no `[*` code), "
        "excluding the listed casual reductions. "
        "A `[*]` that marks a missing word is an omission in every corpus: `0word` itself, "
        "a `[* 0...]` subcode, and a bare `[*]` written on a neighboring word "
        "(`0is upstairs [*]`, `how 0are you doing [*]`, `who [*] 0is [*]`). "
        "Those turns are in neither the trouble set nor the baseline. "
        "Any `[?]` keeps a turn out of the baseline, and so does a `%err` tier. "
        "An error or phonological turn whose wrong word is not in the listener string is excluded."
    )
    add("")
    add(
        f"`INCLUDE_CONTI1_UNRELATED_CONTROLS` was "
        f"{'on' if extraction.include_conti1_unrelated_controls else 'off'} "
        "for this run (default off). `concon*` files were dropped. Rescorla folders named 108 and 156 were dropped."
    )
    add("")
    add("## How child_id is derived")
    add("")
    add("| Corpus | child_id | Group |")
    add("| --- | --- | --- |")
    add(
        "| Feldman/ParentChild | Uppercase Target_Child name in `@Participants`. "
        "If the header is only `CHI Target_Child`, the filename stem in uppercase. | "
        "Folder `SLI` = `focal_lesion` (never `dld`). Folder `TD` = `td`. |"
    )
    add(
        "| Conti1 | Target_Child name in `@Participants` (`Clay`, `Jack`, …). "
        "The same child is one row across mother, father, and `slisib` files. | "
        "`fatsli`, `motsli`, and `slisib` = `dld`. `fatsib` and `motsib` = `td_sibling`. "
        "`fatcon*` / `motcon*` = `td_unrelated` only when the flag is on. |"
    )
    add("| Conti2 | Child folder name (`Andrew`, `Adam`, …). | `SLI` = `dld`. `TD` = `td_sibling`, not `td`. |")
    add("| Conti3 | Child folder name. | `dld`. No controls. |")
    add(
        "| EllisWeismer | Filename stem, the numeric ID (`11005`). "
        "`Phon/` is ignored. | Folder `LT` = `late_talker`, `TD` = `td`. "
        "If an ID is filed in both folders, `0status.xlsx` wins: Status 1 = `late_talker`, Status 0 = `td`. |"
    )
    add(
        "| Rescorla | Filename with the age-folder digits removed (`ale36` → `ale`, `ly236` → `ly2`). "
        "A letter after those digits keeps the whole stem (`gr36a`). | "
        "Folder `LT` = `late_talker`, `TD` = `td`. Age is the `@ID` age; if that is empty, the all-digit parent folder (already in months). |"
    )
    add("")
    add("## Counts")
    add("")
    add(
        "Per-child rows and one corpus total per corpus are in "
        f"`{csv_path.relative_to(PROJECT_ROOT)}`. "
        "A trouble point is a child turn of one of the three types that is directly followed by an adult."
    )
    add("")
    add(
        "| Corpus | Children | Files | Trouble | Unintelligible | Error word | Phonological | Baseline | Omission excluded | [?] excluded | Other error excluded | Gold targets |"
    )
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for corpus in CORPUS_ORDER:
        rows = _corpus_records(extraction, corpus)
        kids = _children(rows)
        totals = _sum_records(rows)
        add(
            f"| {corpus} | {len(kids)} | {totals['n_files']} | {totals['n_trouble_points']} | "
            f"{totals['n_unintelligible']} | {totals['n_error_word']} | {totals['n_phonological_form']} | "
            f"{totals['n_baseline']} | {totals['n_excluded_omission']} | {totals['n_excluded_question']} | "
            f"{totals['n_excluded_other_error']} | {totals['n_gold_targets']} |"
        )
    add("")
    add("### Group breakdown")
    add("")
    add("| Corpus | Group | Children | Files | Trouble | Baseline |")
    add("| --- | --- | ---: | ---: | ---: | ---: |")
    for corpus in CORPUS_ORDER:
        rows = _corpus_records(extraction, corpus)
        groups: dict[str, list[FileRecord]] = defaultdict(list)
        for row in rows:
            groups[row.group].append(row)
        for group in sorted(groups):
            members = groups[group]
            totals = _sum_records(members)
            add(
                f"| {corpus} | {group} | {len(_children(members))} | {totals['n_files']} | "
                f"{totals['n_trouble_points']} | {totals['n_baseline']} |"
            )
    add("")
    add("## Child counts against the TalkBank pages")
    add("")
    add(_talkbank_section(extraction))
    add("")
    add("## Suspicious items")
    add("")
    lines.extend(_suspicious_lines(extraction))
    add("")
    add("## Outputs")
    add("")
    add(f"- Counts: `{csv_path.relative_to(PROJECT_ROOT)}`")
    add(f"- Regex sample: `{sample_path.relative_to(PROJECT_ROOT)}`")
    add("")
    add(
        "The sample quotes transcript lines. It is a local check, and it must not be committed or redistributed."
    )
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _talkbank_section(extraction: Extraction) -> str:
    parts: list[str] = []
    feldman = _corpus_records(extraction, "Feldman")
    focal = sorted({row.child_id for row in feldman if row.group == "focal_lesion"})
    td = sorted({row.child_id for row in feldman if row.group == "td"})
    missing = sorted(FELDMAN_PAGE_TARGET_CODES - set(focal))
    extra = sorted(set(focal) - FELDMAN_PAGE_TARGET_CODES)
    unnamed = sorted({row.child_id for row in feldman if row.child_id.startswith("NCHI")})
    parts.append(
        "Pages checked: "
        "[Feldman Parent-Child](https://talkbank.org/childes/access/Clinical-Eng/Feldman/ParentChild.html) "
        "(header 56 target + 50 control; the target-code table lists 55 codes), "
        "[Conti1](https://talkbank.org/childes/access/Clinical-Eng/Conti/Conti1.html) (12 participants), "
        "[Conti2](https://talkbank.org/childes/access/Clinical-Eng/Conti/Conti2.html) (6), "
        "[Conti3](https://talkbank.org/childes/access/Clinical-Eng/Conti/Conti3.html) (4), "
        "[Ellis Weismer](https://talkbank.org/childes/access/Clinical-Eng/EllisWeismer.html) (138 participants; the prose also says 112 = 56 late talkers + 56 controls), "
        "[Rescorla](https://talkbank.org/childes/access/Clinical-Eng/Rescorla.html) (79 participants). "
        "The Clinical-Eng index also says Rescorla is 38 late talkers and 22 controls, at ages 3, 4, 5, 7, and 9."
    )
    parts.append("")
    parts.append(
        f"Feldman focal-lesion children in these files: {len(focal)}. "
        f"Page target codes not found as header names: {', '.join(missing) if missing else 'none'}. "
        f"Header names not in the page target table: {', '.join(extra) if extra else 'none'}. "
        f"TD children: {len(td)} (page header says 50 controls). "
        f"Files whose header has no child name, kept as separate ids: {', '.join(unnamed) if unnamed else 'none'}. "
        "Control header names are often not the page's code (for example MARY where the page says MA2, GST where it says GAS). "
        "Those were not remapped."
    )
    conti1 = _children(_corpus_records(extraction, "Conti1"))
    dld = sorted(child for child, group in conti1 if group == "dld")
    sib = sorted(child for child, group in conti1 if group == "td_sibling")
    parts.append("")
    parts.append(
        f"Conti1 included children: {len(dld)} dld ({', '.join(dld)}) and {len(sib)} td_sibling ({', '.join(sib)}). "
        "The page header says 12 participants. Its family table lists Rick/Rose, Clay/Charles, Abe/Ann, Martin/Mathue, Kate/Kyle, and Sean/Susan, "
        "and the prose says the Abe and Ann transcripts are missing. "
        "These files have Jack/Jim and Sid instead of Abe/Ann and Sean. Sid is consistent with Sean; Jack/Jim is not a name on the page. "
        "The included count is 12, which matches the header, but the names do not match the table. "
        "Six unrelated-control families in `0control` (Janet, Kaleb, Kevin, Mary, Robert, Sally) are excluded by the flag."
    )
    conti2 = _children(_corpus_records(extraction, "Conti2"))
    parts.append("")
    parts.append(
        "Conti2 children: "
        + ", ".join(f"{child} ({group})" for (child, group) in sorted(conti2))
        + ". The page lists Colin, Andrew, and Mark (SLI) and siblings Chris, Nina, and Adam. Count expected: 6."
    )
    conti3 = _children(_corpus_records(extraction, "Conti3"))
    parts.append("")
    parts.append(
        "Conti3 children: "
        + ", ".join(child for child, _group in sorted(conti3))
        + ". The page lists Bonnie, Dan, Harry, and Nathan. The folder and id used here are Daniel, as requested, not Dan. Count expected: 4."
    )
    ew = _children(_corpus_records(extraction, "EllisWeismer"))
    late = sum(1 for _child, group in ew if group == "late_talker")
    typical = sum(1 for _child, group in ew if group == "td")
    parts.append("")
    parts.append(
        f"Ellis Weismer children after the status-sheet rule: {len(ew)} ({late} late_talker, {typical} td). "
        f"The status sheet has {sum(1 for value in extraction.ew_status.values() if value == '1')} Status 1 and "
        f"{sum(1 for value in extraction.ew_status.values() if value == '0')} Status 0 "
        f"({len(extraction.ew_status)} ids). The page header says 138 participants."
    )
    res = _children(_corpus_records(extraction, "Rescorla"))
    res_late = sum(1 for _child, group in res if group == "late_talker")
    res_td = sum(1 for _child, group in res if group == "td")
    parts.append("")
    parts.append(
        f"Rescorla children kept after dropping 108- and 156-month files: {len(res)} ({res_late} late_talker, {res_td} td). "
        "The page says 79 participants and does not give a separate count for the 36/48/60-month sample. "
        "The Clinical-Eng index says 38 late talkers and 22 controls across ages 3, 4, 5, 7, and 9, so it is not the same denominator."
    )
    return "\n".join(parts)


def _suspicious_lines(extraction: Extraction) -> list[str]:
    lines: list[str] = []
    skip_counts: dict[str, int] = defaultdict(int)
    for _rel, reason in extraction.skipped:
        skip_counts[reason] += 1
    lines.append("Files not scored:")
    lines.append("")
    for reason, count in sorted(skip_counts.items()):
        lines.append(f"- {count}: {reason}")
    if extraction.no_target_child:
        lines.append("")
        lines.append(
            f"Files with no single Target_Child ({len(extraction.no_target_child)}). "
            "These were not scored, because the child and the adult cannot be identified from the roles:"
        )
        lines.append("")
        for item in extraction.no_target_child:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("Ellis Weismer status sheet:")
    lines.append("")
    if extraction.ew_in_both:
        rendered = ", ".join(f"{child} (Status {code})" for child, code in extraction.ew_in_both)
        lines.append(f"- IDs filed in both LT and TD: {rendered}.")
    else:
        lines.append("- No ID was filed in both LT and TD.")
    if extraction.ew_reassigned:
        lines.append("- Reassignments:")
        for item in extraction.ew_reassigned:
            lines.append(f"  - {item}")
    if extraction.ew_status_disagreements:
        lines.append("- Folder and Status disagree for a child who is in only one folder. The folder was kept:")
        for item in extraction.ew_status_disagreements:
            lines.append(f"  - {item}")
    else:
        lines.append("- No single-folder ID disagreed with Status.")
    if extraction.ew_missing_from_sheet:
        lines.append(
            "- Transcript IDs absent from `0status.xlsx`: "
            + ", ".join(extraction.ew_missing_from_sheet)
            + ". Folder group was kept."
        )
    if extraction.ew_sheet_not_in_files:
        lines.append(
            f"- Status-sheet IDs with no transcript in LT or TD ({len(extraction.ew_sheet_not_in_files)}): "
            + ", ".join(extraction.ew_sheet_not_in_files)
            + "."
        )
    lines.append("")
    bare = sum(row.n_bare_zero_baseline for row in extraction.records)
    err_base = sum(row.n_baseline_with_err for row in extraction.records)
    err_not = sum(row.n_err_tier_not_trouble for row in extraction.records)
    q_base = sum(row.n_baseline_with_question for row in extraction.records)
    odd = sum(row.n_odd_unintelligible for row in extraction.records)
    unscoped = sum(row.n_unscoped_bare_star for row in extraction.records)
    gold_ex = sum(row.n_gold_on_excluded for row in extraction.records)
    lines.append(
        f"Bare `0` child turns (no speech / untranscribed) that fell into the baseline: {bare}. "
        "They are not the `0is [*]` omission pattern. They were left in the baseline because the rules do not exclude them."
    )
    www = sum(row.n_baseline_only_www for row in extraction.records)
    lines.append("")
    lines.append(
        f"Baseline turns whose only word is `www` (CHAT's untranscribed marker, not `xxx` or `yyy`): {www}. "
        "They were left in the baseline. "
        f"Adult-followed child turns that contain `xx` or `www` anywhere: {odd}."
    )
    lines.append("")
    lines.append("Per corpus, baseline bare `0`, baseline `www`-only, `%err` without main-tier trouble, and trouble turns that also contain an omission:")
    lines.append("")
    lines.append("| Corpus | Baseline bare 0 | Baseline www only | %err but not main-tier trouble | Trouble turns that also have an omission |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for corpus in CORPUS_ORDER:
        rows = _corpus_records(extraction, corpus)
        lines.append(
            f"| {corpus} | {sum(row.n_bare_zero_baseline for row in rows)} | "
            f"{sum(row.n_baseline_only_www for row in rows)} | "
            f"{sum(row.n_err_tier_not_trouble for row in rows)} | "
            f"{sum(row.n_trouble_with_omission for row in rows)} |"
        )
    lines.append("")
    lines.append(
        f"Child turns followed by an adult that have a `%err` tier but are not main-tier trouble: {err_not}. "
        f"Of those, {err_base} remained in the baseline. "
        "A `%err` tier does not assign a trouble type. It does remove the turn from the baseline."
    )
    lines.append("")
    lines.append(
        f"Baseline turns that also contain `[?]`: {q_base}. "
        "Any `[?]` keeps a turn out of the baseline."
    )
    lines.append("")
    lines.append(
        f"Bare `[*]` that did not sit on a word and was still counted because the turn has a produced word: {unscoped}."
    )
    lines.append("")
    lines.append(
        f"Gold-target strings on excluded turns (not included in `n_gold_targets`): {gold_ex}."
    )
    if extraction.age_mismatches:
        lines.append("")
        lines.append(
            f"`@ID` age and the digit folder differ by more than one month ({len(extraction.age_mismatches)}). "
            "The `@ID` age was kept:"
        )
        lines.append("")
        for item in extraction.age_mismatches[:20]:
            lines.append(f"- {item}")
        if len(extraction.age_mismatches) > 20:
            lines.append(f"- … {len(extraction.age_mismatches) - 20} more")
    follower_totals: dict[str, int] = defaultdict(int)
    for row in extraction.records:
        for role, count in row.follower_counts.items():
            follower_totals[role] += count
    other = {
        role: count
        for role, count in follower_totals.items()
        if role not in ADULT_ROLES and role not in NON_LISTENER_ROLES and role != "end"
    }
    lines.append("")
    lines.append(
        "Child turns whose next speaker is not an adult, a child/sibling, or the end of the file: "
        + (
            ", ".join(f"{role} {count}" for role, count in sorted(other.items()))
            if other
            else "none"
        )
        + ". Those turns were not scored."
    )
    gr_ids = sorted(
        {
            row.child_id
            for row in extraction.records
            if row.corpus == "Rescorla" and re.search(r"\d{2,}[a-z]$", row.child_id)
        }
    )
    if gr_ids:
        lines.append("")
        lines.append(
            "Rescorla ids that kept a trailing letter so they would not be merged: "
            + ", ".join(gr_ids)
            + ". `gr36a` and `gr36b` are both male and aged 3;00, with no names in the header. "
            "They might be two children or two sessions. They were not merged, and they were not joined to `gr1` / `gr2`."
        )
    if extraction.warnings:
        lines.append("")
        lines.append(f"Parse warnings ({len(extraction.warnings)}):")
        lines.append("")
        for item in extraction.warnings[:30]:
            lines.append(f"- {item}")
        if len(extraction.warnings) > 30:
            lines.append(f"- … {len(extraction.warnings) - 30} more")
    lines.append("")
    lines.append("Most common produced forms counted as `phonological_form` (casual reductions on the explicit list are already excluded):")
    lines.append("")
    for corpus in CORPUS_ORDER:
        counts: Counter[str] = Counter()
        for row in _corpus_records(extraction, corpus):
            counts.update(row.phon_forms)
        if not counts:
            continue
        top = ", ".join(f"{form} {count}" for form, count in counts.most_common(12))
        lines.append(f"- {corpus}: {top}")
    lines.append("")
    lines.append(
        "Still open: whether bare `0` and `www`-only turns should leave the baseline, "
        "and whether forms such as `sposta` or `da [: do]` should be added to the casual-reduction list. "
        "A `[*]` subcode on a word the child produced, including `[* m:+ed]`, is now `error_word`. "
        "A turn is not baseline when its only error mark is a `%err` tier."
    )
    return lines


def run(
    data_root: Path,
    csv_path: Path,
    report_path: Path,
    sample_path: Path,
    *,
    include_conti1_unrelated_controls: bool = INCLUDE_CONTI1_UNRELATED_CONTROLS,
) -> Extraction:
    extraction = extract(
        data_root,
        include_conti1_unrelated_controls=include_conti1_unrelated_controls,
    )
    rows = child_rows(extraction)
    write_csv(csv_path, rows)
    write_sample(sample_path, extraction)
    write_report(report_path, extraction, csv_path, sample_path)
    return extraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract CHAT trouble points.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--include-conti1-unrelated-controls",
        action="store_true",
        help="Score Conti1 fatcon* and motcon* files as td_unrelated.",
    )
    args = parser.parse_args(argv)
    extraction = run(
        args.data_root,
        args.csv,
        args.report,
        args.sample,
        include_conti1_unrelated_controls=args.include_conti1_unrelated_controls,
    )
    print(f"wrote {args.csv}")
    print(f"wrote {args.report}")
    print(f"wrote {args.sample}")
    for corpus in CORPUS_ORDER:
        rows = _corpus_records(extraction, corpus)
        totals = _sum_records(rows)
        print(
            f"{corpus}: children={len(_children(rows))} files={totals['n_files']} "
            f"trouble={totals['n_trouble_points']} baseline={totals['n_baseline']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
