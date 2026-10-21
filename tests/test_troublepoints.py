"""Unit tests for the CHAT trouble-point extractor.

These tests use synthetic transcripts. They do not read or modify data/raw.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.troublepoints import (
    age_in_months,
    child_rows,
    classify_utterance,
    extract,
    load_ellisweismer_status,
    parse_chat,
    rescorla_child_id,
    visibility_failure,
)

REAL_STATUS = Path(
    "data/raw/childes_clinical_eng/Clinical-Eng/EllisWeismer/0status.xlsx"
)


def write_cha(root: Path, relative: str, body: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


FELDMAN = """@Begin
@Participants:\tCHI Bar Target_Child , MOT Mother , SIB Sib Sibling , UNC Uncle Relative
@ID:\teng|ParentChild|CHI|1;06.15|male|||Target_Child|||
@ID:\teng|ParentChild|MOT||female|||Mother|||
@ID:\teng|ParentChild|SIB|3;||||Sibling|||
@ID:\teng|ParentChild|UNC|||||Relative|||
*CHI:\txxx open
\tit .
%mor:\tv|open .
*SIB:\tno .
*CHI:\t0is [*] gone .
*MOT:\twhat ?
*CHI:\tring [?] .
*MOT:\tthe ring .
*CHI:\tbaied [: played] the xxx .
*MOT:\tplayed ?
*CHI:\tdunno [: don't know] .
*MOT:\tokay .
*CHI:\tgo sleep [*] .
*SIB:\tno .
*CHI:\tirg [: bird] .
@Comment:\tnoise
*UNC:\ta bird ?
*CHI:\tcome [* 0es] back .
*MOT:\tcome back .
*CHI:\tputted [: put] [* m:+ed] it .
*MOT:\tput it .
*CHI:\t0 .
*MOT:\toh .
@End
"""

UNNAMED = """@Begin
@Participants:\tCHI Target_Child , MOT Mother
@ID:\teng|ParentChild|CHI|3;06.|female|||Target_Child|||
@ID:\teng|ParentChild|MOT||female|||Mother|||
*CHI:\thi .
*MOT:\thello .
@End
"""

CONTI_DLD = """@Begin
@ID:\teng|Conti1|CHI|6;05.05|male|||Target_Child|||
@Participants:\tCHI Clay Target_Child , FAT Father
@ID:\teng|Conti1|FAT|43;||||Father|||
*CHI:\tyyy .
*FAT:\twhat ?
@End
"""

CONTI_SIB = """@Begin
@Participants:\tCHI Charles Target_Child , MOT Mother
@ID:\teng|Conti1|CHI|2;04.|male|||Target_Child|||
@ID:\teng|Conti1|MOT||female|||Mother|||
*CHI:\thi .
*MOT:\thello .
@End
"""

CONTI_CONCON = """@Begin
@Participants:\tCHI Janet Target_Child , SIB Jeff Sibling
@ID:\teng|Conti1|CHI|2;01.|female|||Target_Child|||
@ID:\teng|Conti1|SIB|6;11.|male|||Sibling|||
*CHI:\txxx .
*SIB:\twhat ?
@End
"""

CONTI_FATCON = """@Begin
@Participants:\tCHI Janet Target_Child , FAT Father
@ID:\teng|Conti1|CHI|2;01.|female|||Target_Child|||
@ID:\teng|Conti1|FAT|40;||||Father|||
*CHI:\txxx .
*FAT:\twhat ?
@End
"""

ANDREW = """@Begin
@Participants:\tCHI Andrew Target_Child , MOT Mother
@ID:\teng|Conti2|CHI|5;03.01|male|||Target_Child|||
@ID:\teng|Conti2|MOT||female|||Mother|||
*CHI:\tokay .
*MOT:\tgood .
@End
"""

ADAM = """@Begin
@Participants:\tCHI Adam Target_Child , MOT Mother
@ID:\teng|Conti2|CHI|2;01.04|male|||Target_Child|||
@ID:\teng|Conti2|MOT||female|||Mother|||
*CHI:\tokay .
*MOT:\tgood .
@End
"""

DANIEL = """@Begin
@Participants:\tCHI Daniel Target_Child , MOT Mother
@ID:\teng|Conti3|CHI|2;06.29|male|||Target_Child|||
@ID:\teng|Conti3|MOT||female|||Mother|||
*CHI:\tokay .
*MOT:\tgood .
@End
"""

EW_LT = """@Begin
@Participants:\tCHI Target_Child , MOT Mother
@ID:\teng|EllisWeismer|CHI|3;06.|male|LT||Target_Child|||
@ID:\teng|EllisWeismer|MOT|||||Mother|||
*CHI:\twhere it goes [*] ?
*MOT:\twhere does it go ?
@End
"""

EW_TD_SAME = """@Begin
@Participants:\tCHI Target_Child , INV Investigator
@ID:\teng|EllisWeismer|CHI|4;06.|male|TD||Target_Child|||
@ID:\teng|EllisWeismer|INV|||||Investigator|||
*CHI:\t0is upstairs [*] .
*INV:\tupstairs ?
@End
"""

EW_TYPICAL = """@Begin
@Participants:\tCHI Target_Child , MOT Mother
@ID:\teng|EllisWeismer|CHI|2;06.|female|||Target_Child|||
@ID:\teng|EllisWeismer|MOT|||||Mother|||
*CHI:\thi .
*MOT:\thello .
@End
"""

RES_ALE = """@Begin
@Participants:\tCHI Target_Child , MOT Mother
@ID:\teng|Rescorla|CHI|3;00.|female|||Target_Child|||
@ID:\teng|Rescorla|MOT||female|||Mother|||
*CHI:\t<gonna> [: going to] go .
*MOT:\tokay .
@End
"""

RES_LY = """@Begin
@Participants:\tCHI Target_Child , MOT Mother
@ID:\teng|Rescorla|CHI|||||Target_Child|||
@ID:\teng|Rescorla|MOT||female|||Mother|||
*CHI:\tya [: you] go .
*MOT:\tyou go .
@End
"""

RES_108 = """@Begin
@Participants:\tCHI Target_Child , MOT Mother
@ID:\teng|Rescorla|CHI|9;00.|male|||Target_Child|||
@ID:\teng|Rescorla|MOT||female|||Mother|||
*CHI:\txxx .
*MOT:\twhat ?
@End
"""

RES_PAR = """@Begin
@Participants:\tPAR0 Participant , PAR1 Participant
@ID:\teng|Rescorla|PAR0|||||Participant|||
@ID:\teng|Rescorla|PAR1|||||Participant|||
*PAR0:\thi .
*PAR1:\thello .
@End
"""

RES_GR_A = """@Begin
@Participants:\tCHI Target_Child , MOT Mother
@ID:\teng|Rescorla|CHI|3;00.|male|||Target_Child|||
@ID:\teng|Rescorla|MOT||female|||Mother|||
*CHI:\thi .
*MOT:\thello .
@End
"""


def build_tree(root: Path) -> None:
    write_cha(root, "Feldman/ParentChild/SLI/PC-ac/bar18.cha", FELDMAN)
    write_cha(root, "Feldman/ParentChild/TD/nchi0842.cha", UNNAMED)
    write_cha(root, "Conti/Conti1/fatsli/fatsliclay.cha", CONTI_DLD)
    write_cha(root, "Conti/Conti1/fatsib/fatsibcharles.cha", CONTI_SIB)
    write_cha(root, "Conti/Conti1/0control/conconjanjeff.cha", CONTI_CONCON)
    write_cha(root, "Conti/Conti1/0control/fatconjanet.cha", CONTI_FATCON)
    write_cha(root, "Conti/Conti2/SLI/Andrew/050301.cha", ANDREW)
    write_cha(root, "Conti/Conti2/TD/Adam/020104.cha", ADAM)
    write_cha(root, "Conti/Conti3/Daniel/020629.cha", DANIEL)
    write_cha(root, "EllisWeismer/LT/30ec/22086.cha", EW_LT)
    write_cha(root, "EllisWeismer/TD/54int/22086.cha", EW_TD_SAME)
    write_cha(root, "EllisWeismer/TD/30ec/11006.cha", EW_TYPICAL)
    write_cha(root, "Rescorla/LT/36/ale36.cha", RES_ALE)
    write_cha(root, "Rescorla/LT/36/0nomedia/ly236.cha", RES_LY)
    write_cha(root, "Rescorla/LT/108/ale108.cha", RES_108)
    write_cha(root, "Rescorla/LT/156/ale156.cha", RES_108)
    write_cha(root, "Rescorla/TD/48/bre48.cha", RES_PAR)
    write_cha(root, "Rescorla/TD/36/gr36a.cha", RES_GR_A)
    write_cha(root, "Rescorla/TD/36/gr36b.cha", RES_GR_A)


class ClassifyTests(unittest.TestCase):
    def test_precedence_and_exclusions(self) -> None:
        unintelligible = classify_utterance("baied [: played] the xxx .")
        self.assertEqual(unintelligible.trouble, "unintelligible")
        self.assertEqual(unintelligible.gold_targets, ["played"])

        shifted = classify_utterance("where it goes [*] ?")
        self.assertEqual(shifted.trouble, "error_word")
        later = classify_utterance("0is upstairs [*] .")
        self.assertIsNone(later.trouble)
        self.assertEqual(later.exclusion, "omission")
        self.assertIsNone(classify_utterance("who [*] 0is [*] this ?").trouble)
        self.assertEqual(classify_utterance("who [*] 0is [*] this ?").exclusion, "omission")
        self.assertEqual(classify_utterance("how 0are you doing [*] ?").exclusion, "omission")

        omission = classify_utterance("0is [*] gone .")
        self.assertIsNone(omission.trouble)
        self.assertEqual(omission.exclusion, "omission")
        self.assertTrue(omission.omission_word)

        morpheme = classify_utterance("come [* 0es] back .")
        self.assertEqual(morpheme.exclusion, "omission")
        self.assertTrue(morpheme.omission_morpheme)
        self.assertIsNone(morpheme.trouble)

        question = classify_utterance("ring [?] .")
        self.assertEqual(question.exclusion, "question")
        mixed = classify_utterance("ring [/] ring [?] .")
        self.assertIsNone(mixed.trouble)
        self.assertEqual(mixed.exclusion, "question")
        self.assertIsNone(classify_utterance("word [/?] .").exclusion)

        overregular = classify_utterance("putted [: put] [* m:+ed] it .")
        self.assertEqual(overregular.trouble, "error_word")
        self.assertIsNone(overregular.exclusion)
        self.assertEqual(overregular.gold_targets, ["put"])
        self.assertEqual(classify_utterance("bleded [* m:+ed] .").trouble, "error_word")
        self.assertEqual(classify_utterance("goes [* m:03s] .").trouble, "error_word")
        self.assertEqual(classify_utterance("going [* m:0ing] .").trouble, "error_word")
        self.assertEqual(classify_utterance("<what are [*] this> [<] ?").trouble, "error_word")

        boat = classify_utterance("what [*] kind of boat is that ?")
        self.assertEqual(boat.trouble, "error_word")
        self.assertIsNotNone(
            visibility_failure(
                "what [*] kind of boat is that ?",
                boat,
                "DYS-initial+syllable+repetition",
            )
        )
        produced = classify_utterance("bleded [* m:+ed] .")
        self.assertIsNone(visibility_failure("bleded [* m:+ed] .", produced, ""))
        # The [*] sits on an omitted 0-token or a stray > from a nested group.
        # Those are not produced words, so the turn stays eligible.
        omitted = classify_utterance("<what <0aux> [*] that Ma> [/] what <0aux> [*] that Ma .")
        self.assertEqual(omitted.trouble, "error_word")
        self.assertIsNone(
            visibility_failure(
                "<what <0aux> [*] that Ma> [/] what <0aux> [*] that Ma .",
                omitted,
                "",
            )
        )
        stray = classify_utterance("<<more one> [<]> [*] .")
        self.assertEqual(stray.error_forms, [">"])
        self.assertIsNone(visibility_failure("<<more one> [<]> [*] .", stray, ""))

        phon = classify_utterance("irg [: bird] .")
        self.assertEqual(phon.trouble, "phonological_form")
        self.assertEqual(phon.gold_targets, ["bird"])
        grouped = classify_utterance("<py ad> [: pram] .")
        self.assertEqual(grouped.trouble, "phonological_form")
        self.assertEqual(grouped.gold_targets, ["pram"])

        casual = classify_utterance("dunno [: don't know] .")
        self.assertIsNone(casual.trouble)
        self.assertEqual(casual.gold_targets, [])
        self.assertIsNone(classify_utterance("<gonna> [: going to] go .").trouble)
        self.assertEqual(classify_utterance("ya [: you] go .").gold_targets, [])
        self.assertEqual(classify_utterance("xxx@u now .").trouble, "unintelligible")
        shifted_omission = classify_utterance("0he can't [*] .")
        self.assertIsNone(shifted_omission.trouble)
        self.assertEqual(shifted_omission.exclusion, "omission")

    def test_error_with_target_is_not_phonological(self) -> None:
        coded = classify_utterance("leaffrog [: leapfrog] [*] .")
        self.assertEqual(coded.trouble, "error_word")
        self.assertEqual(coded.gold_targets, ["leapfrog"])


class ParseTests(unittest.TestCase):
    def test_continuations_drop_dependent_tiers_and_keep_adult_after_comment(self) -> None:
        parsed = parse_chat(FELDMAN)
        child = [turn for turn in parsed.turns if turn.role == "Target_Child"]
        self.assertEqual(child[0].text, "xxx open it .")
        self.assertNotIn("v|open", child[0].text)
        roles = [turn.role for turn in parsed.turns]
        bird = next(turn for turn in child if turn.text.startswith("irg"))
        index = parsed.turns.index(bird)
        self.assertEqual(parsed.turns[index + 1].role, "Relative")
        self.assertNotIn("Comment", roles)

    def test_err_tier_is_noticed_after_another_dependent_tier(self) -> None:
        parsed = parse_chat(
            "@Begin\n"
            "@Participants:\tCHI Target_Child , MOT Mother\n"
            "@ID:\teng|X|CHI|3;00.|female|||Target_Child|||\n"
            "@ID:\teng|X|MOT|||||Mother|||\n"
            "*CHI:\thi .\n"
            "%wor:\thi .\n"
            "%err:\t$mor:sub:coc|is=are\n"
            "*MOT:\thello .\n"
            "@End\n"
        )
        child = next(turn for turn in parsed.turns if turn.role == "Target_Child")
        self.assertTrue(child.has_err_tier)
        self.assertEqual(child.text, "hi .")

    def test_name_is_kept_when_id_precedes_participants(self) -> None:
        parsed = parse_chat(CONTI_DLD)
        target = parsed.target_children()[0]
        self.assertEqual(target.name, "Clay")
        self.assertEqual(target.age_months, 77.17)


class AgeAndIdTests(unittest.TestCase):
    def test_age_in_months(self) -> None:
        self.assertIsNone(age_in_months(""))
        self.assertIsNone(age_in_months("nope"))
        self.assertEqual(age_in_months("1;06."), 18)
        self.assertEqual(age_in_months("1;06.15"), 18.5)
        self.assertEqual(age_in_months("6;05.05"), 77.17)
        self.assertEqual(age_in_months("2;6"), 30)

    def test_rescorla_child_id(self) -> None:
        self.assertEqual(rescorla_child_id("ale36", "36"), "ale")
        self.assertEqual(rescorla_child_id("ly236", "36"), "ly2")
        self.assertEqual(rescorla_child_id("gr36a", "36"), "gr36a")
        self.assertEqual(rescorla_child_id("gr36b", "36"), "gr36b")
        self.assertEqual(rescorla_child_id("gr148", "48"), "gr1")


class ExtractTests(unittest.TestCase):
    def test_groups_filters_and_counts(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_tree(root)
            before = {
                path: path.read_bytes()
                for path in root.rglob("*.cha")
            }
            result = extract(
                root,
                ellisweismer_status={"22086": "1", "11006": "0"},
            )
            after = {path: path.read_bytes() for path in root.rglob("*.cha")}
            self.assertEqual(before, after)

            rows = {(row.corpus, row.child_id, row.group): row for row in result.records}
            bar = [row for row in result.records if row.child_id == "BAR"]
            self.assertEqual(len(bar), 1)
            scored = bar[0]
            self.assertEqual(scored.group, "focal_lesion")
            self.assertEqual(scored.age_months, 18.5)
            self.assertEqual(scored.n_child_turns, 10)
            self.assertEqual(scored.n_followed_by_adult, 8)
            self.assertEqual(scored.n_unintelligible, 1)
            self.assertEqual(scored.n_error_word, 1)
            self.assertEqual(scored.n_phonological_form, 1)
            self.assertEqual(scored.n_trouble_points, 3)
            self.assertEqual(scored.n_baseline, 2)
            self.assertEqual(scored.n_excluded_omission, 2)
            self.assertEqual(scored.n_excluded_question, 1)
            self.assertEqual(scored.n_excluded_other_error, 0)
            self.assertEqual(scored.n_gold_targets, 3)
            self.assertEqual(scored.n_gold_on_excluded, 0)
            self.assertEqual(scored.n_bare_zero_baseline, 1)

            self.assertIn(("Feldman", "NCHI0842", "td"), rows)
            self.assertEqual(rows[("Conti1", "Clay", "dld")].n_unintelligible, 1)
            self.assertIn(("Conti1", "Charles", "td_sibling"), rows)
            self.assertNotIn(("Conti1", "Janet", "td_unrelated"), rows)
            self.assertNotIn(("Conti1", "Janet", "dld"), rows)
            self.assertTrue(any("concon" in rel for rel, _reason in result.skipped))
            self.assertTrue(any("fatcon" in rel for rel, _reason in result.skipped))

            self.assertEqual(rows[("Conti2", "Andrew", "dld")].group, "dld")
            self.assertEqual(rows[("Conti2", "Adam", "td_sibling")].group, "td_sibling")
            self.assertIn(("Conti3", "Daniel", "dld"), rows)

            ew = [row for row in result.records if row.child_id == "22086"]
            self.assertEqual({row.group for row in ew}, {"late_talker"})
            self.assertEqual(len(ew), 2)
            self.assertTrue(any("reassigned to late_talker" in note for note in result.ew_reassigned))
            self.assertEqual(rows[("EllisWeismer", "11006", "td")].group, "td")
            self.assertEqual(sum(row.n_error_word for row in ew), 1)

            self.assertIn(("Rescorla", "ale", "late_talker"), rows)
            self.assertEqual(rows[("Rescorla", "ly2", "late_talker")].age_months, 36)
            self.assertIn(("Rescorla", "gr36a", "td"), rows)
            self.assertIn(("Rescorla", "gr36b", "td"), rows)
            self.assertFalse(any(row.child_id == "ale" and row.age_months == 108 for row in result.records))
            self.assertTrue(any("108" in rel or "156" in rel for rel, _reason in result.skipped))
            self.assertTrue(result.no_target_child)
            self.assertTrue(any("bre48" in item for item in result.no_target_child))

            included = extract(
                root,
                include_conti1_unrelated_controls=True,
                ellisweismer_status={"22086": "1", "11006": "0"},
            )
            janet = [
                row
                for row in included.records
                if row.corpus == "Conti1" and row.child_id == "Janet"
            ]
            self.assertEqual(len(janet), 1)
            self.assertEqual(janet[0].group, "td_unrelated")
            self.assertTrue(any("concon" in rel for rel, _reason in included.skipped))

            table = child_rows(result)
            self.assertTrue(any(row["row_type"] == "corpus" and row["corpus"] == "Feldman" for row in table))
            self.assertTrue(any(row["child_id"] == "BAR" and row["group"] == "focal_lesion" for row in table))


class StatusSheetTests(unittest.TestCase):
    def test_real_sheet_marks_22086_as_late_talker(self) -> None:
        if not REAL_STATUS.exists():
            self.skipTest("Ellis Weismer status sheet is not in data/raw")
        status = load_ellisweismer_status(REAL_STATUS)
        self.assertEqual(status["22086"], "1")
        self.assertEqual(status["11006"], "0")
        self.assertGreater(len(status), 100)


if __name__ == "__main__":
    unittest.main()
