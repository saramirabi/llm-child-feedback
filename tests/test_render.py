"""Unit tests for listener rendering and sampling.

Synthetic utterances only. These tests do not read data/raw.
"""

from __future__ import annotations

import unittest

from scripts.render import (
    Item,
    NO_VISIBLE_ERROR_IDS,
    mark_matched_exclusions,
    recent_trouble_in_context,
    render_utterance,
    sample_unit,
    sampling_unit,
    subsample_child,
    target_appears_earlier,
    unit_rng,
)
from scripts.troublepoints import Turn


def item(trouble: str, item_id: str, corpus: str = "Feldman", child: str = "BAR") -> Item:
    return Item(
        item_id=item_id,
        session_id="session",
        corpus=corpus,
        group="td",
        child_id=child,
        child_age_months=18.0,
        trouble_type=trouble,
        trouble_class="none" if trouble == "none" else "error",
        all_trouble_types=[] if trouble == "none" else [trouble],
        in_group_contrast=corpus != "Conti3",
        gold_target=None,
        target_in_context=False,
        context="Child: hi",
        human_next_turn="yes",
        adult_role="Mother",
    )


class RenderTests(unittest.TestCase):
    def test_unintelligible_and_untranscribed(self) -> None:
        self.assertEqual(render_utterance("xxx ."), "(unintelligible)")
        self.assertEqual(render_utterance("yyy"), "(unintelligible)")
        self.assertEqual(render_utterance("www"), "(untranscribed)")
        self.assertEqual(render_utterance("see xxx now"), "see (unintelligible) now")

    def test_only_xxx_is_not_empty(self) -> None:
        self.assertTrue(render_utterance("xxx"))

    def test_error_and_phonological_forms_stay_produced(self) -> None:
        rendered = render_utterance("baied [: played] the xxx .")
        self.assertEqual(rendered, "baied the (unintelligible)")
        self.assertNotIn("played", rendered)
        self.assertEqual(render_utterance("go sleep [*] ."), "go sleep")
        self.assertEqual(render_utterance("putted [: put] [* m:+ed] it ."), "putted it")
        self.assertNotIn("put", render_utterance("putted [: put] it .").split())

    def test_question_retracing_and_timestamps_go(self) -> None:
        self.assertEqual(render_utterance("ring [?] ."), "ring")
        self.assertEqual(render_utterance("what [/] what color ?"), "what what color")
        self.assertEqual(render_utterance("<you do> [//] you do ."), "you do you do")
        self.assertEqual(
            render_utterance("see the xxx .\x152225_3035\x15"),
            "see the (unintelligible)",
        )

    def test_nested_groups_omissions_and_repetition(self) -> None:
        self.assertEqual(
            render_utterance("<the <big dog> barked> ."),
            "the big dog barked",
        )
        self.assertEqual(
            render_utterance("<what <0aux> [*] that Ma> [/] what <0aux> [*] that Ma ."),
            "what that Ma what that Ma",
        )
        self.assertEqual(render_utterance("where 0is <0the ball> ."), "where ball")
        self.assertEqual(render_utterance("0aux 0is 0the ball"), "ball")
        self.assertEqual(
            render_utterance("<baby Name [>] sister> [*] ."),
            "baby Name sister",
        )
        self.assertEqual(
            render_utterance("Gena <<cut [* m:0es]> [/] cut [* m:0es] my hair> [>] ."),
            "Gena cut cut my hair",
        )
        self.assertEqual(render_utterance("↫b-b↫baby"), "b-b-baby")
        self.assertEqual(render_utterance("↫li↫little ca-"), "li-little ca-")
        self.assertEqual(render_utterance("and ↫sh-sh-sh↫she sleep ."), "and sh-sh-sh-she sleep")
        self.assertNotIn("\u21ab", render_utterance("↫wha↫what's this"))

    def test_fillers_fragments_events_and_names(self) -> None:
        self.assertEqual(render_utterance("&=laughs &=coughs"), "")
        self.assertEqual(render_utterance("&-um gone outside ."), "um gone outside")
        self.assertEqual(render_utterance("&~uhn"), "uhn")
        self.assertEqual(render_utterance("&+ba"), "ba-")
        self.assertEqual(render_utterance("&+ba-"), "ba-")
        self.assertEqual(render_utterance("miss Mrs_xxx says"), "miss Mrs (name) says")
        self.assertEqual(render_utterance("Mr_xxx"), "Mr (name)")

    def test_omitted_sound_contractions_join_the_previous_word(self) -> None:
        self.assertEqual(render_utterance("that (i)s mine ."), "that's mine")
        self.assertEqual(render_utterance("it (i)s ."), "it's")
        self.assertEqual(render_utterance("what (i)s that ?"), "what's that")
        self.assertEqual(render_utterance("I (a)m here ."), "I'm here")
        self.assertEqual(render_utterance("let (u)s go ."), "let's go")
        self.assertEqual(render_utterance("where (doe)s it go ?"), "where's it go")
        self.assertEqual(render_utterance("she (ha)s it ."), "she's it")
        self.assertEqual(render_utterance("Barbie can play (a)s_well ."), "Barbie can play 's well")
        self.assertEqual(render_utterance("(be)cause"), "cause")
        self.assertEqual(render_utterance("s(ome) toys"), "s toys")
        self.assertEqual(render_utterance("s@l"), "s")
        self.assertEqual(render_utterance("m@l"), "m")

    def test_tag_marker_is_a_comma_and_curly_quotes_stay(self) -> None:
        self.assertEqual(
            render_utterance("that is good \u201e isn't it ."),
            "that is good, isn't it",
        )
        self.assertEqual(render_utterance('she said \u201chello\u201d .'), 'she said \u201chello\u201d')
        self.assertNotIn("\u201e", render_utterance("gone \u201e hasn't it"))

    def test_lengthening_colon_inside_a_word_is_dropped(self) -> None:
        self.assertEqual(render_utterance("a:h ."), "ah")
        self.assertEqual(render_utterance("&~a:h shakeyshakey@o ."), "ah shakeyshakey")
        self.assertEqual(render_utterance("o:h yea:h ."), "oh yeah")
        self.assertEqual(render_utterance("ba::ll ."), "ball")
        self.assertEqual(render_utterance("s:ti^cked ."), "sti^cked")

    def test_empty_coding_and_omissions(self) -> None:
        self.assertEqual(render_utterance("0"), "")
        self.assertEqual(render_utterance("0is [*] ."), "")
        self.assertEqual(render_utterance("[?]"), "")
        self.assertEqual(render_utterance("(be)cause"), "cause")

    def test_target_earlier_in_context_is_flagged_not_removed(self) -> None:
        earlier = "Adult: she played already"
        self.assertTrue(target_appears_earlier(["played"], earlier))
        self.assertFalse(target_appears_earlier(["played"], "Adult: she playeded"))
        self.assertFalse(target_appears_earlier(["played"], "Child: baied"))


class SamplingTests(unittest.TestCase):
    def test_cap_and_matched_baseline(self) -> None:
        pool = [item("unintelligible", f"u{i}") for i in range(12)]
        pool += [item("error_word", f"e{i}") for i in range(3)]
        pool += [item("none", f"n{i}") for i in range(4)]
        chosen, short = sample_unit(pool, unit_rng("BAR"), k=10)
        kinds = [row.trouble_type for row in chosen]
        self.assertEqual(kinds.count("unintelligible"), 10)
        self.assertEqual(kinds.count("error_word"), 3)
        self.assertEqual(kinds.count("none"), 4)
        self.assertEqual(short, 9)

    def test_seed_is_stable(self) -> None:
        pool = [item("unintelligible", f"u{i:02d}") for i in range(20)]
        first, _ = sample_unit(pool, unit_rng("same"), k=10)
        second, _ = sample_unit(pool, unit_rng("same"), k=10)
        self.assertEqual([row.item_id for row in first], [row.item_id for row in second])

    def test_invisible_error_excludes_its_baseline_slot(self) -> None:
        error_id = next(iter(NO_VISIBLE_ERROR_IDS))
        pool = [
            item("unintelligible", "u0"),
            item("error_word", error_id),
            item("phonological_form", "p0"),
            item("none", "n0"),
            item("none", "n1"),
            item("none", "n2"),
        ]
        chosen, short = sample_unit(pool, unit_rng("slot"), k=10)
        self.assertEqual(short, 0)
        missing = mark_matched_exclusions(chosen)
        self.assertEqual(missing, [])
        trouble = [row for row in chosen if row.trouble_type != "none"]
        baseline = [row for row in chosen if row.trouble_type == "none"]
        self.assertEqual(trouble[1].item_id, error_id)
        self.assertTrue(trouble[1].exclude)
        self.assertTrue(baseline[1].exclude)
        self.assertIn(error_id, baseline[1].exclude_reason or "")
        self.assertFalse(trouble[0].exclude)
        self.assertFalse(baseline[0].exclude)
        self.assertFalse(baseline[2].exclude)

    def test_recent_trouble_uses_the_previous_three_child_turns(self) -> None:
        def turn(role: str, text: str) -> Turn:
            return Turn(speaker="X", role=role, text=text, lineno=1)

        # The fourth child turn back has xxx. The latest three do not.
        turns = [
            turn("Target_Child", "xxx ."),
            turn("Mother", "what ?"),
            turn("Target_Child", "a ball ."),
            turn("Mother", "yes ."),
            turn("Target_Child", "blue ."),
            turn("Mother", "and ?"),
            turn("Target_Child", "car ."),
            turn("Mother", "where ?"),
            turn("Target_Child", "here ."),
        ]
        self.assertFalse(recent_trouble_in_context(turns, len(turns) - 1))
        turns[6] = turn("Target_Child", "go sleep [*] .")
        self.assertTrue(recent_trouble_in_context(turns, len(turns) - 1))
        turns[6] = turn("Target_Child", "baied [: played] .")
        self.assertTrue(recent_trouble_in_context(turns, len(turns) - 1))
        turns[6] = turn("Target_Child", "0is upstairs [*] .")
        self.assertFalse(recent_trouble_in_context(turns, len(turns) - 1))
        turns[0] = turn("Target_Child", "yeah .")
        turns[6] = turn("Sibling", "xxx .")
        self.assertFalse(recent_trouble_in_context(turns, len(turns) - 1))

    def test_recent_trouble_ignores_a_child_turn_outside_the_context(self) -> None:
        def turn(role: str, text: str) -> Turn:
            return Turn(speaker="X", role=role, text=text, lineno=1)

        early = [turn("Target_Child", "xxx ."), turn("Mother", "hmm .")]
        filler = [turn("Mother", f"line {i} .") for i in range(12)]
        target = [turn("Target_Child", "here .")]
        self.assertFalse(recent_trouble_in_context(early + filler + target, len(early) + len(filler)))

    def test_conti3_unit_is_child_and_session(self) -> None:
        self.assertEqual(sampling_unit("Conti3", "Harry", "Conti/Conti3/Harry/040401"), "Harry|Conti/Conti3/Harry/040401")
        self.assertEqual(sampling_unit("Rescorla", "gr1", "Rescorla/TD/60/gr160"), "gr1")

    def test_subsample_keeps_every_child(self) -> None:
        pool = [item("unintelligible", "u1"), item("none", "n1")]
        kept = subsample_child(pool, 0.0, unit_rng("child"))
        self.assertEqual(len(kept), 1)
        half = subsample_child(
            [item("unintelligible", f"u{i}") for i in range(10)],
            0.5,
            unit_rng("half"),
        )
        self.assertEqual(len(half), 5)


if __name__ == "__main__":
    unittest.main()
