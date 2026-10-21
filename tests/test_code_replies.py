"""Judge parsing and target matching. No model calls."""

from __future__ import annotations

import unittest

from scripts.code_replies import (
    GOLD,
    CLARIFICATION,
    build_tasks,
    cohens_kappa,
    paired_binary,
    contains_target,
    corrects_user,
    judge_spec,
    judge_user,
    n_words,
    notice_user,
    parse_flag,
    parse_notice_judgment,
    parse_reply_judgment,
    present_context,
    spoken_child_turn,
    visible_user,
    validation_sample,
)


class ParseTests(unittest.TestCase):
    def test_one_json_object_is_accepted(self) -> None:
        parsed = parse_reply_judgment('{"category":"open_request","wrong_role":false}')
        self.assertEqual(parsed["category"], "open_request")
        self.assertFalse(parsed["wrong_role"])

    def test_extra_text_is_rejected(self) -> None:
        self.assertIsNone(parse_reply_judgment('Sure. {"category":"acknowledge","wrong_role":false}'))
        self.assertIsNone(parse_reply_judgment('{"category":"acknowledge","wrong_role":false}\n{"category":"echo_error","wrong_role":false}'))
        self.assertIsNone(parse_reply_judgment('{"category":"not_a_category","wrong_role":false}'))
        self.assertIsNone(parse_notice_judgment('{"noticeable_error":true,"category":"echo_error"}'))

    def test_notice_object(self) -> None:
        self.assertEqual(parse_notice_judgment('{"noticeable_error":true}')["noticeable_error"], True)

    def test_markdown_fence_is_one_object(self) -> None:
        parsed = parse_reply_judgment('```json\n{"category":"acknowledge","wrong_role":false}\n```')
        self.assertEqual(parsed["category"], "acknowledge")

    def test_prompt_marks_the_child_turn_and_does_not_force_not_noticeable(self) -> None:
        context = "Adult: look\nChild: baby eat the supper"
        prompt = judge_user(context, "mhm", qwen=False)
        self.assertIn("CHILD'S TURN BEING RESPONDED TO: Child: baby eat the supper", prompt)
        self.assertIn("A clarification (open_request, restricted_request, candidate_check)", prompt)
        self.assertIn("I didn't quite catch that, can you say it again?", prompt)
        self.assertIn("Is baby making a funny face in the mirror?", prompt)
        self.assertIn("Who's Huhica?", prompt)
        self.assertIn("it's just for the stable", prompt)
        self.assertIn("Short adult reactions are never wrong_role", prompt)
        self.assertEqual(present_context(context).splitlines()[-1][:5], "CHILD")
        notice = notice_user("Adult: look\nChild: baby (unintelligible) eat", qwen=False)
        self.assertIn("baby eat", notice)
        self.assertNotIn("(unintelligible)", notice)
        self.assertIn("baby eat the supper", notice)
        self.assertIn("I still got the same fish", notice)
        self.assertNotIn('Output exactly {"noticeable_error":false}', notice)
        self.assertEqual(spoken_child_turn("Child: when (unintelligible) him did that"), "when him did that")

    def test_corrects_and_visible_prompts_are_separate(self) -> None:
        context = "Adult: look\nChild: baby eat the supper"
        corrects = corrects_user(context, "Oh, the baby eats supper?", qwen=False)
        self.assertIn("Oh, the baby eats supper?", corrects)
        self.assertIn("corrects_error", corrects)
        visible = visible_user("Child: two three nine", qwen=False)
        self.assertIn("car crash into people", visible)
        self.assertIn("now what's wrong dear", visible)
        self.assertIn("counting error", visible)
        self.assertEqual(parse_flag('{"corrects_error":true}', "corrects_error")["corrects_error"], True)
        self.assertIsNone(parse_flag('{"visible_trouble":true,"category":"ignore_shift"}', "visible_trouble"))

    def test_gpt_replies_go_to_the_claude_judge(self) -> None:
        config = {
            "judges": {
                "gpt_replies": {"provider": "anthropic", "model": "claude-test"},
                "other_replies": {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
            }
        }
        gpt = judge_spec(config, {"task": "reply", "listener": "repair"})
        human = judge_spec(config, {"task": "reply", "listener": "human", "judge_slot": "other_replies"})
        human_claude = judge_spec(config, {"task": "reply", "listener": "human", "judge_slot": "gpt_replies"})
        self.assertEqual(human_claude["model"], "claude-test")
        child_turn = judge_spec(config, {"task": "visible", "listener": None})
        self.assertEqual(gpt["model"], "claude-test")
        self.assertEqual(human["model"], "gpt-4.1-2025-04-14")
        self.assertEqual(child_turn["provider"], "openai")
        with self.assertRaises(SystemExit):
            judge_spec({"judges": {"gpt_replies": {"provider": "anthropic", "model": None}}}, {"task": "reply", "listener": "plain"})

    def test_kappa_is_one_when_judges_agree(self) -> None:
        kappa, agree, count = cohens_kappa([("acknowledge", "acknowledge"), ("open_request", "open_request")])
        self.assertEqual(kappa, 1.0)
        self.assertEqual((agree, count), (2, 2))
        self.assertIsNone(cohens_kappa([])[0])

    def test_binary_kappa_drops_a_missing_corrects_error(self) -> None:
        left = {
            "a": {"category": "open_request", "corrects_error": True},
            "b": {"category": "acknowledge", "corrects_error": None},
        }
        right = {
            "a": {"category": "topic_question", "corrects_error": False},
            "b": {"category": "acknowledge", "corrects_error": True},
        }
        clar = paired_binary(left, right, "category", CLARIFICATION)
        self.assertEqual(clar, [("yes", "no"), ("no", "no")])
        corrects = paired_binary(left, right, "corrects_error", True)
        self.assertEqual(corrects, [("yes", "no")])
        kappa, agree, count = cohens_kappa(clar)
        self.assertEqual((agree, count), (1, 2))
        self.assertEqual(kappa, 0.0)

    def test_claude_replies_use_the_gpt_judge(self) -> None:
        items = [
            {
                "item_id": "a",
                "human_next_turn": "mhm",
                "context": "Child: hi",
                "trouble_class": "none",
                "gold_target": None,
            }
        ]
        generations = [
            {"item_id": "a", "model": "claude-sonnet-5-5", "prompt_condition": "plain", "reply": "hello"},
            {"item_id": "a", "model": "gemma-3-4b", "prompt_condition": "plain", "reply": "hello"},
        ]
        tasks = build_tasks(items, generations, ["claude-sonnet-5-5", "gemma-3-4b"])
        slots = {
            (task["speaker_model"], task["judge_slot"])
            for task in tasks
            if task.get("listener") == "plain"
        }
        self.assertEqual(
            slots,
            {("claude-sonnet-5-5", "other_replies"), ("gemma-3-4b", "gpt_replies")},
        )

    def test_human_replies_are_assigned_to_both_judges(self) -> None:
        items = [
            {
                "item_id": "a",
                "human_next_turn": "mhm",
                "context": "Child: hi",
                "trouble_class": "none",
                "gold_target": None,
            }
        ]
        tasks = build_tasks(items, [], "gpt")
        human = [task["judge_slot"] for task in tasks if task.get("listener") == "human"]
        self.assertEqual(human, ["gpt_replies", "other_replies"])
        visible = [task for task in tasks if task["task"] == "visible"]
        self.assertEqual(visible[0]["prompt_version"], "v2")

    def test_gold_set_counts(self) -> None:
        self.assertEqual(sum(1 for case in GOLD if case["expected"] == "open_request"), 4)
        self.assertEqual(sum(1 for case in GOLD if case["expected"] == "acknowledge"), 7)
        self.assertEqual(sum(1 for case in GOLD if case["expected"] == "topic_question"), 3)
        self.assertEqual(len(GOLD), 19)

    def test_target_ignores_case_and_punctuation(self) -> None:
        self.assertTrue(contains_target("The dog?", ["dog"]))
        self.assertTrue(contains_target("it is there", ["it is"]))
        self.assertFalse(contains_target("doggy", ["dog"]))
        self.assertIsNone(contains_target("yeah", None))
        self.assertEqual(n_words("good job"), 2)


class ValidationTests(unittest.TestCase):
    def test_sample_spreads_category_and_listener(self) -> None:
        records = []
        for category in ("acknowledge", "open_request", "ignore_shift"):
            for listener in ("human", "plain"):
                for index in range(8):
                    records.append(
                        {
                            "task": "reply",
                            "failed": False,
                            "category": category,
                            "listener": listener,
                            "item_id": f"{category}-{listener}-{index}",
                            "context": "Child: hi",
                            "reply": "yes",
                            "trouble_class": "none",
                        }
                    )
        chosen = validation_sample(records, n=10, seed=42)
        self.assertEqual(len(chosen), 10)
        cells = {(row["category"], row["listener"]) for row in chosen}
        self.assertEqual(len(cells), 6)
        again = validation_sample(records, n=10, seed=42)
        self.assertEqual([row["item_id"] for row in chosen], [row["item_id"] for row in again])


if __name__ == "__main__":
    unittest.main()
