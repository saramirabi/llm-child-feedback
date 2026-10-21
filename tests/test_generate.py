"""Parser, age, stratification, and config tests. No API calls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate import (
    claude_max_tokens,
    format_age,
    listener_messages,
    load_config,
    parse_reply,
    projected_cost,
    stratify_pilot,
    strip_thinking,
    system_prompt,
)


def row(item_id: str, corpus: str, trouble_class: str, exclude: bool = False) -> dict:
    return {
        "item_id": item_id,
        "corpus": corpus,
        "trouble_class": trouble_class,
        "exclude": exclude,
    }


class ParseTests(unittest.TestCase):
    def test_leading_adult_label_is_removed(self) -> None:
        reply, multi, meta = parse_reply("Adult: what's that")
        self.assertEqual(reply, "what's that")
        self.assertFalse(multi)
        self.assertFalse(meta)

    def test_later_speaker_line_is_cut(self) -> None:
        reply, multi, _meta = parse_reply("Adult: what's that\nChild: a ball")
        self.assertEqual(reply, "what's that")
        self.assertTrue(multi)

    def test_second_adult_turn_is_cut(self) -> None:
        reply, multi, _meta = parse_reply("hello\nAdult: again")
        self.assertEqual(reply, "hello")
        self.assertTrue(multi)

    def test_meta_is_whole_words_only(self) -> None:
        _reply, _multi, meta = parse_reply("I would say the dog")
        self.assertTrue(meta)
        _reply, _multi, meta = parse_reply("as an ai I can help")
        self.assertTrue(meta)
        _reply, _multi, meta = parse_reply("he said the train is aside")
        self.assertFalse(meta)
        _reply, _multi, meta = parse_reply("what's that")
        self.assertFalse(meta)

    def test_age_drops_the_day_fraction(self) -> None:
        self.assertEqual(format_age(30.5), "2 years 6 months")
        self.assertEqual(format_age(12), "1 year")
        self.assertEqual(format_age(1), "1 month")
        self.assertEqual(format_age(0), "0 months")

    def test_claude_keeps_sixty_visible_tokens(self) -> None:
        self.assertEqual(claude_max_tokens(60), 60)

    def test_gemma_system_prompt_goes_in_the_user_message(self) -> None:
        messages = listener_messages("Be brief.", "Child: hi", system_delivery="user_message")
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Be brief.", messages[0]["content"])
        self.assertIn("Child: hi", messages[0]["content"])
        self.assertEqual(len(messages), 1)

    def test_qwen_think_block_is_removed(self) -> None:
        visible, had = strip_thinking("<think>planning</think>\nhello there")
        self.assertEqual(visible, "hello there")
        self.assertTrue(had)
        visible, had = strip_thinking("<think>\n</think>\nhello")
        self.assertEqual(visible, "hello")
        self.assertFalse(had)

    def test_prompts(self) -> None:
        self.assertNotIn("{age}", system_prompt("plain", "2 years 6 months"))
        self.assertIn("2 years 6 months", system_prompt("child", "2 years 6 months"))
        self.assertIn("check your understanding", system_prompt("repair", "2 years 6 months"))


class SampleTests(unittest.TestCase):
    def test_pilot_skips_excluded_and_draws_fifty(self) -> None:
        items = []
        for corpus in ("Feldman", "Conti1", "Conti2", "Conti3", "EllisWeismer", "Rescorla"):
            for trouble in ("unintelligible", "error", "none"):
                for index in range(20):
                    items.append(row(f"{corpus}-{trouble}-{index}", corpus, trouble))
        items.append(row("excluded-1", "Feldman", "error", exclude=True))
        chosen = stratify_pilot(items, n=50, seed=42)
        self.assertEqual(len(chosen), 50)
        self.assertTrue(all(not item["exclude"] for item in chosen))
        self.assertEqual(len({item["item_id"] for item in chosen}), 50)
        again = stratify_pilot(items, n=50, seed=42)
        self.assertEqual([item["item_id"] for item in chosen], [item["item_id"] for item in again])

    def test_projection_scales_by_usable_items(self) -> None:
        full, subsample = projected_cost(1.5, 50, 1000)
        self.assertAlmostEqual(full, 30.0)
        self.assertAlmostEqual(subsample, 9.9)


class ConfigTests(unittest.TestCase):
    def test_missing_config_stops(self) -> None:
        with self.assertRaises(SystemExit):
            load_config(Path(tempfile.gettempdir()) / "no-such-listener-config.yaml")

    def test_model_list_parses(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(
                "\n".join(
                    [
                        "seed: 42",
                        "max_output_tokens: 60",
                        "pilot_n: 50",
                        "models:",
                        "  - id: gpt-test",
                        "    provider: openai",
                        "    reasoning: true",
                        "    input_per_million: 0.05",
                        "    output_per_million: 0.40",
                        "  - id: local",
                        "    provider: lmstudio",
                        "    reasoning: false",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["models"][0]["id"], "gpt-test")
        self.assertEqual(config["models"][0]["reasoning"], True)
        self.assertEqual(config["models"][0]["input_per_million"], 0.05)
        self.assertEqual(config["models"][1]["provider"], "lmstudio")


if __name__ == "__main__":
    unittest.main()
