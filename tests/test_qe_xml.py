from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess_qe_wmt21 import to_verl_record  # noqa: E402
from qe_reward import compute_score  # noqa: E402
from qe_xml_utils import (  # noqa: E402
    build_messages_xml_mt,
    first_nonempty_line,
    labels_to_xml,
    xml_to_labels,
)


def _load_sft_data_utils():
    path = PROJECT_DIR.parent / "work_SFT" / "QE_SFT_8B" / "src" / "models" / "data_utils.py"
    spec = importlib.util.spec_from_file_location("sft_qe_data_utils_for_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class XmlCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sft = _load_sft_data_utils()

    def test_prompt_matches_sft(self):
        src = "This is a source."
        mt = "Dies ist eine Übersetzung ."
        expected = self.sft.build_messages_any("xml_mt", src, mt, labels=None)
        self.assertEqual(build_messages_xml_mt(src, mt), expected)

    def test_conversion_matches_sft_and_round_trips(self):
        tokens = "eins zwei drei vier fünf".split()
        labels = "OK BAD BAD OK BAD".split()
        annotated = labels_to_xml(tokens, labels)
        self.assertEqual(annotated, self.sft.labels_to_xml(tokens, labels))
        parsed, stats = xml_to_labels(annotated, tokens)
        self.assertEqual(parsed, labels)
        self.assertTrue(stats["copy_exact"])
        self.assertTrue(stats["tags_balanced"])

    def test_tags_without_surrounding_spaces_are_accepted(self):
        tokens = "eins zwei drei".split()
        labels, stats = xml_to_labels("eins <e>zwei</e> drei", tokens)
        self.assertEqual(labels, ["OK", "BAD", "OK"])
        self.assertTrue(stats["copy_exact"])
        self.assertTrue(stats["tags_balanced"])

    def test_first_nonempty_line_matches_sft_behavior(self):
        text = "\n  eins <e> zwei </e>\nrepeated output"
        self.assertEqual(first_nonempty_line(text), "eins <e> zwei </e>")


class PreprocessTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "src": "one two three four",
            "mt": "eins zwei drei vier",
            "labels": "OK BAD BAD OK",
        }

    def test_xml_record_uses_xml_prompt_and_ground_truth(self):
        record = to_verl_record(self.row, "train", 7, output_format="xml_mt")
        self.assertEqual(record["prompt"], build_messages_xml_mt(self.row["src"], self.row["mt"]))
        self.assertEqual(record["reward_model"]["ground_truth"], "eins <e> zwei drei </e> vier")
        self.assertEqual(record["extra_info"]["output_format"], "xml_mt")
        self.assertEqual(record["extra_info"]["num_words"], 4)

    def test_labels_record_remains_supported(self):
        record = to_verl_record(self.row, "train", 7, output_format="labels")
        self.assertEqual(record["reward_model"]["ground_truth"], self.row["labels"])
        self.assertTrue(record["prompt"][1]["content"].endswith("Labels:"))

    def test_invalid_row_fails_before_writing_parquet(self):
        bad_row = dict(self.row, labels="OK BAD")
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            to_verl_record(bad_row, "train", 7, output_format="xml_mt")


class RewardTest(unittest.TestCase):
    XML_GT = "eins <e> zwei drei </e> vier"
    XML_EXTRA = {
        "mt": "eins zwei drei vier",
        "num_words": 4,
        "output_format": "xml_mt",
    }
    COMMON = {
        "data_source": "wmt22_en_de_qe",
        "metric": "token_mix",
        "w_bad": 2.5,
        "w_ok": 1.0,
        "token_mix_weight": 0.8,
        "bad_f1_weight": 0.2,
    }

    def xml_score(self, solution: str):
        return compute_score(
            solution_str=solution,
            ground_truth=self.XML_GT,
            extra_info=self.XML_EXTRA,
            output_format="xml_mt",
            **self.COMMON,
        )

    def test_perfect_xml_gets_full_reward(self):
        result = self.xml_score(self.XML_GT)
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertEqual(result["metric_xml_copy_exact"], 1.0)
        self.assertEqual(result["metric_xml_tags_balanced"], 1.0)

    def test_semantically_wrong_but_well_formed_xml_has_lower_reward(self):
        result = self.xml_score("eins zwei drei vier")
        self.assertLess(result["score"], 1.0)
        self.assertEqual(result["metric_format_factor"], 1.0)

    def test_copy_change_gets_format_penalty(self):
        result = self.xml_score("eins <e> ZWEI drei </e> vier")
        self.assertEqual(result["metric_xml_copy_exact"], 0.0)
        self.assertAlmostEqual(result["metric_format_factor"], 0.25)
        self.assertLessEqual(result["score"], 0.25)

    def test_unbalanced_tags_get_format_penalty(self):
        result = self.xml_score("eins <e> zwei drei vier")
        self.assertEqual(result["metric_xml_tags_balanced"], 0.0)
        self.assertAlmostEqual(result["metric_format_factor"], 0.25)

    def test_all_ok_xml_without_tags_can_get_full_reward(self):
        result = compute_score(
            data_source="wmt22_en_de_qe",
            solution_str="eins zwei",
            ground_truth="eins zwei",
            extra_info={"mt": "eins zwei", "num_words": 2, "output_format": "xml_mt"},
            output_format="xml_mt",
            metric="token_mix",
        )
        self.assertAlmostEqual(result["score"], 1.0)

    def test_reward_data_format_mismatch_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "reward/data format mismatch"):
            compute_score(
                data_source="wmt22_en_de_qe",
                solution_str=self.XML_GT,
                ground_truth=self.XML_GT,
                extra_info=self.XML_EXTRA,
                output_format="labels",
            )

    def test_existing_labels_reward_still_gets_full_score(self):
        result = compute_score(
            data_source="wmt22_en_de_qe",
            solution_str="OK BAD BAD OK",
            ground_truth="OK BAD BAD OK",
            extra_info={"num_words": 4, "output_format": "labels"},
            output_format="labels",
            metric="token_mix",
        )
        self.assertAlmostEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
