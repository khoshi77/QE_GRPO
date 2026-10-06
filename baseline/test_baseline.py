"""Offline integration tests. No API key or network required."""
import argparse
import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_baseline as baseline
from qe_xml_utils import build_messages_xml_mt, labels_to_xml


def fixture(fmt):
    return {"prompt": build_messages_xml_mt("A cat", "Eine Katze"),
            "data_source": "qe", "reward_model": {"ground_truth":
                "OK BAD" if fmt == "labels" else labels_to_xml(["Eine", "Katze"], ["OK", "BAD"])},
            "extra_info": {"src": "A cat", "mt": "Eine Katze", "num_words": 2, "output_format": fmt}}


class FakeResponse:
    def __init__(self, text, status="completed"):
        self.output_text = text
        self.status = status

    def model_dump(self, **kwargs):
        return {"id": "resp_test", "status": self.status, "model": "test-model", "output": [],
                "incomplete_details": {"reason": "max_output_tokens"} if self.status == "incomplete" else None,
                "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22}}


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value


class BaselineTests(unittest.TestCase):
    def config(self, directory, fmt="labels", profile=None):
        args = argparse.Namespace(config="config.json", model=None, format=fmt, split="dev", profile=profile,
                                  reasoning_effort=None, samples=None, max_output_tokens=None,
                                  output_dir=str(directory), temperature="profile", top_p="profile", limit=None, input=None)
        return baseline.resolve_config(args)

    def flags(self, **kwargs):
        return SimpleNamespace(dry_run=False, resume=False, evaluate_only=False, **kwargs)

    def test_profiles(self):
        cfg = self.config("results")
        self.assertEqual((cfg["temperature"], cfg["top_p"], cfg["samples"], cfg["max_output_tokens"]),
                         (0, 1, 1, 256))
        self.assertEqual(self.config("results", "xml")["max_output_tokens"], 448)
        rollout = self.config("results", profile="rollout")
        self.assertEqual((rollout["temperature"], rollout["top_p"], rollout["samples"]), (0.5, 0.9, 8))
        cfg.update(reasoning_effort="omit", temperature=None, top_p=None)
        params = baseline.request_parameters(cfg)
        self.assertNotIn("reasoning", params)
        self.assertNotIn("temperature", params)
        self.assertNotIn("top_p", params)

    def test_both_formats_and_evaluate_only(self):
        for fmt, output in (("labels", "OK BAD"), ("xml_mt", "Eine <e>Katze</e>")):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as temp:
                cfg = self.config(temp, fmt)
                rows = [fixture(fmt)]
                client = FakeClient([FakeResponse(output)])
                baseline.run_split(rows, cfg, "dev", self.flags(), client)
                self.assertEqual(client.requests[0]["input"], rows[0]["prompt"])
                self.assertNotIn("ground_truth", client.requests[0])
                directory = Path(temp) / cfg["model"] / fmt / "eval" / "dev"
                metrics = json.loads((directory / "metrics.json").read_text())
                self.assertEqual(metrics["corpus_metrics"]["mcc"], 1)
                self.assertEqual(metrics["sentence_mean_reward_metrics"]["score"], 1)
                flags = self.flags()
                flags.evaluate_only = True
                with patch.object(baseline, "get_client", side_effect=AssertionError("must not call API")):
                    baseline.run_split(rows, cfg, "dev", flags)

    def test_omit_output_limit_in_config_and_request(self):
        with tempfile.TemporaryDirectory() as temp:
            saved = json.loads((baseline.HERE / "config.json").read_text())
            saved["max_output_tokens"] = "omit"
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(saved))
            args = argparse.Namespace(config=str(path), model=None, format="labels", split="dev", profile=None,
                                      reasoning_effort=None, samples=None, max_output_tokens=None,
                                      output_dir=temp, temperature=None, top_p=None, limit=None, input=None)
            cfg = baseline.resolve_config(args)
            self.assertEqual(cfg["max_output_tokens"], "omit")
            client = FakeClient([FakeResponse("OK BAD")])
            baseline.run_split([fixture("labels")], cfg, "dev", self.flags(), client)
            self.assertNotIn("max_output_tokens", client.requests[0])
            self.assertEqual(baseline.output_token_limit("omit"), "omit")
            self.assertEqual(baseline.output_token_limit("2048"), 2048)

    def test_resume_after_failure_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.config(temp)
            rows = [fixture("labels"), fixture("labels")]
            client = FakeClient([FakeResponse("OK BAD"), RuntimeError("simulated connection failure")])
            with self.assertRaises(RuntimeError):
                baseline.run_split(rows, cfg, "dev", self.flags(), client)
            directory = Path(temp) / cfg["model"] / "labels/eval/dev"
            with (directory / "responses.jsonl").open("ab") as stream:
                stream.write(b'{"row_id":')
            flags = self.flags()
            flags.resume = True
            resumed = FakeClient([FakeResponse("OK BAD")])
            baseline.run_split(rows, cfg, "dev", flags, resumed)
            self.assertEqual(len(resumed.requests), 1)
            self.assertEqual(len(baseline.read_cache(directory / "responses.jsonl")), 2)
            cfg["temperature"] = 0.5
            with self.assertRaisesRegex(ValueError, "mismatch"):
                baseline.run_split(rows, cfg, "dev", flags, resumed)

    def test_incomplete_and_multisample(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.config(temp)
            cfg["samples"] = 2
            client = FakeClient([FakeResponse("OK", "incomplete"), FakeResponse("OK BAD")])
            baseline.run_split([fixture("labels")], cfg, "dev", self.flags(), client)
            directory = Path(temp) / cfg["model"] / "labels/eval/dev"
            metrics = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(metrics["status_counts"], {"incomplete": 1, "completed": 1})
            self.assertEqual(metrics["n_outputs"], 2)
            self.assertEqual(metrics["corpus_metrics"]["mcc"], 1)  # short labels pad BAD
            self.assertTrue((directory / "predictions.sample_001.tsv").exists())

    def test_invalid_data_fails_before_api(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl"
            row = fixture("labels")
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "dataset format"):
                baseline.load_records(path, "xml_mt")
            row["reward_model"]["ground_truth"] = "OK"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "gold"):
                baseline.load_records(path, "labels")

    def test_legacy_labels_and_dry_run_without_api(self):
        with tempfile.TemporaryDirectory() as temp:
            row = fixture("labels")
            del row["extra_info"]["output_format"]
            path = Path(temp) / "input.jsonl"
            path.write_text(json.dumps(row) + "\n")
            rows = baseline.load_records(path, "labels")
            flags = self.flags()
            flags.dry_run = True
            with patch.object(baseline, "get_client", side_effect=AssertionError("must not call API")):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    baseline.run_split(rows, self.config(temp), "dev", flags)
            self.assertEqual(json.loads(output.getvalue())["api_requests"], 1)
            self.assertFalse((Path(temp) / "gpt-5.6-luna").exists())

    def test_original_reward_parity(self):
        source = baseline.HERE.parent / "scripts/qe_reward.py"
        if not source.exists():
            self.skipTest("Original project unavailable in standalone copy")
        spec = importlib.util.spec_from_file_location("original_qe_reward", source)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        from qe_reward import compute_score
        for fmt, outputs in (("labels", ["OK BAD", "OK", "blah BAD OK"]),
                             ("xml_mt", ["Eine <e> Katze </e>", "Eine <e> Katze", "changed Katze", ""])):
            row = fixture(fmt)
            for output in outputs:
                args = ("qe", output, row["reward_model"]["ground_truth"], row["extra_info"])
                self.assertEqual(compute_score(*args, output_format=fmt, metric="f1_macro"),
                                 original.compute_score(*args, output_format=fmt, metric="f1_macro"))


if __name__ == "__main__":
    unittest.main()
