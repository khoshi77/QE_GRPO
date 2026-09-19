from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_qe_eos import inspect_model, prepare_model  # noqa: E402


class QeEosTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def checkpoint(self, name="base", chat=False):
        path = self.root / name
        vocab = {"[UNK]": 0, "<|endoftext|>": 1, "<|im_end|>": 2,
                 "OK": 3, "BAD": 4, "<e>": 5, "</e>": 6}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]",
            eos_token="<|im_end|>" if chat else "<|endoftext|>",
            pad_token="<|endoftext|>",
            additional_special_tokens=["<|im_end|>"],
        )
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + '<|im_end|>\\n' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
        )
        tokenizer.save_pretrained(path)
        (path / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "eos_token_id": 2 if chat else 1,
            "pad_token_id": 1, "vocab_size": len(vocab),
        }))
        (path / "generation_config.json").write_text(json.dumps({
            "eos_token_id": [2, 1] if chat else 1, "max_new_tokens": 448,
        }))
        (path / "model.safetensors").write_bytes(b"test weights; never loaded")
        return path

    def test_base_metadata_weights_padding_and_prompts(self):
        source = self.checkpoint()
        before = {f.name: f.read_bytes() for f in source.iterdir()}
        original = AutoTokenizer.from_pretrained(source, local_files_only=True)
        self.assertTrue(inspect_model(source)["needs_fix"])
        output = prepare_model(source, self.root / "prepared")
        fixed = AutoTokenizer.from_pretrained(output, local_files_only=True)
        self.assertEqual(fixed.eos_token_id, 2)
        self.assertEqual(fixed.pad_token_id, original.pad_token_id)
        self.assertEqual(AutoConfig.from_pretrained(output).eos_token_id, 2)
        self.assertEqual(GenerationConfig.from_pretrained(output).eos_token_id, [2, 1])
        self.assertFalse(inspect_model(output)["needs_fix"])
        self.assertTrue((output / "model.safetensors").is_symlink())
        self.assertEqual((output / "model.safetensors").resolve(), source / "model.safetensors")
        self.assertFalse((output / "tokenizer_config.json").is_symlink())
        self.assertTrue((output / "qe_eos_manifest.json").is_file())
        for f in source.iterdir():
            self.assertEqual(f.read_bytes(), before[f.name], f.name)
        for response in ["OK BAD", "eins <e> zwei </e>"]:
            messages = [{"role": "user", "content": "Tag this."},
                        {"role": "assistant", "content": response}]
            for tok in [original, fixed]:
                rendered = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)
                self.assertIn(2, tok.encode(rendered, add_special_tokens=False))
            self.assertEqual(original.apply_chat_template(messages), fixed.apply_chat_template(messages))
            self.assertEqual(
                original.apply_chat_template(messages[:1], add_generation_prompt=True),
                fixed.apply_chat_template(messages[:1], add_generation_prompt=True),
            )

    def test_chat_is_noop_and_corrected_model_can_be_reused(self):
        chat = self.checkpoint("chat", chat=True)
        self.assertEqual(prepare_model(chat, self.root / "unused"), chat)
        self.assertFalse((self.root / "unused").exists())
        prepared = prepare_model(self.checkpoint(), self.root / "prepared")
        self.assertEqual(prepare_model(prepared, self.root / "unused"), prepared)

    def test_refuse_existing_or_nested_destination(self):
        source = self.checkpoint()
        with self.assertRaises(FileExistsError):
            prepare_model(source, source)
        with self.assertRaises(ValueError):
            prepare_model(source, source / "nested")
        link = self.root / "link"
        link.symlink_to(source, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            prepare_model(source, link)

    def test_fail_on_wrong_template_or_model_family(self):
        source = self.checkpoint()
        (source / "chat_template.jinja").write_text("{{ messages[-1]['content'] }}<|endoftext|>")
        with self.assertRaisesRegex(ValueError, "template"):
            prepare_model(source, self.root / "prepared")
        self.assertFalse((self.root / "prepared").exists())
        (source / "config.json").write_text('{"model_type": "llama"}')
        with self.assertRaisesRegex(ValueError, "Qwen3"):
            inspect_model(source)

    def test_generation_only_mismatch_and_missing_generation_file(self):
        source = self.checkpoint(chat=True)
        (source / "generation_config.json").write_text('{"eos_token_id": 1}')
        self.assertTrue(inspect_model(source)["needs_fix"])
        prepared = prepare_model(source, self.root / "fixed_generation")
        self.assertEqual(GenerationConfig.from_pretrained(prepared).eos_token_id, [2, 1])
        base = self.checkpoint("base_without_generation")
        (base / "generation_config.json").unlink()
        prepared = prepare_model(base, self.root / "new_generation")
        self.assertEqual(GenerationConfig.from_pretrained(prepared).eos_token_id, [2, 1])

    def test_hf_generation_stops_at_assistant_end(self):
        # CPU-only regression with forced logits: identical weights and tokens,
        # only the EOS metadata changes. No 8B model or GPU is needed.
        import torch
        from transformers import LogitsProcessor, Qwen3Config, Qwen3ForCausalLM

        class ForceResponse(LogitsProcessor):
            def __call__(self, input_ids, scores):
                token = {1: 3, 2: 2}.get(input_ids.shape[1], 4)
                scores.fill_(-float("inf"))
                scores[:, token] = 0
                return scores

        config = Qwen3Config(
            vocab_size=7, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, bos_token_id=None, eos_token_id=1, pad_token_id=1,
        )
        model = Qwen3ForCausalLM(config).eval()
        inputs = torch.tensor([[3]])
        kwargs = dict(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=5, do_sample=False, logits_processor=[ForceResponse()],
        )
        source = self.checkpoint()
        original = AutoTokenizer.from_pretrained(source)
        fixed = AutoTokenizer.from_pretrained(prepare_model(source, self.root / "prepared"))
        with torch.no_grad():
            old = model.generate(**kwargs, eos_token_id=original.eos_token_id)
            new = model.generate(**kwargs, eos_token_id=fixed.eos_token_id)
        self.assertEqual(old[0].tolist(), [3, 3, 2, 4, 4, 4])
        self.assertEqual(new[0].tolist(), [3, 3, 2])

    def test_diagnostic_keeps_ignored_eos_and_trailing_tokens(self):
        from diagnose_rollout import generation_record

        tok = AutoTokenizer.from_pretrained(self.checkpoint())
        old = generation_record([3, 2, 4, 4], tok, 4, "length")
        self.assertEqual(old["im_end_positions"], [1])
        self.assertEqual(old["tokens_after_first_im_end"], 2)
        self.assertTrue(old["reached_length_limit"])
        self.assertIn("<|im_end|>", old["raw_text"])
        fixed = generation_record([3, 2], tok, 4, "stop")
        self.assertEqual(fixed["tokens_after_first_im_end"], 0)
        self.assertFalse(fixed["reached_length_limit"])


if __name__ == "__main__":
    unittest.main()
