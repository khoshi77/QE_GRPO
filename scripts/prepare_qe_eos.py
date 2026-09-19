"""Align Qwen3 QE generation EOS with the assistant terminator used by SFT.

The source checkpoint is read-only. If needed, create a new model directory
with corrected JSON metadata and symlinks to the original weights/tokenizer.
Stdout contains only the usable model path; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


ASSISTANT_END = "<|im_end|>"
MANIFEST = "qe_eos_manifest.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(value) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    if any(type(item) is not int or item < 0 for item in values):
        raise ValueError(f"invalid eos_token_id: {value!r}")
    return values


def inspect_model(model_path: Path) -> dict:
    """Validate the actual template/tokenizer and report EOS consistency."""
    source = model_path.resolve(strict=True)
    config = _read_json(source / "config.json")
    if config.get("model_type") != "qwen3":
        raise ValueError("EOS preparation supports Qwen3 QE checkpoints only")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    end_id = tokenizer.convert_tokens_to_ids(ASSISTANT_END)
    if (
        end_id is None
        or end_id not in tokenizer.all_special_ids
        or tokenizer.encode(ASSISTANT_END, add_special_tokens=False) != [end_id]
    ):
        raise ValueError(f"{ASSISTANT_END} must be an existing single special token")

    # Check the SFT template, rather than guessing from a 'Base' folder name.
    marker = "QE_EOS_PROBE_42"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tag this."},
         {"role": "assistant", "content": marker}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    if marker not in rendered or rendered.rsplit(marker, 1)[1].strip() != ASSISTANT_END:
        raise ValueError(f"SFT assistant template does not end in {ASSISTANT_END}")

    generation_path = source / "generation_config.json"
    generation = _read_json(generation_path) if generation_path.exists() else {}
    model_ids = _ids(config.get("eos_token_id"))
    generation_ids = _ids(generation.get("eos_token_id", config.get("eos_token_id")))
    # Retain existing EOS alternatives (e.g. <|endoftext|>) in generation_config.
    desired_ids = list(dict.fromkeys(
        [end_id] + generation_ids + model_ids + _ids(tokenizer.eos_token_id)
    ))
    return {
        "source_model": str(source),
        "assistant_end_token": ASSISTANT_END,
        "assistant_end_token_id": end_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "model_eos_token_id": config.get("eos_token_id"),
        "generation_eos_token_id": generation.get("eos_token_id", config.get("eos_token_id")),
        "pad_token_id": tokenizer.pad_token_id,
        "prepared_generation_eos_token_id": desired_ids,
        "needs_fix": not (
            tokenizer.eos_token_id == end_id
            and end_id in model_ids
            and end_id in generation_ids
        ),
    }


def prepare_model(model_path: Path, output_dir: Path) -> Path:
    """Return the source if already aligned, otherwise create a new model view."""
    report = inspect_model(model_path)
    source = Path(report["source_model"])
    if not report["needs_fix"]:
        print(f"[EOS] already aligned: {source}", file=sys.stderr)
        return source

    output = output_dir.absolute()
    # Refuse overwrite, including a symlink to an existing checkpoint.
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"use a new output directory: {output}")
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(source) or source.is_relative_to(resolved_output):
        raise ValueError("output must be separate from the source checkpoint")

    config = _read_json(source / "config.json")
    config["eos_token_id"] = report["assistant_end_token_id"]
    tokenizer_config = _read_json(source / "tokenizer_config.json")
    tokenizer_config["eos_token"] = ASSISTANT_END
    generation_path = source / "generation_config.json"
    generation = _read_json(generation_path) if generation_path.exists() else {}
    generation["eos_token_id"] = report["prepared_generation_eos_token_id"]
    replacements = {
        "config.json": config,
        "tokenizer_config.json": tokenizer_config,
        "generation_config.json": generation,
    }
    special_map = source / "special_tokens_map.json"
    if special_map.exists():
        mapping = _read_json(special_map)
        mapping["eos_token"] = ASSISTANT_END
        replacements[special_map.name] = mapping

    output.mkdir(parents=True, exist_ok=False)
    for name, data in replacements.items():
        (output / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    for item in source.iterdir():
        if item.name not in replacements and item.name != MANIFEST:
            (output / item.name).symlink_to(item, target_is_directory=item.is_dir())

    # Reload using the same HF entry point as VERL and standalone inference.
    after = inspect_model(output)
    if after["needs_fix"] or after["pad_token_id"] != report["pad_token_id"]:
        raise RuntimeError("prepared EOS metadata failed validation")
    (output / MANIFEST).write_text(
        json.dumps({"before": report, "after": after}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[EOS] {report['tokenizer_eos_token_id']} -> {report['assistant_end_token_id']}; "
        f"generation EOS={after['generation_eos_token_id']}; pad unchanged; source={source}",
        file=sys.stderr,
    )
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="print JSON; do not write files")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(inspect_model(args.model_path), ensure_ascii=False, indent=2))
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required unless --check-only is given")
        print(prepare_model(args.model_path, args.output_dir))


if __name__ == "__main__":
    main()
