"""Pure helpers shared by QE preprocessing and the GRPO XML reward.

The prompt and XML conversion rules intentionally match
``work_SFT/QE_SFT_8B/src/models/data_utils.py``.  Keeping this module free of
third-party imports lets Ray reward workers load it without importing the SFT
training stack.
"""

from __future__ import annotations

import difflib


XML_OPEN = "<e>"
XML_CLOSE = "</e>"

SYSTEM_PROMPT_XML_MT = (
    "You are a machine translation quality estimator. "
    "Given a source sentence and its translation, reproduce the translation exactly, "
    f"wrapping every erroneous span in {XML_OPEN} and {XML_CLOSE} tags. "
    "Do not add, remove, or change any words. "
    "Output only the annotated translation, with no extra text."
)


def build_messages_xml_mt(src: str, mt: str) -> list[dict[str, str]]:
    """Build the inference prompt used by an ``xml_mt`` SFT model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_XML_MT},
        {
            "role": "user",
            "content": (
                f"Source: {src.strip()}\n"
                f"Translation: {mt.strip()}\n\n"
                "Annotated translation:"
            ),
        },
    ]


def labels_to_xml(tokens: list[str], labels: list[str]) -> str:
    """Wrap consecutive BAD spans in ``<e>`` and ``</e>`` tags."""
    out: list[str] = []
    in_span = False
    for token, label in zip(tokens, labels):
        bad = label.upper() == "BAD"
        if bad and not in_span:
            out.append(XML_OPEN)
            in_span = True
        elif not bad and in_span:
            out.append(XML_CLOSE)
            in_span = False
        out.append(token)
    if in_span:
        out.append(XML_CLOSE)
    return " ".join(out)


def xml_to_labels(annotated: str, ref_tokens: list[str]) -> tuple[list[str], dict[str, int | bool]]:
    """Convert an annotated MT string to labels aligned with ``ref_tokens``.

    Tags are recognized with or without surrounding whitespace.  If the model
    changes the copied MT, ``SequenceMatcher`` transfers labels for aligned
    tokens; the caller can use ``copy_exact`` to penalize that invalid format.
    """
    text = annotated.replace(XML_OPEN, f" {XML_OPEN} ").replace(XML_CLOSE, f" {XML_CLOSE} ")
    words: list[str] = []
    word_labels: list[str] = []
    depth = 0
    unbalanced = False
    for token in text.split():
        if token == XML_OPEN:
            depth += 1
        elif token == XML_CLOSE:
            depth -= 1
            if depth < 0:
                unbalanced = True
                depth = 0
        else:
            words.append(token)
            word_labels.append("BAD" if depth > 0 else "OK")
    if depth != 0:
        unbalanced = True

    copy_exact = words == list(ref_tokens)
    if copy_exact:
        labels = word_labels
    else:
        labels = ["OK"] * len(ref_tokens)
        matcher = difflib.SequenceMatcher(a=list(ref_tokens), b=words, autojunk=False)
        for operation, i1, i2, j1, j2 in matcher.get_opcodes():
            if operation in ("equal", "replace"):
                for offset in range(min(i2 - i1, j2 - j1)):
                    labels[i1 + offset] = word_labels[j1 + offset]

    return labels, {
        "copy_exact": copy_exact,
        "tags_balanced": not unbalanced,
        "n_output_words": len(words),
    }


def first_nonempty_line(text: str) -> str:
    """Return the first generated line, matching SFT XML inference behavior."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()
