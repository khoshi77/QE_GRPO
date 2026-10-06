---
name: paper-review
description: Rigorous internal pre-submission review of authors' NLP and Computational Linguistics manuscripts for ARR, ACL, EMNLP, NAACL, EACL, and Findings. Use for ACL/ARR readiness, reviewer weaknesses, submission preflight, claim-evidence consistency, experiment validity, or reproducibility audits. Excludes ordinary sentence edits, translation, basic LaTeX help, typos, and academic English polishing without scientific review.
---

# Paper Review

Help authors improve their own manuscript before submission through demanding, fair scientific review. This is an internal assessment, not an official review of another author's confidential submission or a reliable decision forecast.

## Choose scope and mode

Honor the requested scope. Default to `review` with standard depth. A section-only or reproducibility-only request does not authorize conclusions about the whole manuscript's readiness.

| Invocation | Scope | References to load |
| --- | --- | --- |
| `preflight` | Submission validity and genuine desk-rejection risks | [preflight.md](references/preflight.md) |
| `review` / `standard review` | Full manuscript-level scientific review | [arr-review.md](references/arr-review.md), [review-output.md](references/review-output.md) |
| `quick review` | Review mode focused on approximately 3–7 highest-impact issues; no exhaustive line editing | Same review references, applied selectively |
| `deep review` | Standard review plus tracing important numerical and methodological claims to available repository evidence | Same review references plus [reproducibility.md](references/reproducibility.md) |

Load [experimental-nlp.md](references/experimental-nlp.md) for experimental checks or additional experiment requests, [reproducibility.md](references/reproducibility.md) for reproduction or artifact checks, and [policy.md](references/policy.md) whenever current ARR criteria or submission policies matter. Do not load unrelated detailed checklists.

## Review contract

Understand the research question, method, contribution, evidence, and conclusion before criticizing. State the inspected files and unavailable materials. Do not manufacture weaknesses or strengths. Prefer a few consequential, well-supported findings over a long list of preferences.

Never invent results, dataset statistics, hyperparameters, implementation details, citations, or significance. For unavailable evidence say: `Cannot verify from the available artifacts.` Distinguish demonstrated flaws, missing information, and unresolved uncertainty; an unknown is not proof of invalidity.

Tie every issue to a location, evidence, consequence, and concrete action. Separate soundness and reproducibility from novelty, significance, and subjective excitement. Narrow scope, honest limitations, or limited benchmark breadth alone do not justify a low soundness judgment. Do not demand unrealistic experiment scale; follow the experimental reference for any additional experiment request.

## Shared severity

Use these labels in every mode; do not create a separate preflight scheme.

| Severity | Meaning |
| --- | --- |
| `BLOCKER` | Potential desk rejection, serious policy violation, invalid central experiment, or a fundamentally unreviewable or scientifically invalid submission. Reserve for genuinely serious, evidenced risks; do not turn uncertain policy interpretations into blockers. |
| `MAJOR` | Could materially change the acceptance recommendation or invalidate an important central claim. |
| `MODERATE` | Important weakness, unlikely by itself to invalidate the paper. |
| `MINOR` | Clarity, presentation, a small completeness issue, or a reproduction detail with limited consequence. |
| `OPTIONAL` | Useful improvement, unnecessary for submission or acceptance. |
| `PASS` | Actually checked aspect with no meaningful issue found; never use for uninspected or unavailable material. |

Keep severity separate from evidence status and internal readiness. Use the output template for the selected mode, or the user's requested format.

## Files, citations, and other skills

Review without modifying manuscript, bibliography, or experiment files by default. Findings from other artifacts never authorize silently replacing manuscript values.

For detailed reference verification, use `citation-check` if available, preserving this review's read-only scope; report unresolved needs as `Citation needed` rather than inserting markers or editing BibTeX. Otherwise report the limits of verification. Use literature-review capabilities selectively for unresolved novelty or critical related work. Use paper-writing only when the user requests revisions.

If revisions are explicitly requested, preserve scientific meaning and verified numbers; do not strengthen unsupported claims, add unverified citations, hide scientific problems, or remove meaningful limitations to improve appearances. Flag unresolved issues. Say when a new experiment is needed rather than suggesting prose can repair the scientific problem.
