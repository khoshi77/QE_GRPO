# ARR-oriented scientific review

Use for quick, standard, or deep scientific review. Apply the shared severity and policy workflow in the entrypoint; use the report structure in [review-output.md](review-output.md).

## Understand and assess

Read the relevant manuscript, tables, figures, appendix, and bibliography before forming the assessment. In standard review inspect the available manuscript as a whole; in quick or scoped review state coverage limits. Summarize the research question, proposed approach, actual contribution, experimental evidence, and conclusion in terms precise enough to expose likely reviewer misunderstanding. Do not merely paraphrase the abstract.

Identify substantive strengths in methodology, empirical findings, analysis, resources, technical insight, clarity, or practical value to a narrow or broad NLP community. If none can be substantiated, say so rather than inventing balance.

Prioritize central unsupported claims, flawed inference, unfair comparisons, critical missing baselines, inadequate evaluation, confounds, leakage, contamination, invalid statistics, unclear contribution, insufficient method description, or inability to reproduce a central result. Use the experimental and reproduction references for detailed checks. Treat terminology, presentation, and small omissions according to their actual consequence; a stylistic preference is not a major scientific flaw.

## Assessment dimensions

Report applicable dimensions from the current verified ARR form; otherwise use qualitative judgments covering:

- **Soundness:** Explicit claims, experiments that test them, valid methodology, fair comparisons, and conclusions supported by evidence. Tie adverse judgments to concrete technical problems.
- **Excitement / significance / interest:** Separately assess novelty, usefulness, insight, impact, and ACL relevance. A narrow subfield contribution can be valuable.
- **Overall assessment:** Explain the drivers and keep validity separate from perceived novelty or excitement. Apply a numeric overall recommendation only when its current scale is verified.
- **Reviewer confidence:** Ground it in actual manuscript coverage, appendix/supplement availability, related-work verification, checked mathematics, and inspected implementation/configs/results. State what would increase confidence; certainty of tone is not evidence of high confidence. Use the official scale only if verified.
- **Reproducibility:** Whether another researcher can reproduce the main result, with the detailed audit in [reproducibility.md](reproducibility.md).
- **Limitations and societal impact:** Evaluate relevant untested languages, domains, datasets, model families/sizes, cost, data assumptions, contamination, generalization, deployment, and failure modes. Generic boilerplate is insufficient, but acknowledgment of real limitations is not itself a weakness.
- **Ethics / Responsible NLP:** Identify issues materially connected to the work. Distinguish actual ethical concern, disclosure/checklist issue, methodological limitation, and purely hypothetical harm. Check relevant current Responsible NLP requirements through the policy workflow; do not invent ethical objections.
- **Dataset/software contribution, only when meaningful:** For introduced or promised datasets, annotations, software, models, benchmarks, or evaluation tools, assess usefulness, documentation, accessibility, license, data statements, completeness, reproducibility, reuse, and consistency with contribution claims. Use applicable current form fields; omit this assessment for papers without an artifact contribution.

## Citation coverage, correctness, and novelty

Coverage: Check references for prior methods, dataset descriptions, known findings, historical statements, and benchmark state-of-the-art claims. Authors' own observations do not inherently need outside citations.

Correctness: When a source can be inspected, check that it supports the whole attributed statement. Flag mischaracterization, partial support, incorrect attribution, visible metadata errors, and unsupported state-of-the-art claims. Follow the entrypoint's citation-skill delegation and read-only boundary; never propose an unverified paper as a factual reference. Distinguish unchecked sources from demonstrably incorrect ones.

Classify the contribution as a method, application, empirical finding, analysis, resource, combination, or engineering contribution, as appropriate. Existing components do not automatically make their combination or resulting insight unoriginal. Separate novelty of method, finding, and application. When closest related work is unverified, say: `Novelty assessment is provisional because the nearest related work has not been independently verified.`

Discuss main-conference versus Findings concerns only when useful and grounded in current official guidance. Do not invent fixed thresholds, equate narrow scope with Findings, treat limited novelty as automatic rejection, or let novelty excuse unsound methodology. Keep soundness, reproducibility, novelty, impact, and excitement distinguishable. Internal readiness is not a prediction of acceptance.
