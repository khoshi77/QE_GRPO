# Experimental NLP and ML checks

Apply only checks relevant to the claims and experimental setup. This covers experimental NLP broadly, including translation, quality estimation, token prediction, multilingual evaluation, SFT, prompting, decoding, and LLM/RL studies.

## Claim–Evidence Audit

For each important claim, identify the claim, supporting experiment, metric/result, and whether that evidence suffices. Include claims from the abstract, introduction, contribution list, results discussion, and conclusion; check that their strength matches the experiment's scope.

| Claim and location | Experiment / metric / result and locator | Assessment | Problem or limit |
| --- | --- | --- | --- |
| Actual manuscript claim | Available supporting evidence | Directly supported / partially supported / unsupported / cannot verify | Concrete gap, if any |

Unsupported means available evidence fails to support the claim. Cannot verify means evidence needed to decide is unavailable or uninspected. Do not conflate these with severity.

## Design, comparison, and metrics

- Check train/dev/test separation, leakage, duplicate examples, contamination, preprocessing, and label construction. Examine whether supervision or reward data expose evaluation labels.
- Compare datasets, splits, preprocessing, supervision, external resources, model size, metrics, and protocol across systems. Extra data, supervision, model scale, sampling, or test-time compute can confound claimed method gains.
- Check that metrics match the claim and are implemented consistently. Inspect aggregation level, averaging, denominators, units, label polarity, token alignment, masking/padding, and thresholds where relevant. For multilingual evaluation, check coverage and aggregation rather than assuming a language average supports every language.
- Do not demand every known baseline. A missing baseline is `MAJOR` only if necessary to interpret the core contribution; name the exact uncertainty it would resolve.
- Check whether ablations or controls isolate the proposed contribution. A larger benchmark inventory does not by itself establish soundness.

## Statistics and selection

Where uncertainty affects a central claim, inspect seeds, variance, confidence intervals, sample size, paired evaluation, significance tests, dev-set tuning, and multiple comparisons. Distinguish numerical improvement, statistically supported improvement, and practical effect size.

Never call a result statistically significant without an appropriate test. Check the experimental unit and dependence structure: tokens or repeated generations from the same example may not be independent observations. Do not require significance tests universally; explain why uncertainty matters to the particular conclusion.

Check checkpoint/model selection, early stopping, dev-versus-test selection, and comparable tuning effort. Determine whether reported checkpoints were chosen after inspecting test performance. Flag test-driven selection, or unclear selection that could bias a result, according to its consequence.

## LLM, prompting, and decoding

When applicable, inspect exact model/version/checkpoint, size, base versus instruct/chat model, prompt/template, system prompt, decoding parameters, temperature, top-p, sampling, generations per example, post-processing, constrained decoding, and inference cost. For external APIs or continuously updated proprietary models, check model version/date and reproducibility limits. Identify contamination evidence or unresolved risk, proprietary access constraints, and any assumptions about hidden reasoning. Do not infer unreported settings or inaccessible reasoning traces as facts.

## RL, RLHF, GRPO, PPO, and preference optimization

When applicable, inspect reward definition, aggregation level and sparsity, advantage computation and normalization, KL treatment, sampling policy, rollout configuration, samples per prompt, policy/reference relationship, policy-loss formulation, clipping, training/inference mismatch, and alignment of objective with evaluation metric.

Check whether gains can be attributed to optimization rather than altered decoding, output/prediction distribution, checkpoint choice, post-processing, extra sampling, or inference settings. A distribution change may be an outcome of learning; establish what evidence separates it from the stronger mechanism claim. Missing documentation is an uncertainty, not permission to reconstruct implementation from assumptions.

## Additional experiments

For each experiment request, supply the reviewer concern, priority (essential / strongly recommended / optional), and conclusion that would change with its outcome. Prefer a direct test of a central uncertainty or confound. Where existing artifacts or a claim correction can resolve the issue, say so rather than reflexively requesting new runs.
