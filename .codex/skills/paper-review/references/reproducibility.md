# Reproducibility and repository evidence

Read for reproducibility audits, artifact contributions, and deep review. Inspect only relevant available material; do not launch expensive training, rerun the full study, or change artifacts merely to review them.

## Central-result recipe

Determine whether another researcher could reproduce the central result from the disclosed recipe. Check, where relevant:

- Dataset identity/version, exact train/dev/test splits, preprocessing, and access conditions.
- Model/checkpoint, prompts/templates, inference procedure, decoding, and output handling.
- Hyperparameters, optimizer, learning rate, effective batch size, steps/epochs, seeds, and checkpoint/model selection.
- Evaluation script and metric definition, statistical testing procedure, and necessary intermediate artifacts.
- Hardware and software/framework versions when they could materially affect reproduction.
- Code/data/model availability statements and whether promised artifacts are accessible and sufficient.

In the report classify details as **Sufficiently specified**, **Missing**, or **Ambiguous**, and identify their location: main paper, appendix/supplement, repository, or genuinely unavailable. Repository-only information can support an internal audit while still needing an anonymous reviewer-accessible description or pointer. Do not describe a paper omission as globally unavailable when it is documented elsewhere.

Assess materiality: a missing engineering detail is not automatically `MAJOR`. Explain how missing information could change the central result or prevent reasonable reproduction.

## Deep cross-check

Trace important numerical and methodological claims across manuscript, appendix, figures, tables, bibliography, supplement, results, logs, configs, evaluation scripts, and implementation when available.

As guidance, prioritize raw or authoritative experimental outputs, then finalized evaluation artifacts/tables, configs, implementation, manuscript claims, and informal notes. First establish provenance: split, seed, checkpoint, metric version, units, aggregation, and whether code/config corresponds to the reported run. A current config or code path alone does not prove what ran. Different runs and rounding can explain apparent disagreements.

Assign evidence status to each traced claim:

- **Verified:** Matching authoritative evidence was inspected and supports the claim; specify whether this was an artifact check or independent reproduction.
- **Internally consistent but not independently verified:** Inspected sources agree, but authoritative or independent confirmation is absent.
- **Inconsistent:** Matching sources disagree; show both values or procedures and locators.
- **Unavailable:** Required evidence cannot be accessed or traced; state the missing artifact.

Keep these statuses separate from severity. Report disagreements rather than choosing a source silently or rewriting manuscript values. If evidence is insufficient, use the entrypoint's unavailable-evidence wording. Record actual inspection coverage; never imply logs, supplementary material, mathematics, or code were checked when they were not.
