# Submission preflight

Prioritize submission validity, not scientific rewriting. Read [policy.md](policy.md) for all current requirements. Establish venue, cycle, submission stage, paper type, and track before certifying a venue-specific check.

## Checks

- **Format:** Inspect available source and rendered PDF for the applicable ACL template, page budget and treatment of references/limitations/appendices, margins, fonts, layout, and required sections. Source inspection alone cannot certify rendered layout; a PDF cannot establish every source-level detail. Report what was actually inspected and what remains unchecked.
- **Anonymity:** Inspect manuscript, appendix, supplement, acknowledgments, self-references, filenames/PDF metadata, links, repositories, and visible artifact contents for accidental identifying information where applicable. Check linked repositories rather than assuming an anonymous URL hides authors. Apply verified rules; do not invent a preprint embargo.
- **Limitations and Responsible NLP:** Check the applicable requirement for a `Limitations` section, meaningful paper-specific limitations, ethical disclosures, current checklist completion and consistency, evidence locators/justifications, and applicable generative AI disclosure rules. Do not equate a negative or N/A checklist answer with noncompliance without the governing rule and context.
- **Submission package:** Check required documents and form fields, supplementary-file references and availability, submission eligibility/scope, originality/overlap or dual-submission declarations, and cycle-specific registration/service obligations only where current rules and relevant evidence are available. Do not infer that administrative checks passed from manuscript inspection.
- **Integrity and reviewability:** Inspect broken cross-references, undefined citations, missing sections, unreadable tables/figures, and discrepancies among manuscript, appendix, and supplement. Compare abstract/conclusion claims with the body; flag unsupported central assertions without turning preflight into a full review. Consult the experimental audit only when a consequential inconsistency needs it.

Check compilation logs or reference definitions when available; do not claim a build succeeded without observing one. If inspecting rendered PDFs requires unavailable tools, record that limitation.

## Output

Use this structure unless the user requests another format:

```markdown
# ACL/ARR Submission Preflight

## Blocking risks
## Major risks
## Moderate risks
## Minor issues
## Passed checks
## Items that could not be verified
```

Map Blocking/Major/Moderate/Minor to `BLOCKER`/`MAJOR`/`MODERATE`/`MINOR`; mark passed checks `PASS`. Put `OPTIONAL` suggestions at the end of Minor issues, explicitly labeled, only if useful. Empty risk sections may say no issue was identified within the inspected scope.

For each policy or formatting concern, give severity, the requirement, current official verification status with source/date, manuscript/repository location and evidence, and a concrete correction. Label general advice and skill recommendations as such. Put unresolved checks in Items that could not be verified, never Passed checks. An unavailable checklist or PDF is not itself proof of a violation. Do not convert an uncertain interpretation into definite desk rejection.
