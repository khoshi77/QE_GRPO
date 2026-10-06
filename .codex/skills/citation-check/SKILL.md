---
name: citation-check
description: Verify academic references and their support for specific claims before citing them in NLP research papers. Use when adding or auditing citations, checking paper metadata, or creating or correcting BibTeX entries.
---

# Citation Check

Never invent citations, bibliographic fields, identifiers, or evidence. A remembered paper, an existing BibTeX entry, and a search result are candidates for verification, not proof.

## Identify the claim and reference

Read the sentence being cited and enough surrounding text to establish its meaning. Inspect relevant existing bibliography entries and citation commands before making changes. Treat bibliographic accuracy and support for the claim as separate checks; both must pass before adding a citation or BibTeX entry.

## Verify existence and metadata

Use live source lookup when available. Prefer [ACL Anthology](https://aclanthology.org/), [DOI resolution](https://doi.org/) and [Crossref](https://search.crossref.org/), publisher pages, [arXiv](https://arxiv.org/), and [OpenAlex](https://openalex.org/). Use search engines to locate these records, then open the records themselves. Prefer the official publication record for metadata; use OpenAlex for discovery and corroboration.

Confirm that a matching paper exists and verify:

- Exact title, allowing only harmless formatting differences.
- Complete author list and order, preserving names and diacritics.
- Publication year of the version being cited.
- Conference, proceedings, or journal; identify an arXiv-only work as a preprint without assigning it a peer-reviewed venue.
- DOI, ACL Anthology ID, or arXiv ID, checking that the identifier resolves to the same paper. If none exists, verify a stable publisher URL or OpenAlex record rather than inventing an identifier.

Resolve discrepancies against the relevant publication record and paper. Keep preprint and published versions distinct: do not combine one version's title, year, venue, or identifier with another's. A preprint posting year or DOI registration date is not automatically the publication year. Search snippets and secondary bibliographies alone are insufficient. If essential metadata remains unresolved, verification fails.

## Verify support for the claim

Read the paper's relevant passages and, for empirical claims, the applicable tables, figures, and experimental conditions. Record a section, page, table, or passage locator that shows what the paper actually establishes.

Check the exact scope of the claim: method, task, dataset, languages, comparisons, assumptions, and strength of the conclusion. Topical similarity does not establish support. Distinguish the paper's own findings from related work it cites; follow the original source when attributing a finding. An abstract can support a claim it states explicitly, but cannot verify unreported experimental details or a stronger generalization.

If support is partial, state the limitation and narrow the claim only when the editing task authorizes that change. Otherwise leave the claim unresolved. Do not treat an inaccessible full text as verified support when the accessible material does not establish the claim.

## Update citations and BibTeX

Only add a BibTeX entry after both checks pass. Prefer official BibTeX exports, then compare their fields with the verified record. Include only verified fields; do not guess page ranges, issue numbers, URLs, or DOIs.

Preserve existing BibTeX keys exactly when correcting metadata or reusing a verified reference. Reuse an existing entry for the same paper rather than duplicating it. For a new paper, follow the repository's key convention and check for collisions; never overwrite a different paper under an existing key. Preserve citation commands and bibliography formatting where possible.

## Handle failures and report evidence

If existence, essential metadata, or support for the claim cannot be verified, use the literal marker `[CITATION NEEDED]` at the affected claim and explain what remains unresolved. Missing tools or inaccessible sources are verification failures, not permission to fill gaps from memory. Do not add an unverified BibTeX entry or silently delete or rename an existing one. When an existing citation fails, flag it and mark the claim rather than presenting it as verified.

Provide a concise verification summary for each checked reference: existing or proposed key, status, source links, verified metadata, and the evidence locator with any limits on claim support. Report bibliographic verification separately from claim support when their outcomes differ. A verified paper that does not support the claim still requires `[CITATION NEEDED]` for that claim.
