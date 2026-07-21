# JOSS-ready paper draft

Revised JOSS manuscript addressing the blocking review items. **This folder does not replace** `paper/paper.md` until you explicitly merge it.

## Contents

| File | Role |
|---|---|
| `paper.md` | JOSS manuscript (Pandoc/Markdown) |
| `paper.bib` | Bibliography |
| `structure.png` | Architecture figure referenced by the paper |

## What changed vs `paper/paper.md`

1. **Added** required `# Research impact statement`
2. **Rewrote** `# Software Design` around trade-offs (pure-Python core vs optional Bayes; trial-array API; validation contract) instead of an API laundry list
3. **Softened** absolute “only package” language → “to our knowledge, the first…”
4. **Qualified** full-benchmark / parity claims; points readers to documented exceptions
5. **Fixed** AI disclosure: “the author” → “the authors”
6. **Trimmed** Bayesian method catalogue (detail belongs in docs, not JOSS)

## Still needed before actual JOSS submission (repo, not this file)

These are outside the manuscript but reviewers will check them:

- [ ] Tagged GitHub release + preferably PyPI upload
- [ ] Optionally add `CODE_OF_CONDUCT.md` and `CITATION.cff`
- [ ] Fix remaining MATLAB↔Python numeric gaps (meta-noise; SDT-expected proportions) *or* keep them documented as here
- [ ] When accepted, add archive DOI (Zenodo) to the paper

## Preview / compile

JOSS compiles `paper.md` via their draft system. Locally you can check length/structure with:

```bash
# word count of body (approx.)
wc -w paper/joss_ready/paper.md
```

Or open a draft preview at https://whedon.theoj.org (paste the GitHub path to this file once pushed).

## Promote to main paper path

When ready to replace the submission manuscript:

```bash
cp paper/joss_ready/paper.md paper/paper.md
# keep bib/figure in sync if edited further
cp paper/joss_ready/paper.bib paper/paper.bib
cp paper/joss_ready/structure.png paper/structure.png
```
