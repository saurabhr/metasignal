# Validated JOSS manuscript package

This folder contains a new validation-focused manuscript. It does not overwrite
`paper/paper.md` or `paper/joss_ready/`.

## Manuscript

- `metasignal_joss_validated.md` — editable source
- `metasignal_joss_validated.pdf` — compiled manuscript

## Numerical corrections report

- `MATLAB_PYTHON_CORRECTIONS_REPORT.md` — editable source
- `MATLAB_PYTHON_CORRECTIONS_REPORT.pdf` — compiled report

## Assets

- `paper.bib` — bibliography
- `structure.png` — software architecture
- `validation_main.png` — paper/MATLAB/Python validation figure

## Important interpretation

The 18 non-model-based measures now match MATLAB to numerical precision, and the
effect profiles (task, bias, response) correlate at *r* = 1.000. Three confirmed
Python errors were corrected (meta-noise search/interpolation, meta-noise
criteria boundary handling, SDT-expected proportions), the meta-uncertainty
optimizer was stabilized, and the reliability/precision caches were rebuilt to
the paper protocol (MATLAB↔Python *r* = 0.992 / 0.999 / 0.996). The manuscript
still does **not** claim universal bit-for-bit identity: bounded maximum-likelihood
optimizer variation in the meta-d' family and a few low-information model fits
remain documented limitations.

## Rebuild PDFs

```bash
cd paper/joss_validation_ready

pandoc metasignal_joss_validated.md \
  --standalone --citeproc --bibliography=paper.bib \
  --resource-path=. --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -o metasignal_joss_validated.pdf

pandoc MATLAB_PYTHON_CORRECTIONS_REPORT.md \
  --standalone --resource-path=. --pdf-engine=xelatex \
  -V geometry:margin=0.8in \
  -o MATLAB_PYTHON_CORRECTIONS_REPORT.pdf
```
