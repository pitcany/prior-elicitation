# Corrected manuscript checkpoint

This checkpoint preserves the proof-corrected manuscript from the September 19,
2026 revision discussed in “Review proofs,” before the subsequent rewrite for
another journal. The two LaTeX files at the repository root are exact copies
from `elicitation_arxiv_corrected.tar.gz`:

- `elicitation_arxiv.tex`: article wrapper, abstract, code availability, and
  inline bibliography; the Acknowledgments section has been removed.
- `elicitation_body.tex`: corrected main text, proofs, and appendix.

The original author footnote describing the origins of the work is retained.
The separate Bayesian Analysis formatting package is not needed for this build.

## Build

From the repository root, with a standard TeX Live installation:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error elicitation_arxiv.tex
```

Alternatively, run `pdflatex elicitation_arxiv.tex` twice. The bibliography is
inline, so BibTeX and Biber are not required. The source uses `\today`; the
rendered title-page date therefore follows the build date. Generated PDFs and
LaTeX auxiliary files are not part of this source checkpoint.

## Corrections preserved

- Corrected the sign-dependent scalar brackets and delta-method functions,
  justified their ordering, and separated score notation from expert reports.
- Added fixed-threshold deviation control and carried the exceptional-event
  error through the nonuniform Berry--Esseen comparison.
- Made centering, positive variance, local regularity, optimizer, and
  identification assumptions explicit; distinguished population identification
  from finite-sample recovery and uniform separation.
- Constrained the profiled reporting scale and clarified interior-scale
  requirements; corrected the normal-location flat-risk case at scale 1/2 and
  the beta population-minimizer discussion.
- Reworked the multivariate proof using a quadratic implicit proxy and the
  corresponding assumptions, remainder estimates, and probability bounds.
- Removed the Acknowledgments section.

## Verification and preservation

`verification.txt` is the unmodified verification record saved with the earlier
revision. It records 10,000 scalar-bracket checks and 20 Gaussian quadrature
checks; these are targeted checks, not a formal verification of every proof.
The full simulation experiments were not rerun for this checkpoint.

On September 20, 2026, the exact source files were rebuilt against the existing
repository figures: 34 pages, no LaTeX errors or warnings, no overfull or
underfull boxes, 168 resolved references to 67 unique labels, and 44 resolved
citation keys. The compiled text contains neither an Acknowledgments section
nor unresolved-reference placeholders.

All files already tracked at parent commit
`e7e5463c791b853bdda2870b3ef7ca3debc016f6` are preserved unchanged, including the
reproduction code, data, and nine figure PDFs. Eight figures are byte-identical
to the revision archive. `pictures/ch5_scale.pdf` differs from the archived
copy only in its PDF creation timestamp; its text and rendered image are
identical. The existing repository copy is retained.

## Source provenance

SHA-256 digests of the recovered revision artifacts:

| File | SHA-256 |
| --- | --- |
| `elicitation_arxiv_corrected.tar.gz` | `d082f35533d1cd5fae75e3c2e21f873da2c442cef87ba9c803d569902efb3f88` |
| `elicitation_arxiv.tex` | `075899ec915d3752cde836b3b0662485f9dba68d46d525734d2f3049579d3f30` |
| `elicitation_body.tex` | `e0925127fd01901b8a01f9a6d1e38337d10f3a3a0bfe6f16472add6a4526f6a7` |
