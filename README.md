# A Least-Squares Approach to Sample-Based Prior Elicitation

Code and data to reproduce every figure and every reported number in the paper

> Yannik Pitcan, *A Least-Squares Approach to Sample-Based Prior Elicitation*.

An expert supplies example points together with their approximate
plausibilities; the prior is fitted by minimizing the squared discrepancy
between a parametric density and the elicited values. The paper develops the
resulting M-estimator's consistency, asymptotic normality, and a non-asymptotic
Berry–Esseen bound, then relaxes the assumptions that most limit it in
practice — an unknown reporting scale, a scalar parameter, and a noise model
without an error floor.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run from the repository root — scripts write to pictures/
python experiments/ch5_elicitation.py
```

`numpy >= 2.0` is a hard requirement (`np.trapezoid`). `requirements.txt` pins
the exact versions that produced the numbers in the paper.

## What produces what

| Script | Produces |
| --- | --- |
| `ch5_elicitation.py` | Success-rate, error-rate and normality figures; the estimator and both families, imported by most other scripts |
| `ch5_ls_vs_mle.py` | Least-squares vs. maximum-likelihood comparison and the crossover σ\* |
| `ch5_design.py` | Design comparison, and the degeneracy of the optimal design under purely multiplicative noise |
| `ch5_cauchy.py` | The no-moments (Cauchy location family) results |
| `ch5_scale.py` | Joint (θ, c) estimation under an unknown reporting scale: inconsistency of the fixed-scale fit, the 2×2 sandwich, the inflation factor |
| `ch5_multivariate.py` | Multivariate parameters (normal location–scale): matrix sandwich and directional normality |
| `ch5_floor.py` | Additive error floor: floored sandwich and the resulting interior optimal designs |
| `ch5_semisynthetic.py` | Reporting-noise fits to eleven human frequency-judgment datasets, and the calibrated-expert simulation |

Every script is seeded and reproduces its reported figures and numbers exactly.
Committed PDFs in `pictures/` are the versions used in the paper; rerunning a
script overwrites its own outputs.

## Data

`experiments/data/risk_judgments/` holds aggregate judged and actual annual
frequencies of death for up to 41 causes, across eleven studies collated by

> T. Pachur (2024). The perception of dramatic risks: Biased media, but
> unbiased minds. *Cognition* 246:105736. Data: https://osf.io/u4d7g

`risk_judgments.csv` is a tidy conversion of that repository's per-study
spreadsheets (columns: `dataset`, `source_file`, `risk`, `actual`, `estimate`;
rows with missing or nonpositive values dropped). `fetch_convert.py`
regenerates it from the original source, documenting the provenance.

## Citation

Please cite the paper. A `CITATION.cff` will be added once the arXiv
identifier is issued.

## License

MIT (see `LICENSE`). The risk-judgment data is redistributed from its original
open repository; please cite Pachur (2024) if you use it.
