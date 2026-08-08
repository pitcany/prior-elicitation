"""Semi-synthetic check: reporting noise in real human judgment data,
and what elicitation can still recover at those noise levels.

Part 1 -- How noisy are real reported magnitudes? The datasets collated
by Pachur (2024) give, for 11 studies/subconditions across five decades
and three countries, the aggregate judged annual frequency of death for
20-41 causes together with the actual frequency. We fit the chapter's
reporting models per dataset:

  scale-only (the chapter's model):  z = c * p * (1 + sigma*xi)
  power (compression allowed):       z = c * p^b * (1 + sigma*xi)

by OLS on the log scale; the implied relative-error level is the
conditional coefficient of variation sigma_cv = sqrt(exp(s^2) - 1)
where s is the residual SD of log z. Compression is universal
(b ~ 0.42-0.72) and even after allowing it, sigma_cv exceeds the
crossover sigma* ~ 0.64 in every dataset: judgments of ABSOLUTE
frequencies are noisier than the level at which reported plausibilities
beat sample-only maximum likelihood. Two caveats cut opposite ways:
these are aggregates over 39-85 participants (individual noise is
higher), and the task -- absolute frequencies of 41 disparate causes
spanning five orders of magnitude -- is a memory task about the world,
not a local relative-plausibility judgment about one's own belief.

Part 2 -- Calibrated semi-synthetic elicitation. Simulate an expert with
the empirically fitted response function z = c*p^b(1+sigma*xi), i.e.
including the compression that the chapter's estimator does NOT model,
and run the joint (theta, c) estimator. For the symmetric location
family, compression is a symmetric widening (p^b proportional to a
density with the same center), so the location estimate stays unbiased
even at the worst-case calibration -- only its variance grows. Shape
parameters are not protected: the beta fit acquires an asymptotic bias.

Run from the repository root:
    python experiments/ch5_semisynthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch5_elicitation import BetaShape, NormalLoc, rng  # noqa: E402
from ch5_scale import fit_joint  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "risk_judgments" / "risk_judgments.csv"
SIGMA_STAR = 0.643  # normal-family crossover from the LS-vs-MLE section


# ---------------------------------------------------------------- part 1
def fit_reporting_models() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    rows = []
    for name, g in df.groupby("dataset"):
        lp, lz = np.log(g.actual.values), np.log(g.estimate.values)
        b, a = np.polyfit(lp, lz, 1)
        s = (lz - (a + b * lp)).std(ddof=2)
        r1 = lz - lp
        s1 = (r1 - r1.mean()).std(ddof=1)
        rows.append({"dataset": name, "n": len(g), "b": b,
                     "sigma_power": float(np.sqrt(np.expm1(s**2))),
                     "sigma_scaleonly": float(np.sqrt(np.expm1(s1**2)))})
    tab = pd.DataFrame(rows).sort_values("sigma_power").reset_index(drop=True)
    print(tab.to_string(index=False,
                        float_format=lambda v: f"{v:.2f}"))
    print(f"\n[1] median b = {tab.b.median():.2f}; "
          f"sigma_power range {tab.sigma_power.min():.2f}-{tab.sigma_power.max():.2f} "
          f"(all above sigma* = {SIGMA_STAR}); "
          f"scale-only range {tab.sigma_scaleonly.min():.1f}-{tab.sigma_scaleonly.max():.1f}")
    return tab


# ---------------------------------------------------------------- part 2
def calibrated_expert(fam, n: int, b: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    x = fam.sample(fam.theta0, n)
    z = fam.pdf(x, fam.theta0) ** b * (1.0 + sigma * rng.normal(size=n))
    return x, z


def calibrated_simulation(reps: int = 1000) -> dict:
    configs = [
        ("worst-case calibration (b=0.50, sigma=1.00)", 0.50, 1.00),
        ("best dataset (b=0.45, sigma=0.79)", 0.45, 0.79),
        ("no compression, at sigma* (b=1, sigma=0.64)", 1.00, 0.64),
    ]
    ns = [8, 20, 40, 80]
    curves = {}
    for fam in (NormalLoc, BetaShape):
        for label, b, sigma in configs:
            succ, bias = [], []
            for n in ns:
                errs = np.zeros(reps)
                for r in range(reps):
                    x, z = calibrated_expert(fam, n, b, sigma)
                    errs[r] = fit_joint(fam, x, z)[0] - fam.theta0
                succ.append(float((np.abs(errs) < 0.25).mean()))
                bias.append(float(errs.mean()))
            curves[(fam.__name__, label)] = (ns, succ, bias)
            print(f"[2] {fam.__name__:10s} {label}: success {succ}, bias "
                  + "[" + ", ".join(f"{v:+.2f}" for v in bias) + "]")
    return curves


# ---------------------------------------------------------------- figure
def make_figure(tab: pd.DataFrame, curves: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

    y = np.arange(len(tab))
    axes[0].scatter(tab.sigma_power, y, s=18, zorder=3)
    axes[0].axvline(SIGMA_STAR, color="k", ls="--", lw=0.9,
                    label=rf"$\sigma^*={SIGMA_STAR}$")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(tab.dataset, fontsize=6.5)
    axes[0].set_xlabel(r"residual relative error $\hat\sigma$ (power fit)")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].invert_yaxis()

    styles = {"worst-case calibration (b=0.50, sigma=1.00)": ("o-", "worst-case ($b{=}0.5$, $\\sigma{=}1.0$)"),
              "best dataset (b=0.45, sigma=0.79)": ("s-", "best dataset ($b{=}0.45$, $\\sigma{=}0.79$)"),
              "no compression, at sigma* (b=1, sigma=0.64)": ("^-", "at $\\sigma^*$ ($b{=}1$, $\\sigma{=}0.64$)")}
    for (fam, label), (ns, succ, _) in curves.items():
        if fam != "NormalLoc":
            continue
        style, leg = styles[label]
        axes[1].plot(ns, succ, style, ms=4, label=leg)
    axes[1].set_xlabel("samples per batch $n$")
    axes[1].set_ylabel(r"success rate ($|\hat\theta-\theta_0|<0.25$)")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig("pictures/ch5_semisynthetic.pdf")
    print("saved pictures/ch5_semisynthetic.pdf")


if __name__ == "__main__":
    table = fit_reporting_models()
    sim_curves = calibrated_simulation()
    make_figure(table, sim_curves)
