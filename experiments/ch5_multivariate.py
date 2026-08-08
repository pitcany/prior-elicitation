"""Multivariate elicitation: theta in R^k, illustrated on the normal
location-scale family theta = (mu, s), p_theta(x) = phi((x-mu)/s)/s.

The chapter's multivariate section states, for the least-squares elicitation
estimator with theta in R^k:
  (i)  consistency and asymptotic normality with the matrix sandwich
         sqrt(n)(theta_hat - theta0) -> N(0, V),
         V = I2^{-1} I1 I2^{-1},
         I2 = E[grad p grad p^T],  I1 = sigma^2 E[p^2 grad p grad p^T]
       (expectations under X ~ p_theta0, gradients at theta0);
  (ii) a directional Berry-Esseen bound: for each unit vector a,
         sup_z | P( sqrt(n) a^T(theta_hat-theta0) / sqrt(a^T V a) <= z )
                - Phi(z) |  <=  C / sqrt(n),
       obtained from the Newton-step representation and the multivariate
       delta-method bounds of Pinelis and Molzon (2016).

This script verifies, for theta0 = (2, 1.5) and sigma = 0.1:
  [1] per-coordinate MSE decays at the n^{-1} rate;
  [2] the empirical covariance of sqrt(n)(theta_hat - theta0) matches V,
      and standardized errors along directions e1, e2, (e1+e2)/sqrt(2)
      are N(0,1) (mean, sd, KS);
  [3] the Kolmogorov distance of the standardized directional error to
      N(0,1) decays like n^{-1/2} (the Berry-Esseen signature), by
      regressing log KS on log n.

Run from the repository root:
    python experiments/ch5_multivariate.py
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

rng = np.random.default_rng(23)

THETA0 = np.array([2.0, 1.5])           # (mu, s)
BOUNDS = [(-3.0, 7.0), (0.3, 5.0)]      # search box, theta0 well interior
SIGMA = 0.1


def pdf(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    return stats.norm.pdf(x, mu, s)


def grad_pdf(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    """Gradient in (mu, s); rows are d/dmu, d/ds."""
    p = pdf(x, mu, s)
    u = (x - mu) / s
    return np.stack([p * u / s, p * (u**2 - 1.0) / s])


def elicit(n: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(THETA0[0], THETA0[1], n)
    z = pdf(x, *THETA0) * (1.0 + sigma * rng.normal(size=n))
    return x, z


def fit(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Coarse grid then Nelder-Mead inside the search box."""
    def Q(t):
        return float(((pdf(x, t[0], t[1]) - z) ** 2).sum())

    mus = np.linspace(*BOUNDS[0], 30)
    ss = np.linspace(*BOUNDS[1], 30)
    qs = [(Q((m, s)), (m, s)) for m in mus for s in ss]
    t_init = min(qs)[1]
    res = minimize(Q, t_init, method="Nelder-Mead", bounds=BOUNDS,
                   options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000})
    return np.asarray(res.x)


def asymptotic_V(sigma: float) -> np.ndarray:
    """V = I2^{-1} I1 I2^{-1} by quadrature under X ~ p_theta0."""
    x = np.linspace(THETA0[0] - 12 * THETA0[1], THETA0[0] + 12 * THETA0[1], 20000)
    p = pdf(x, *THETA0)
    g = grad_pdf(x, *THETA0)
    I2 = np.array([[np.trapezoid(g[i] * g[j] * p, x) for j in (0, 1)] for i in (0, 1)])
    I1 = sigma**2 * np.array(
        [[np.trapezoid(g[i] * g[j] * p**3, x) for j in (0, 1)] for i in (0, 1)])
    I2inv = np.linalg.inv(I2)
    return I2inv @ I1 @ I2inv


DIRECTIONS = {
    "e1 (location)": np.array([1.0, 0.0]),
    "e2 (scale)": np.array([0.0, 1.0]),
    "(e1+e2)/sqrt2": np.array([1.0, 1.0]) / np.sqrt(2.0),
}


def experiment() -> None:
    V = asymptotic_V(SIGMA)
    print("asymptotic V =", np.round(V, 5).tolist())

    # [1] + [3]: rate and Berry-Esseen decay over a range of n
    ns = [25, 50, 100, 200, 400]
    reps = 4000
    mse = np.zeros((len(ns), 2))
    ks_dir = {name: np.zeros(len(ns)) for name in DIRECTIONS}
    errs_by_n = []
    for i, n in enumerate(ns):
        errs = np.zeros((reps, 2))
        for r in range(reps):
            x, z = elicit(n, SIGMA)
            errs[r] = fit(x, z) - THETA0
        errs_by_n.append(errs)
        mse[i] = (errs**2).mean(axis=0)
        for name, a in DIRECTIONS.items():
            sd = np.sqrt(a @ V @ a / n)
            ks_dir[name][i] = stats.kstest(errs @ a / sd, "norm").statistic
        print(f"n={n}: MSE(mu)={mse[i, 0]:.5f} MSE(s)={mse[i, 1]:.5f}  "
              + "  ".join(f"KS[{k}]={v[i]:.4f}" for k, v in ks_dir.items()))

    for j, name in enumerate(("mu", "s")):
        slope = np.polyfit(np.log(ns), np.log(mse[:, j]), 1)[0]
        print(f"[1] MSE slope ({name}): {slope:.2f}  (want ~ -1)")

    # [2] full sandwich check at n = 200
    i200 = ns.index(200)
    errs = errs_by_n[i200]
    emp = np.cov(errs.T) * 200
    print(f"[2] n*Cov empirical {np.round(emp, 5).tolist()} vs V {np.round(V, 5).tolist()}")
    for name, a in DIRECTIONS.items():
        sd = np.sqrt(a @ V @ a / 200)
        std = errs @ a / sd
        ks = stats.kstest(std, "norm")
        print(f"[2] {name}: mean {std.mean():+.3f} sd {std.std():.3f} "
              f"KS {ks.statistic:.4f} (p={ks.pvalue:.3f})")

    # [3] Berry-Esseen signature: log KS vs log n
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for name in DIRECTIONS:
        slope = np.polyfit(np.log(ns), np.log(ks_dir[name]), 1)[0]
        print(f"[3] KS decay slope [{name}]: {slope:.2f}  (Berry-Esseen: -0.5; "
              f"KS noise floor ~ {1.63 / np.sqrt(reps):.4f} at {reps} reps)")
        axes[1].loglog(ns, ks_dir[name], "o-", label=f"{name} (slope {slope:.2f})")
    x = np.array(ns, float)
    axes[1].loglog(x, ks_dir["e1 (location)"][0] * (x / x[0]) ** -0.5, "k--",
                   lw=0.8, label=r"$n^{-1/2}$ reference")
    # E[KS] for a truly normal sample of `reps` points: the resolution floor
    floor = 0.8687 / np.sqrt(reps)
    axes[1].axhline(floor, color="gray", lw=0.8, ls=":",
                    label=f"KS resolution floor ({reps} reps)")
    axes[1].set_xlabel("samples $n$"); axes[1].set_ylabel("Kolmogorov distance")
    axes[1].legend(fontsize=7, frameon=False)

    for j, lab in enumerate((r"$\hat\mu$", r"$\hat s$")):
        axes[0].loglog(ns, mse[:, j], "o-", label=lab)
    axes[0].loglog(x, mse[0, 0] * (x / x[0]) ** -1, "k--", lw=0.8,
                   label=r"$n^{-1}$ reference")
    axes[0].set_xlabel("samples $n$"); axes[0].set_ylabel("MSE")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig("pictures/ch5_multivariate.pdf")
    print("saved pictures/ch5_multivariate.pdf")


if __name__ == "__main__":
    experiment()
