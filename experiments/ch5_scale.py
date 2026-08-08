"""Joint estimation of (theta, c) when plausibilities arrive on an unknown scale.

Model: the expert reports z = c0 * p_theta0(x) * (1 + sigma*xi), with c0 > 0
an unknown reporting scale. The theta-only least-squares estimator of the
chapter is inconsistent when c0 != 1; the joint estimator

    (theta_hat, c_hat) = argmin_{theta, c} sum_i (c * p_theta(x_i) - z_i)^2

restores consistency. For fixed theta the optimal scale is the linear
least-squares coefficient

    c_hat(theta) = sum_i z_i p_theta(x_i) / sum_i p_theta(x_i)^2,

so the joint fit is a one-dimensional profile search over theta -- the same
grid-plus-refinement used throughout the chapter, at essentially no extra
cost.

Sandwich variance: with beta = (theta, c), mean function m_beta(x) =
c * p_theta(x), gradient grad m = (c*dp_theta, p_theta)^T, and moments under
X ~ p_theta0 (write p = p_theta0, dp = dp_theta0)

    G = E[grad m grad m^T],   W = E[p^2 grad m grad m^T],
    sqrt(n) (beta_hat - beta0)  ->  N(0, V),   V = sigma^2 c0^2 G^{-1} W G^{-1}.

In closed form, with A = E[p^2 dp^2], B = E[dp^2], a = E[p^3 dp],
b = E[p dp], m2 = E[p^2], m4 = E[p^4], Delta = B*m2 - b^2:

    V_thetatheta = sigma^2 * (A*m2^2 - 2*a*b*m2 + b^2*m4) / Delta^2,

which does NOT depend on c0 (rescaling z rescales c_hat only). The price of
not knowing the scale is the inflation factor R = V_thetatheta / (sigma^2
A/B^2), and the LS-vs-MLE crossover becomes sigma*_scale = 1/sqrt(Vt * I_F)
with Vt = V_thetatheta / sigma^2.

This script verifies each of these claims by simulation:
  (1) inconsistency of the theta-only fit at c0 != 1 vs consistency of the
      joint fit (MSE vs n),
  (2) the n^{-1} rate and the c0-independence of the theta-variance,
  (3) the 2x2 sandwich against Monte-Carlo covariance and the normality of
      the standardized errors,
  (4) the closed-form V_thetatheta against the matrix expression, and the
      inflation factor / new crossover for both families.

Run from the repository root:
    python experiments/ch5_scale.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch5_elicitation import BetaShape, NormalLoc  # noqa: E402

rng = np.random.default_rng(11)


# ------------------------------------------------------------ moments and V
def moments(fam) -> dict[str, float]:
    """Population moments under X ~ p_theta0, by quadrature on the design grid."""
    x = fam.design_grid()
    p = fam.pdf(x, fam.theta0)
    dp = fam.dpdf(x, fam.theta0)
    ex = lambda f: float(np.trapezoid(f * p, x))  # E[f(X)], X ~ p_theta0
    return {
        "A": ex(p**2 * dp**2), "B": ex(dp**2), "a": ex(p**3 * dp),
        "b": ex(p * dp), "m2": ex(p**2), "m4": ex(p**4),
        "IF": ex((dp / p) ** 2),
    }


def sandwich(fam, sigma: float, c0: float) -> np.ndarray:
    """Asymptotic covariance V = sigma^2 c0^2 G^{-1} W G^{-1} of (theta_hat, c_hat)."""
    mom = moments(fam)
    x = fam.design_grid()
    p = fam.pdf(x, fam.theta0)
    dp = fam.dpdf(x, fam.theta0)
    grads = np.stack([c0 * dp, p])                       # grad m at truth, 2 x grid
    G = np.array([[np.trapezoid(grads[i] * grads[j] * p, x) for j in (0, 1)] for i in (0, 1)])
    W = np.array([[np.trapezoid(grads[i] * grads[j] * p**3, x) for j in (0, 1)] for i in (0, 1)])
    Ginv = np.linalg.inv(G)
    V = sigma**2 * c0**2 * Ginv @ W @ Ginv

    # closed-form V_thetatheta must agree with the matrix expression
    A, B, a, b, m2, m4 = (mom[k] for k in ("A", "B", "a", "b", "m2", "m4"))
    delta = B * m2 - b**2
    v_theta = sigma**2 * (A * m2**2 - 2 * a * b * m2 + b**2 * m4) / delta**2
    assert abs(V[0, 0] - v_theta) < 1e-8 * v_theta, (V[0, 0], v_theta)
    return V


# ------------------------------------------------------------ estimators
def elicit(fam, n: int, sigma: float, c0: float):
    x = fam.sample(fam.theta0, n)
    z = c0 * fam.pdf(x, fam.theta0) * (1.0 + sigma * rng.normal(size=n))
    return x, z


def fit_theta_only(fam, x, z) -> float:
    """The chapter's original estimator: assumes the density's own scale."""
    def Q(t):
        return ((fam.pdf(x, t) - z) ** 2).sum()
    grid = np.linspace(fam.lo, fam.hi, 400)
    t0 = grid[np.argmin([Q(t) for t in grid])]
    w = (fam.hi - fam.lo) / 400
    return float(minimize_scalar(Q, bounds=(t0 - 2 * w, t0 + 2 * w), method="bounded").x)


def fit_joint(fam, x, z) -> tuple[float, float]:
    """Profile fit: c has a closed form at each theta, so search theta only."""
    def profile(t):
        p = fam.pdf(x, t)
        pp = float((p**2).sum())
        if pp <= 0.0:
            return float((z**2).sum()), 0.0
        c = float((z * p).sum()) / pp
        return float(((c * p - z) ** 2).sum()), c

    def Q(t):
        return profile(t)[0]

    grid = np.linspace(fam.lo, fam.hi, 400)
    t0 = grid[np.argmin([Q(t) for t in grid])]
    w = (fam.hi - fam.lo) / 400
    t_hat = float(minimize_scalar(Q, bounds=(t0 - 2 * w, t0 + 2 * w), method="bounded").x)
    return t_hat, profile(t_hat)[1]


# ------------------------------------------------------------ experiments
def exp_inconsistency(fam, sigma: float, c0: float, ax) -> None:
    """(1)+(2): theta-only is biased at c0 != 1; the joint fit is n^{-1}."""
    ns = [10, 20, 40, 80, 160, 320]
    reps = 400
    mse_only = np.zeros(len(ns))
    mse_joint = np.zeros(len(ns))
    for i, n in enumerate(ns):
        e1, e2 = np.zeros(reps), np.zeros(reps)
        for r in range(reps):
            x, z = elicit(fam, n, sigma, c0)
            e1[r] = fit_theta_only(fam, x, z) - fam.theta0
            e2[r] = fit_joint(fam, x, z)[0] - fam.theta0
        mse_only[i], mse_joint[i] = (e1**2).mean(), (e2**2).mean()
    slope = np.polyfit(np.log(ns), np.log(mse_joint), 1)[0]
    print(f"[1] {fam.name}  c0={c0}: theta-only MSE at n=320: {mse_only[-1]:.4f} "
          f"(plateau = bias^2), joint MSE slope {slope:.2f} (want ~ -1)")
    ax.loglog(ns, mse_only, "o-", label=rf"$\theta$-only ($c_0={c0}$)")
    ax.loglog(ns, mse_joint, "s-", label=rf"joint $(\theta,c)$ ($c_0={c0}$)")
    ax.set_xlabel("samples $n$"); ax.set_ylabel(r"MSE$(\hat\theta)$")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(fam.name, fontsize=9)


def exp_sandwich(fam, sigma: float, n: int = 200, reps: int = 2000) -> None:
    """(3): 2x2 sandwich vs Monte Carlo; c0-independence of the theta-variance."""
    for c0 in (0.5, 1.0, 2.0):
        V = sandwich(fam, sigma, c0)
        est = np.zeros((reps, 2))
        for r in range(reps):
            x, z = elicit(fam, n, sigma, c0)
            est[r] = fit_joint(fam, x, z)
        emp = np.cov((est - [fam.theta0, c0]).T) * n
        std_t = np.sqrt(n / V[0, 0]) * (est[:, 0] - fam.theta0)
        std_c = np.sqrt(n / V[1, 1]) * (est[:, 1] - c0)
        print(f"[3] {fam.name}  c0={c0}:  n*Var(theta) emp {emp[0, 0]:.4g} vs "
              f"asym {V[0, 0]:.4g};  n*Var(c) emp {emp[1, 1]:.4g} vs asym {V[1, 1]:.4g};  "
              f"std theta mean/sd {std_t.mean():+.3f}/{std_t.std():.3f}, "
              f"std c mean/sd {std_c.mean():+.3f}/{std_c.std():.3f}")


def exp_price_of_scale() -> None:
    """(4): inflation factor R and the scale-free crossover sigma*."""
    for fam in (NormalLoc, BetaShape):
        mom = moments(fam)
        A, B = mom["A"], mom["B"]
        v_known = A / B**2                                   # V_theta / sigma^2, c known
        v_free = sandwich(fam, 1.0, 1.0)[0, 0]               # sigma = 1, c0 = 1
        star_known = B / np.sqrt(A * mom["IF"])
        star_free = 1.0 / np.sqrt(v_free * mom["IF"])
        print(f"[4] {fam.name}: inflation R = {v_free / v_known:.3f}, "
              f"sigma* known-scale {star_known:.3f} -> scale-free {star_free:.3f}")


if __name__ == "__main__":
    sigma = 0.1
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    exp_inconsistency(NormalLoc, sigma, c0=1.5, ax=axes[0])
    exp_inconsistency(BetaShape, sigma, c0=1.5, ax=axes[1])
    fig.tight_layout()
    fig.savefig("pictures/ch5_scale.pdf")
    print("saved pictures/ch5_scale.pdf")
    exp_sandwich(NormalLoc, sigma)
    exp_sandwich(BetaShape, sigma)
    exp_price_of_scale()
