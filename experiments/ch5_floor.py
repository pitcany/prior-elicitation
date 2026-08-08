"""The additive error floor: z = p_theta0(x)(1+sigma*xi) + tau*eta.

Under the purely multiplicative reporting model the absolute error
sigma*p_theta0(x) vanishes wherever the density does, so the 'optimal'
design runs into the tails (Remark 'The optimal design is degenerate' in
the chapter). An additive floor tau > 0 models the fact that a report
about an implausible value still carries error. Consequences verified
here (conditional mean is unchanged, so the estimator, consistency and
normality all carry over; only the conditional variance changes, to
Var(z|x) = sigma^2 p0(x)^2 + tau^2):

  [1] i.i.d.-from-belief design: the asymptotic variance becomes
        (sigma^2 A + tau^2 B) / (n B^2),   A = E[p0^2 dp0^2], B = E[dp0^2]
      -- the floor enters as exactly + tau^2/(nB). Verified by simulation.
  [2] under a design measure q the variance is
        V(q) = E_q[(sigma^2 p0^2 + tau^2) dp0^2] / (n E_q[dp0^2]^2),
      which now blows up as the design escapes to the tails (dp0 -> 0
      while the numerator floor stays): interior optima exist.
      Normal family, symmetric two-point design at theta0 +- d:
        V(d) = (sigma^2 phi(d)^2 + tau^2) / (n phi(d)^2 d^2),
      minimized at a finite d*(sigma, tau); the tau = 0 pathology
      (d* = infinity) is gone. Verified: empirical MSE at d* matches
      V(d*), and the formerly 'best' d = 4 design is now far worse.
  [3] beta family, one-point designs: interior optimum x0*(sigma, tau)
      instead of x0 -> 0.

Run from the repository root:
    python experiments/ch5_floor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch5_elicitation import BetaShape, NormalLoc, fit, rng  # noqa: E402

SIGMA = 0.1


def moments(fam) -> tuple[float, float]:
    """A = E[p0^2 dp0^2], B = E[dp0^2] under X ~ p_theta0, by quadrature."""
    x = fam.design_grid()
    p = fam.pdf(x, fam.theta0)
    dp = fam.dpdf(x, fam.theta0)
    A = float(np.trapezoid(p**2 * dp**2 * p, x))
    B = float(np.trapezoid(dp**2 * p, x))
    return A, B


def elicit_floor(fam, x: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    p = fam.pdf(x, fam.theta0)
    return p * (1.0 + sigma * rng.normal(size=len(x))) + tau * rng.normal(size=len(x))


def design_variance(fam, x: np.ndarray, sigma: float, tau: float) -> float:
    """V(q) * n for the empirical design q = point masses at x."""
    p = fam.pdf(x, fam.theta0)
    dp = fam.dpdf(x, fam.theta0)
    num = np.mean((sigma**2 * p**2 + tau**2) * dp**2)
    den = np.mean(dp**2) ** 2
    return float(num / den)


def mc_mse(fam, x: np.ndarray, sigma: float, tau: float, reps: int = 2000) -> float:
    errs = np.zeros(reps)
    for r in range(reps):
        z = elicit_floor(fam, x, sigma, tau)
        errs[r] = fit(fam, x, z) - fam.theta0
    return float((errs**2).mean())


# ------------------------------------------------------------ [1] iid sandwich
def exp_iid_sandwich(n: int = 200, reps: int = 2000) -> None:
    for fam in (NormalLoc, BetaShape):
        A, B = moments(fam)
        for tau in (0.02, 0.05):
            v_pred = (SIGMA**2 * A + tau**2 * B) / B**2
            errs = np.zeros(reps)
            for r in range(reps):
                x = fam.sample(fam.theta0, n)
                z = elicit_floor(fam, x, SIGMA, tau)
                errs[r] = fit(fam, x, z) - fam.theta0
            print(f"[1] {fam.__name__} tau={tau}: n*Var emp {n * errs.var():.4f} "
                  f"vs (s^2 A + t^2 B)/B^2 = {v_pred:.4f} "
                  f"(floor share {tau**2 / B / v_pred:.0%})")


# ------------------------------------------------------------ [2] normal 2-pt
def exp_normal_twopoint(n: int = 30) -> None:
    t0 = NormalLoc.theta0

    def V(d: float, tau: float) -> float:
        phi2 = NormalLoc.pdf(t0 + d, t0) ** 2
        return (SIGMA**2 * phi2 + tau**2) / (phi2 * d**2)

    print(f"\n[2] normal two-point design at theta0 +- d, sigma={SIGMA}, n={n}:")
    for tau in (0.01, 0.02, 0.05):
        d_star = minimize_scalar(lambda d: V(d, tau), bounds=(0.1, 6.0),
                                 method="bounded").x
        x_star = np.repeat([t0 - d_star, t0 + d_star], n // 2)
        x_four = np.repeat([t0 - 4.0, t0 + 4.0], n // 2)
        mse_star = mc_mse(NormalLoc, x_star, SIGMA, tau)
        mse_four = mc_mse(NormalLoc, x_four, SIGMA, tau)
        print(f"    tau={tau}: d* = {d_star:.2f}   V(d*)/n = {V(d_star, tau)/n:.2e} "
              f"emp {mse_star:.2e}   |   d=4: V/n = {V(4.0, tau)/n:.2e} "
              f"emp {mse_four:.2e}  ({mse_four/mse_star:.0f}x worse than d*)")


# ------------------------------------------------------------ [3] beta 1-pt
def exp_beta_onepoint(n: int = 30) -> None:
    def V(x0: float, tau: float) -> float:
        p = float(BetaShape.pdf(np.array([x0]), BetaShape.theta0)[0])
        dp = float(BetaShape.dpdf(np.array([x0]), BetaShape.theta0)[0])
        return (SIGMA**2 * p**2 + tau**2) / dp**2

    print(f"\n[3] beta one-point design at x0, sigma={SIGMA}, n={n}:")
    for tau in (0.01, 0.02, 0.05):
        x_star = minimize_scalar(lambda x0: V(x0, tau), bounds=(1e-4, 0.9999),
                                 method="bounded").x
        mse_star = mc_mse(BetaShape, np.full(n, x_star), SIGMA, tau, reps=800)
        mse_tail = mc_mse(BetaShape, np.full(n, 0.002), SIGMA, tau, reps=800)
        print(f"    tau={tau}: x0* = {x_star:.3f}   V(x0*)/n = {V(x_star, tau)/n:.2e} "
              f"emp {mse_star:.2e}   |   x0=0.002: V/n = {V(0.002, tau)/n:.2e} "
              f"emp {mse_tail:.2e}  ({mse_tail/mse_star:.0f}x worse than x0*)")


if __name__ == "__main__":
    exp_iid_sandwich()
    exp_normal_twopoint()
    exp_beta_onepoint()
