"""Least-squares elicitation for a family with no moments (Cauchy location).

The least-squares objective is defined by the density alone, so it stays well
posed for families where moment matching is undefined. The Cauchy location
family p_theta(x) = 1/(pi(1 + (x-theta)^2)) is the canonical case: it has no
mean and no finite absolute moment of order >= 1 (absolute moments of order
0 <= p < 1 are finite, equal to sec(pi*p/2) in the standard case), so the
sample mean does not converge and a method-of-moments fit has nothing to
match.

Reports, for theta0 = 2:
  (1) the sample mean over several batches, to show it does not settle;
  (2) success rate and MSE of the least-squares elicitation estimator;
  (3) the log-log MSE slope in n, against the n^{-1} rate predicted by the
      asymptotic normality result.

Run from the repository root:  python experiments/ch5_cauchy.py
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(3)

THETA0 = 2.0
LO, HI, GRID = -8.0, 12.0, 400


def pdf(x: np.ndarray, theta: float) -> np.ndarray:
    """Cauchy(theta, 1) density."""
    return stats.cauchy.pdf(x, theta, 1.0)


def fit(x: np.ndarray, z: np.ndarray) -> float:
    """Minimize sum_i (p_theta(x_i) - z_i)^2: grid search then local refinement."""
    def Q(t: float) -> float:
        return float(((pdf(x, t) - z) ** 2).sum())

    grid = np.linspace(LO, HI, GRID)
    t0 = grid[np.argmin([Q(t) for t in grid])]
    w = (HI - LO) / GRID
    res = minimize_scalar(Q, bounds=(t0 - 2 * w, t0 + 2 * w), method="bounded")
    return float(res.x)


def elicit(n: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """One elicited batch: samples from the belief, plausibilities with noise."""
    x = THETA0 + rng.standard_cauchy(n)
    z = pdf(x, THETA0) * (1.0 + sigma * rng.normal(size=n))
    return x, z


def main() -> None:
    print(f"Cauchy(theta,1) location family, theta0 = {THETA0} "
          f"(no mean, no variance)\n")

    print(" method of moments is undefined here; the sample mean does not settle.")
    print(" sample mean over 5 independent batches of n=40:")
    for _ in range(5):
        print(f"    mean(x) = {THETA0 + rng.standard_cauchy(40).mean():+10.3f}")

    print("\n least-squares elicitation estimator (300 batches per configuration):")
    print(f"{'n':>5} {'sigma':>7} {'MSE':>12} {'success |err|<0.25':>20}")
    for sigma in (0.05, 0.1, 0.2):
        for n in (10, 20, 40):
            errs = np.array([fit(*elicit(n, sigma)) - THETA0 for _ in range(300)])
            print(f"{n:5d} {sigma:7.2f} {np.mean(errs ** 2):12.3e} "
                  f"{np.mean(np.abs(errs) < 0.25):20.2f}")

    ns = [10, 20, 40, 80]
    mses = []
    for n in ns:
        errs = np.array([fit(*elicit(n, 0.1)) - THETA0 for _ in range(300)])
        mses.append(float(np.mean(errs ** 2)))
    slope = float(np.polyfit(np.log(ns), np.log(mses), 1)[0])
    print(f"\n rate check (sigma=0.1): MSE {['%.3e' % m for m in mses]}")
    print(f"   log-log slope = {slope:+.3f}   (asymptotic theory predicts -1)")


if __name__ == "__main__":
    main()
