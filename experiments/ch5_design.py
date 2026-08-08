"""Does the placement of the elicited points matter? (Design comparison.)

The least-squares elicitation estimator is a nonlinear least-squares fit in which
the elicited points x_i act as DESIGN points. Under a design measure q the
asymptotic variance is

    Var_q(theta_hat) ~ sigma^2 * E_q[p0^2 pdot0^2] / (n * (E_q[pdot0^2])^2),

so drawing the x_i i.i.d. from the expert's belief is only one choice of q.
This script compares that choice against deliberately spread-out designs, for
both a location family and a shape family. The designs must respect the
family's support, so they differ between the two: for N(theta,1) they are
intervals around theta0, and for Beta(theta,2) they are subsets of (0,1).

It also reports a degeneracy of the multiplicative noise model. For a one-point
design at x0 the variance above reduces to 1 / (d/dtheta log p_theta0(x0))^2,
i.e. the reciprocal squared score, which is unbounded in both families. Because
the absolute error sigma*p(x) vanishes where the density does, the model treats
far-tail points as nearly noiseless and the "optimal" design runs off to where
the expert has no usable opinion. The symmetric two-point design at theta0 +- d
in the normal family makes this concrete: its asymptotic variance is exactly
sigma^2/(n d^2), decreasing in d without bound.

Run from the repository root:  python experiments/ch5_design.py
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(7)

SIGMA, N, REPS = 0.1, 30, 2000


class NormalLoc:
    name = "Normal(theta,1), theta0=2"
    theta0, lo, hi = 2.0, -3.0, 7.0

    @staticmethod
    def pdf(x, t):
        return stats.norm.pdf(x, t, 1.0)

    @staticmethod
    def dpdf(x, t):
        return stats.norm.pdf(x, t, 1.0) * (x - t)

    @classmethod
    def designs(cls):
        t0 = cls.theta0
        return {
            "iid from belief":        lambda: rng.normal(t0, 1.0, N),
            "uniform [t0-2, t0+2]":   lambda: rng.uniform(t0 - 2, t0 + 2, N),
            "uniform [t0-3, t0+3]":   lambda: rng.uniform(t0 - 3, t0 + 3, N),
            "equispaced [t0-2,t0+2]": lambda: np.linspace(t0 - 2, t0 + 2, N),
            "two-point at t0 +- 1":   lambda: np.repeat([t0 - 1, t0 + 1], N // 2),
            "two-point at t0 +- 2":   lambda: np.repeat([t0 - 2, t0 + 2], N // 2),
        }


class BetaShape:
    name = "Beta(theta,2), theta0=3"
    theta0, lo, hi = 3.0, 1.1, 10.0  # Theta = [1.1, 10], where (A6) is proved

    @staticmethod
    def pdf(x, t):
        return stats.beta.pdf(x, t, 2.0)

    @staticmethod
    def dpdf(x, t, h=1e-5):
        return (stats.beta.pdf(x, t + h, 2.0)
                - stats.beta.pdf(x, t - h, 2.0)) / (2 * h)

    @classmethod
    def designs(cls):
        # x lives on (0,1); theta is a shape parameter, so the location-family
        # designs above do not transfer. The score is 1/t0 + 1/(t0+1) + log x,
        # whose magnitude grows as x -> 0.
        return {
            "iid from belief":     lambda: rng.beta(cls.theta0, 2.0, N),
            "uniform (0,1)":       lambda: rng.uniform(1e-3, 1 - 1e-3, N),
            "equispaced (0,1)":    lambda: np.linspace(0.02, 0.98, N),
            "two-point 0.2, 0.7":  lambda: np.repeat([0.2, 0.7], N // 2),
            "two-point 0.05, 0.5": lambda: np.repeat([0.05, 0.5], N // 2),
        }


def asymptotic_variance(fam, xs):
    p, dp = fam.pdf(xs, fam.theta0), fam.dpdf(xs, fam.theta0)
    return SIGMA ** 2 * float(np.mean(p ** 2 * dp ** 2)) / (
        N * float(np.mean(dp ** 2)) ** 2)


def fit(fam, x, z, grid=400):
    def Q(t):
        return float(((fam.pdf(x, t) - z) ** 2).sum())

    g = np.linspace(fam.lo, fam.hi, grid)
    t = g[np.argmin([Q(v) for v in g])]
    w = (fam.hi - fam.lo) / grid
    return float(minimize_scalar(Q, bounds=(t - 2 * w, t + 2 * w),
                                 method="bounded").x)


def compare(fam):
    print(f"\n=== {fam.name}, sigma={SIGMA}, n={N}, {REPS} reps ===")
    print(f"{'design':24s} {'asym Var':>10s} {'empirical MSE':>14s} "
          f"{'vs iid':>8s} {'pred/obs':>9s}")
    baseline = None
    for name, draw in fam.designs().items():
        av = asymptotic_variance(fam, np.concatenate([draw() for _ in range(120)]))
        errs = []
        for _ in range(REPS):
            x = draw()
            z = fam.pdf(x, fam.theta0) * (1.0 + SIGMA * rng.normal(size=len(x)))
            errs.append((fit(fam, x, z) - fam.theta0) ** 2)
        mse = float(np.mean(errs))
        baseline = mse if baseline is None else baseline
        print(f"{name:24s} {av:10.3e} {mse:14.3e} {mse / baseline:7.2f}x "
              f"{av / mse:8.2f}x")


def degeneracy_normal():
    """Two-point design at theta0 +- d in the normal family: Var = sigma^2/(n d^2)."""
    print(f"\n=== degeneracy, normal family: two-point at theta0 +- d "
          f"({REPS} reps) ===")
    print(f"{'d':>5} {'asym Var':>11} {'sigma^2/(n d^2)':>16} "
          f"{'empirical MSE':>14} {'p(theta0+d)':>12}")
    t0 = NormalLoc.theta0
    for d in (0.5, 1.0, 2.0, 3.0, 4.0):
        xs = np.repeat([t0 - d, t0 + d], N // 2)
        av = asymptotic_variance(NormalLoc, xs)
        errs = []
        for _ in range(REPS):
            z = NormalLoc.pdf(xs, t0) * (1.0 + SIGMA * rng.normal(size=len(xs)))
            errs.append((fit(NormalLoc, xs, z) - t0) ** 2)
        print(f"{d:5.1f} {av:11.3e} {SIGMA ** 2 / (N * d ** 2):16.3e} "
              f"{np.mean(errs):14.3e} {NormalLoc.pdf(np.array([t0 + d]), t0)[0]:12.2e}")


def degeneracy_beta(reps=800):
    """One-point design at x0: Var = sigma^2 / (n score(x0)^2). For Beta(theta,2)
    the score 1/t0 + 1/(t0+1) + log x is unbounded as x -> 0."""
    t0 = BetaShape.theta0
    print(f"\n=== degeneracy, beta family: one-point design at x0 "
          f"({reps} reps) ===")
    print(f"{'x0':>7} {'score':>8} {'pred Var':>11} {'empirical MSE':>14} "
          f"{'p(x0)':>10}")
    for x0 in (0.5, 0.2, 0.05, 0.01, 0.002):
        xs = np.full(N, x0)
        score = 1 / t0 + 1 / (t0 + 1) + np.log(x0)
        errs = []
        for _ in range(reps):
            z = BetaShape.pdf(xs, t0) * (1.0 + SIGMA * rng.normal(size=N))
            errs.append((fit(BetaShape, xs, z) - t0) ** 2)
        print(f"{x0:7.3f} {score:8.2f} {SIGMA ** 2 / (N * score ** 2):11.3e} "
              f"{np.mean(errs):14.3e} {BetaShape.pdf(np.array([x0]), t0)[0]:10.3e}")
    print(" (at x0=0.5 the score is near zero, so theta is barely identified and")
    print("  the asymptotic approximation is poor; elsewhere it tracks closely.)")


def summary() -> None:
    print("\n The variance falls without bound as the design moves to where the")
    print(" density is small, in BOTH families, because the absolute error")
    print(" sigma*p(x) vanishes there. The 'optimal' design therefore runs into")
    print(" the tails, where no expert has a usable opinion. This is a")
    print(" limitation of the multiplicative noise model, not a recommendation:")
    print(" an additive error floor would remove the degeneracy.")


if __name__ == "__main__":
    compare(NormalLoc)
    compare(BetaShape)
    degeneracy_normal()
    degeneracy_beta()
    summary()
