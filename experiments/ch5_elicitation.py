"""Simulation experiments for the least-squares elicitation chapter.

Setup (matching the chapter): simulate the batch data an elicitation would
produce by sampling a "target" distribution p_{theta0} and reporting
sample/likelihood pairs (x_i, z_i) with z_i = p_{theta0}(x_i), optionally
corrupted by multiplicative noise z_i = p_{theta0}(x_i) * (1 + sigma*xi_i)
modelling a variably-good expert. The estimator is

    theta_hat = argmin_theta  sum_i ( p_theta(x_i) - z_i )^2 .

Experiments:
  (1) success rate of elicitation vs samples-per-batch, several noise levels;
  (2) mean squared parameter error and KL(p_theta0 || p_theta_hat) vs n;
  (3) sampling distribution of the standardized error vs N(0,1)
      (illustrates the asymptotic normality / Berry-Esseen theory).

Families: Normal(theta, 1) location family; Beta(theta, 2) shape family.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import polygamma
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)


class NormalLoc:
    name = "Normal$(\\theta,1)$"
    theta0, lo, hi = 2.0, -3.0, 7.0

    @staticmethod
    def sample(theta, n):
        return rng.normal(theta, 1.0, n)

    @staticmethod
    def pdf(x, theta):
        return stats.norm.pdf(x, theta, 1.0)

    @staticmethod
    def dpdf(x, theta):  # d/dtheta pdf
        return stats.norm.pdf(x, theta, 1.0) * (x - theta)

    @classmethod
    def kl(cls, theta_hat):  # KL(p0 || p_hat), closed form
        return 0.5 * (cls.theta0 - theta_hat) ** 2

    @classmethod
    def design_grid(cls):  # x-grid for numerically integrating against p_theta0
        return np.linspace(cls.theta0 - 10, cls.theta0 + 10, 8000)

    @classmethod
    def fisher_info(cls):  # Fisher information of N(theta,1) is 1
        return 1.0

    @staticmethod
    def mle(x):  # sample-only MLE (uses x, ignores z)
        return float(np.mean(x))


class BetaShape:
    name = "Beta$(\\theta,2)$"
    # Search interval matches the parameter set Theta = [1.1, 10] on which
    # condition (A6) is verified in the appendix: (B1) fails for theta <= 1,
    # where p_theta and its theta-derivative are unbounded as x -> 0.
    theta0, lo, hi = 3.0, 1.1, 10.0

    @staticmethod
    def sample(theta, n):
        return rng.beta(theta, 2.0, n)

    @staticmethod
    def pdf(x, theta):
        return stats.beta.pdf(x, theta, 2.0)

    @staticmethod
    def dpdf(x, theta, h=1e-5):
        return (stats.beta.pdf(x, theta + h, 2.0)
                - stats.beta.pdf(x, theta - h, 2.0)) / (2 * h)

    @classmethod
    def kl(cls, theta_hat, m=20000):
        x = np.linspace(1e-4, 1 - 1e-4, m)
        p = stats.beta.pdf(x, cls.theta0, 2.0)
        q = stats.beta.pdf(x, theta_hat, 2.0)
        return float(np.trapezoid(p * (np.log(p) - np.log(q)), x))

    @classmethod
    def design_grid(cls):
        return np.linspace(1e-4, 1 - 1e-4, 8000)

    @classmethod
    def fisher_info(cls):  # Beta(theta,b) shape Fisher info = psi'(theta) - psi'(theta+b)
        return float(polygamma(1, cls.theta0) - polygamma(1, cls.theta0 + 2.0))

    @staticmethod
    def mle(x):  # sample-only MLE for the shape parameter (uses x, ignores z)
        def nll(t):
            return -np.sum(stats.beta.logpdf(x, t, 2.0))
        # Same parameter space Theta = [1.1, 10] as the LS estimator, so the
        # two methods are compared over an identical search set.
        res = minimize_scalar(nll, bounds=(1.1, 10.0), method="bounded")
        return float(res.x)


def elicit_batch(fam, n, sigma):
    x = fam.sample(fam.theta0, n)
    z = fam.pdf(x, fam.theta0) * (1.0 + sigma * rng.normal(size=n))
    return x, z


def fit(fam, x, z):
    """Global grid search + local refinement of the least-squares objective."""
    def Q(t):
        return ((fam.pdf(x, t) - z) ** 2).sum()

    grid = np.linspace(fam.lo, fam.hi, 400)
    t0 = grid[np.argmin([Q(t) for t in grid])]
    w = (fam.hi - fam.lo) / 400
    res = minimize_scalar(Q, bounds=(t0 - 2 * w, t0 + 2 * w), method="bounded")
    return float(res.x)


def experiment_success_and_rate(fam, fname_succ, fname_rate):
    ns = [2, 4, 8, 12, 16, 20, 28, 40]
    sigmas = [0.05, 0.1, 0.2]
    batches = 200
    tol = 0.25

    fig1, ax1 = plt.subplots(figsize=(4.6, 3.2))
    fig2, ax2 = plt.subplots(figsize=(4.6, 3.2))
    for sigma in sigmas:
        succ, mse, kls = [], [], []
        for n in ns:
            errs, ks = [], []
            for _ in range(batches):
                x, z = elicit_batch(fam, n, sigma)
                th = fit(fam, x, z)
                errs.append(th - fam.theta0)
                ks.append(fam.kl(th))
            errs = np.array(errs)
            succ.append((np.abs(errs) < tol).mean())
            mse.append((errs ** 2).mean())
            kls.append(np.mean(ks))
        ax1.plot(ns, succ, marker="o", ms=3.5, label=rf"$\sigma={sigma}$")
        ax2.plot(ns, mse, marker="o", ms=3.5, label=rf"MSE, $\sigma={sigma}$")
        # log-log MSE slopes quoted in the chapter: full range and n >= 8
        slope_full = np.polyfit(np.log(ns), np.log(mse), 1)[0]
        slope_tail = np.polyfit(np.log(ns[2:]), np.log(mse[2:]), 1)[0]
        print(f"{fam.__name__} sigma={sigma}: success={np.round(succ,2).tolist()} "
              f"slope_full={slope_full:.2f} slope_n>=8={slope_tail:.2f}")
    ax1.set_xlabel("samples per batch $n$")
    ax1.set_ylabel(rf"success rate ($|\hat\theta-\theta_0|<{tol}$)")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=8)
    fig1.tight_layout(); fig1.savefig(fname_succ)

    x = np.array(ns, float)
    ax2.plot(x, mse[0] * (x / x[0]) ** -1, "k--", lw=0.8, label=r"$n^{-1}$ reference")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xticks(ns)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.minorticks_off()
    ax2.set_xlabel("samples per batch $n$")
    ax2.set_ylabel(r"mean $(\hat\theta-\theta_0)^2$")
    ax2.legend(frameon=False, fontsize=8)
    fig2.tight_layout(); fig2.savefig(fname_rate)
    print("saved", fname_succ, fname_rate)


def experiment_normality(fam, fname, n=200, sigma=0.1, reps=2000):
    """Standardize with the sandwich variance sqrt(n) I2 / sqrt(I1)."""
    # Monte Carlo estimates of I1 = E psi(theta0)^2 and I2 = E psi'(theta0)
    # for psi((x,z),theta) = (p_theta(x) - z) * dpdf(x,theta).
    m = 400000
    x = fam.sample(fam.theta0, m)
    xi = rng.normal(size=m)
    z = fam.pdf(x, fam.theta0) * (1.0 + sigma * xi)
    psi = (fam.pdf(x, fam.theta0) - z) * fam.dpdf(x, fam.theta0)
    I1 = float((psi ** 2).mean())
    # psi' = dpdf^2 + (pdf - z) * d2pdf ; E[(pdf-z)*d2pdf] = 0 (noise indep.)
    I2 = float((fam.dpdf(x, fam.theta0) ** 2).mean())

    zs = []
    for _ in range(reps):
        xb, zb = elicit_batch(fam, n, sigma)
        th = fit(fam, xb, zb)
        zs.append(np.sqrt(n) * I2 / np.sqrt(I1) * (th - fam.theta0))
    zs = np.array(zs)
    ks = stats.kstest(zs, "norm")
    print(f"{fam.__name__} normality: mean={zs.mean():.3f} sd={zs.std():.3f} "
          f"KS stat={ks.statistic:.3f} p={ks.pvalue:.3f}")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(zs, bins=40, density=True, alpha=0.55, label="standardized errors")
    t = np.linspace(-4, 4, 200)
    ax.plot(t, stats.norm.pdf(t), "k-", lw=1.2, label=r"$\mathcal{N}(0,1)$")
    ax.set_xlabel(r"$\sqrt{n\, I_2(\theta_0)^2 / I_1(\theta_0)}\;(\hat\theta-\theta_0)$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fname)
    print("saved", fname)
    return zs


if __name__ == "__main__":
    experiment_success_and_rate(NormalLoc, "pictures/ch5_success_normal.pdf",
                                "pictures/ch5_rate_normal.pdf")
    experiment_success_and_rate(BetaShape, "pictures/ch5_success_beta.pdf",
                                "pictures/ch5_rate_beta.pdf")
    experiment_normality(NormalLoc, "pictures/ch5_normality.pdf")
