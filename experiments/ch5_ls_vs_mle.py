"""When do reported plausibilities help? Least-squares elicitation vs. MLE.

The least-squares estimator uses the pairs (x_i, z_i): the sampled points and
the expert's reported plausibilities. The natural sample-only baseline is the
maximum-likelihood estimator on the x_i alone, which -- when the examples are
drawn i.i.d. from the expert's belief p_{theta0} -- is Cramer-Rao efficient
among estimators using only the samples. Comparing the two therefore isolates
the *value of the reported plausibilities* as a function of how noisy they are.

Asymptotic variances (per sample), at theta0, with z = p0(x)(1+sigma*xi):
    psi((x,z),theta) = (p_theta(x) - z) * dp_theta(x)
    I1 = E[psi^2] = sigma^2 * A,   A = E_{x~p0}[ p0(x)^2 dp0(x)^2 ]
    I2 = E[d/dtheta psi] = B,      B = E_{x~p0}[ dp0(x)^2 ]
    => Var_LS(theta_hat) ~ (sigma^2 A / B^2) / n
    Var_MLE(theta_hat)  ~ 1 / (I_F n),   I_F = Fisher information.

The LS variance grows like sigma^2 (exact recovery as sigma -> 0), while the
MLE variance is constant in sigma. They cross at
    sigma* = B / sqrt(A * I_F),
below which the plausibility labels strictly help, above which a reliable
expert's samples alone (via MLE) are worth more than noisy plausibilities.

Note: this i.i.d.-from-belief design is the *conservative* comparison for the
LS method, since it is exactly where the sample-only baseline is strongest. In
a diverse-sampling design the x_i no longer encode the density and MLE-on-x is
not valid, so the LS advantage would only widen.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ch5_elicitation import NormalLoc, BetaShape, fit, rng


def asymptotic_constants(fam):
    x = fam.design_grid()
    p = fam.pdf(x, fam.theta0)
    dp = fam.dpdf(x, fam.theta0)
    B = float(np.trapezoid(p * dp ** 2, x))        # E[dp^2]
    A = float(np.trapezoid(p ** 3 * dp ** 2, x))   # E[p^2 dp^2]
    IF = fam.fisher_info()
    return A, B, IF


def run(fam, ax, n=30, reps=800):
    A, B, IF = asymptotic_constants(fam)
    sigma_star = B / np.sqrt(A * IF)
    c_ls = lambda s: s ** 2 * A / B ** 2   # per-sample asymptotic variance of LS
    c_mle = 1.0 / IF                       # per-sample asymptotic variance of MLE
    print(f"\n{fam.__name__}: A={A:.4g} B={B:.4g} I_F={IF:.4g}  "
          f"sigma*={sigma_star:.3f}")

    sigmas = np.logspace(np.log10(0.05), np.log10(2.0), 12)
    mse_ls, mse_mle = [], []
    for s in sigmas:
        e_ls, e_mle = [], []
        for _ in range(reps):
            x = fam.sample(fam.theta0, n)
            z = fam.pdf(x, fam.theta0) * (1.0 + s * rng.normal(size=n))
            e_ls.append((fit(fam, x, z) - fam.theta0) ** 2)
            e_mle.append((fam.mle(x) - fam.theta0) ** 2)
        mse_ls.append(np.mean(e_ls))
        mse_mle.append(np.mean(e_mle))
        print(f"  sigma={s:5.3f}  MSE_LS={mse_ls[-1]:.2e}  MSE_MLE={mse_mle[-1]:.2e}"
              f"  {'LS wins' if mse_ls[-1] < mse_mle[-1] else 'MLE wins'}")

    ss = np.logspace(np.log10(0.05), np.log10(2.0), 200)
    ax.plot(ss, c_ls(ss) / n, color="C0", lw=1.0,
            label="LS asymptotic")
    ax.axhline(c_mle / n, color="C1", lw=1.0, label="MLE asymptotic")
    ax.plot(sigmas, mse_ls, "o", color="C0", ms=4,
            label="LS (uses $x_i,z_i$)")
    ax.plot(sigmas, mse_mle, "s", color="C1", ms=4,
            label="MLE (uses $x_i$ only)")
    ax.axvline(sigma_star, color="k", ls=":", lw=0.9)
    ax.axvspan(sigmas.min(), sigma_star, color="C0", alpha=0.06)
    ax.annotate(rf"$\sigma^\ast\!\approx\!{sigma_star:.2f}$",
                xy=(sigma_star, ax.get_ylim()[0]), fontsize=8,
                xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"expert noise $\sigma$")
    ax.set_ylabel(r"mean $(\hat\theta-\theta_0)^2$" + rf"  ($n={n}$)")
    ax.set_title(fam.name, fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    return sigma_star


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    run(NormalLoc, axes[0])
    run(BetaShape, axes[1])
    fig.tight_layout()
    fig.savefig("pictures/ch5_ls_vs_mle.pdf")
    print("\nsaved pictures/ch5_ls_vs_mle.pdf")
