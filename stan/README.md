# Binned background fit

This directory holds two models. `background.stan` is documented here.
`signal.stan`, the fixed-background signal fit at one mass, is documented in
[`docs/fitting.md`](../docs/fitting.md#signal-fit).

`background.stan` fits independent, raw histogram counts with

$
n_b\sim\operatorname{Poisson}(\lambda_b),\qquad
\lambda_b=N_{2\nu}\int_{L_b}^{U_b}f_{\rm PR}(E)\,dE
+\int_{L_b}^{U_b}\left[c_0(1-E/Q)+c_Q E/Q\right]dE
+N_{\rm Ar39}\left[F_{\rm Ar39}(U_b)-F_{\rm Ar39}(L_b)\right].
$

The product of Poisson terms runs only over bins with `use_bin = 1`. Bins with
`use_bin = 0`, normally the gamma-peak windows flagged by `fit_mask` from
`prepare_background_spectra`, are left out of the likelihood. Their observed
counts include the peaks, which the model does not describe. The model still
computes $\lambda_b$ for every bin, so the predicted background under each
excluded peak is available in `expected_counts` and `counts_rep`. Do not set
counts in excluded windows to zero instead: a Poisson term with $n_b=0$ adds
$-\lambda_b$ to the log-likelihood and pulls the background down.

The three functions follow the supplied Python definitions. The PR integral is
evaluated exactly as a weighted sum of beta CDFs. The linear integral is
`(hi - lo) * (c0 * (1 - midpoint / Q) + cQ * midpoint / Q)`. The modified beta
CDF applies
[Stan's beta CDF](https://mc-stan.org/docs/functions-reference/continuous_distributions_on_0_1.html)
to the transformed base beta CDF. No bin-center approximation is used.

Both components use matching names: `pr_cdf` and `ar39_cdf` hold CDF values at
the bin edges; `pr_bin_probability` and `ar39_bin_probability` hold the
differences between adjacent CDF values. The PR quantities are computed in
`transformed data` because their shape is fixed; the Ar39 quantities are
computed in `transformed parameters` because their shape is fitted.

`N_2nu` counts events over `[0, Q]`; `N_ar39` counts events over
`[loc, loc + scale]`. A restricted fit window retains only the corresponding
fractions of these yields. `c0` and `cQ` are endpoint densities in counts/keV,
so the full linear yield over `[0, Q]` is `Q * (c0 + cQ) / 2`. All energies and
support parameters are in keV. `sigma` is a dimensionless beta shape parameter.

The model fits `N_2nu, N_ar39, c0, cQ, mu, sigma, tau, base_alpha, base_beta`
and fixes `Q, loc, scale` through data inputs. `c0` and `cQ` can also be fixed
for a run (see [Fix parameters](#fix-parameters)); fixing both to zero gives the
model without the linear continuum. All five Ar39 shape parameters are fitted,
as in the Python fit. Allowing both the base beta and outer beta shapes to vary
can introduce degeneracies: when `tau = 1` and `base_beta = sigma = 1`, only the
product `base_alpha * mu` determines the CDF.

This is a joint fit of all three components over the supplied histogram. The
Python workflow first fits PR plus the linear continuum on 600--1100 keV, then
fixes `N_2nu` while fitting Ar39 plus the continuum on 60--1100 keV. The joint
Stan model instead samples `N_2nu` together with the other parameters.

Provide positive lognormal priors using means and standard deviations **in log
parameter space**, with the ordering specified in the model. If a prior has
median `m`, set its log location to `np.log(m)`. The example values below only
illustrate the interface; choose priors using exposure and calibration
information, and check sensitivity to them.

From the `bosonic-DM` directory, run Python through `pixi run python3`:

```python
import numpy as np
from cmdstanpy import CmdStanModel

# spectrum = spectrum_in_range(prepare_background_spectra(...)[group], (60, 1100))
# counts, edges = spectrum.counts_analysis, spectrum.bin_edges
# Edges must increase strictly and lie within [0, Q].
data = {
    "n_bins": len(counts),
    "edges": edges,
    "counts": counts,
    "use_bin": spectrum.fit_mask.astype(int),  # 0 = excluded peak window
    "Q": 2039.0,
    "loc": 0.0,
    "scale": 565.8,
    # 1 = sample c0 / cQ; 0 = fix it to c0_fixed / cQ_fixed (counts/keV).
    "fit_c0": 1,
    "fit_cQ": 1,
    "c0_fixed": 0.0,
    "cQ_fixed": 0.0,
    # Order: N_2nu, N_ar39, c0, cQ (entries of fixed parameters are ignored).
    "log_rate_prior_mean": np.log([1000.0, 1000.0, 1.0, 1.0]),
    "log_rate_prior_sd": [1.0, 1.0, 1.0, 1.0],
    # Order: mu, sigma, tau, base_alpha, base_beta.
    "log_shape_prior_mean": np.log([1.0, 1.0, 1.0, 2.0, 3.0]),
    "log_shape_prior_sd": [0.5, 0.5, 0.5, 0.5, 0.5],
}
model = CmdStanModel(stan_file="stan/background.stan")
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.9,
    seed=12345,
    output_dir="stan/output/background",
)
# Component bin expectations: expected_2nu, expected_linear, expected_ar39.
# Total bin expectations: expected_counts.
# Posterior predictive counts (all bins): counts_rep.
# Per-bin log likelihood (fitted bins only, in order): log_lik.
```

`fitting/background_fit.py` wraps these steps: `priors_from_prefit`,
`build_stan_data`, `run_background_fit`, `summarize_background_fit`, and
`save_background_fit`.

The model block uses `target += ..._lpdf(...)` and `target += ..._lpmf(...)`
rather than `~` statements, so every normalizing constant is kept and `lp__` is
the full log posterior density on the unconstrained scale. Sampling does not
depend on these constants, but marginal likelihoods do (see
[Compare models](#compare-models)).

Use raw counts, not background-subtracted, weighted, or efficiency-corrected
histograms. This initial model assumes perfect energy reconstruction and unit
acceptance; energy response, efficiencies, and additional spectral components
need to be added when required by the dataset. The Ar39 support is sharp at
`loc + scale` in this version.

## Check chains and sampler diagnostics

Run the following examples after sampling, using the matching `fit` and `data`
objects for one detector group. These examples use the ArviZ 1.x API installed
in the Pixi environment: `ci_prob` controls the credible interval probability,
and `plot_dist` plots posterior densities. Older ArviZ 0.x examples use
different arguments and function names.

Convert the fit while retaining the separate chain and draw dimensions. The
`counts_rep` and `log_lik` outputs are assigned to their corresponding groups,
and the counts of the fitted bins are stored as observed data, matching
`log_lik`. Excluded bins are not part of the observed data. See the
[ArviZ CmdStanPy conversion guide](https://python.arviz.org/projects/base/en/v1.0.0/how_to/ConversionGuideCmdStanPy.html).

```python
import arviz as az
import matplotlib.pyplot as plt

parameter_names = [
    "N_2nu", "N_ar39", "c0", "cQ",
    "mu", "sigma", "tau", "base_alpha", "base_beta",
]
bin_variables = [
    "counts_rep", "expected_counts",
    "expected_2nu", "expected_ar39", "expected_linear",
]
use_bin = np.asarray(data["use_bin"], dtype=bool)
idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive=["counts_rep"],
    observed_data={"counts": np.asarray(data["counts"])[use_bin]},
    log_likelihood={"counts": "log_lik"},
    coords={
        "energy_bin": np.arange(data["n_bins"]),
        "fit_bin": np.flatnonzero(use_bin),
    },
    dims={
        **{name: ["energy_bin"] for name in bin_variables},
        "counts": ["fit_bin"],
        "log_lik": ["fit_bin"],
    },
)

print(fit.diagnose())
print("Divergences per chain:", fit.divergences)
print("Maximum-treedepth hits per chain:", fit.max_treedepths)
print("E-BFMI per chain:", az.bfmi(idata))

summary = az.summary(
    idata, var_names=parameter_names, ci_prob=0.95, ci_kind="eti", round_to=3
)
print(summary)

az.plot_trace(
    idata, var_names=parameter_names, backend="matplotlib", col_wrap=3,
    figure_kwargs={"figsize": (12, 9), "layout": "constrained"},
)
plt.show()

az.plot_rank(
    idata, var_names=parameter_names, backend="matplotlib", col_wrap=3,
    figure_kwargs={"figsize": (12, 9), "layout": "constrained"},
)
plt.show()
```

The trace plots should show chains moving through the same range without
persistent drift or long stuck segments. Rank plots should not show a chain
systematically occupying different ranks from the others. Use the numerical
diagnostics as well; a visually overlapping trace alone is insufficient.

| Check                    | Initial target                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `r_hat`                  | Below 1.01 for every reported parameter or derived quantity                             |
| `ess_bulk`, `ess_tail`   | At least 400 across four chains as a starting target; increase for the precision needed |
| `mcse_mean`              | Small relative to the posterior `sd` and the required reporting precision               |
| Divergences after warmup | Zero                                                                                    |
| Maximum-treedepth hits   | Zero unresolved hits                                                                    |
| E-BFMI                   | Values below 0.3 call for investigation                                                 |

These are screening criteria, not proof of convergence. For interpretation, see
[Stan's sampler diagnostic guide](https://mc-stan.org/docs/cmdstan-guide/diagnose_utility.html).
`sd` measures posterior uncertainty; `mcse_mean` measures numerical uncertainty
in the estimated posterior mean. More draws can reduce Monte Carlo error, but
will not resolve a nonidentifiable parameterization. In this model, inspect the
Ar39 shape correlations in particular. Persistent problems may require better
parameterization or scientifically justified calibration priors; increasing
`adapt_delta` alone does not address every cause.

## Inspect parameter posteriors

After checking the chains, plot the marginal posterior for each parameter with a
95% equal-tailed credible interval, and inspect correlations between the five
Ar39 shape parameters. The intervals in `summary` above use the same convention.
See
[ArviZ's posterior density plotting reference](https://python.arviz.org/projects/plots/en/stable/api/generated/arviz_plots.plot_dist.html).

```python
az.plot_dist(
    idata,
    var_names=parameter_names,
    ci_prob=0.95,
    ci_kind="eti",
    point_estimate="mean",
    backend="matplotlib",
    col_wrap=3,
    figure_kwargs={"figsize": (12, 9), "layout": "constrained"},
)
plt.show()

az.plot_pair(
    idata,
    var_names=["mu", "sigma", "tau", "base_alpha", "base_beta"],
    backend="matplotlib",
)
plt.show()
```

Broad or strongly correlated shape posteriors can indicate that the spectrum
constrains combinations of parameters better than individual ones. A narrow
posterior can also reflect a restrictive prior, so assess prior sensitivity.

## Plot the fitted spectrum against the data

Use Stan's component bin expectations for every retained posterior draw. This
preserves parameter correlations and the exact bin integrals. Evaluating the
density at the posterior mean parameters generally gives a different curve from
averaging the predicted spectrum over the posterior.

The following plot shows bin-averaged densities using the exact histogram passed
to Stan. Set `exposure_kg_yr` to the group's exposure (for example,
`ar_spectrum.expo`) to obtain the same counts/(keV kg yr) units as the Python
plots. Leave it as `None` for counts/keV. Use the exposure corresponding to the
counts being fitted; it is used here only for display.

```python
edges = np.asarray(data["edges"], dtype=float)
counts = np.asarray(data["counts"])
widths = np.diff(edges)
centres = 0.5 * (edges[:-1] + edges[1:])

exposure_kg_yr = None  # Replace with ar_spectrum.expo for this detector group.
norm = widths.copy()
ylabel = "Counts / keV"
if exposure_kg_yr is not None:
    norm *= float(exposure_kg_yr)
    ylabel = r"Counts / (keV kg yr)"

# Shape: (all retained draws from all chains, n_bins).
total_draws = fit.stan_variable("expected_counts") / norm
lo95, lo68, hi68, hi95 = np.quantile(
    total_draws, [0.025, 0.16, 0.84, 0.975], axis=0
)

fig, ax = plt.subplots(figsize=(11, 6))
ax.fill_between(
    edges, np.r_[lo95, lo95[-1]], np.r_[hi95, hi95[-1]],
    step="post", color="C3", alpha=0.12, label="95% credible band",
)
ax.fill_between(
    edges, np.r_[lo68, lo68[-1]], np.r_[hi68, hi68[-1]],
    step="post", color="C3", alpha=0.25, label="68% credible band",
)
ax.stairs(total_draws.mean(axis=0), edges, color="C3", label="Total posterior mean")

for variable, label, color in [
    ("expected_ar39", r"$^{39}$Ar", "C0"),
    ("expected_2nu", r"$2\nu\beta\beta$", "C1"),
    ("expected_linear", "Linear", "C2"),
]:
    component_mean = fit.stan_variable(variable).mean(axis=0) / norm
    ax.stairs(component_mean, edges, color=color, label=label)

ax.plot(centres, counts / norm, ".", color="black", label="Data", zorder=5)
# Shade bins left out of the likelihood (use_bin = 0).
for b in np.flatnonzero(np.asarray(data["use_bin"]) == 0):
    ax.axvspan(edges[b], edges[b + 1], color="grey", alpha=0.2, lw=0)
ax.set(xlabel="Energy [keV]", ylabel=ylabel, xlim=(edges[0], edges[-1]))
# ax.set_yscale("log")  # Optional; zero-count bins will not be visible.
ax.legend()
fig.tight_layout()
plt.show()
# fig.savefig("background_fit.png", dpi=200)
```

The component posterior means sum to the plotted total mean. The bands are
pointwise equal-tailed intervals for the expected spectrum, not simultaneous
bands over all bins. They describe parameter uncertainty and do not include the
Poisson fluctuations of a new histogram. These predictions cover the fitted
bins; plotting a wider interval requires evaluating the model there.

## Check posterior predictive counts

Use `counts_rep` to include both parameter uncertainty and Poisson variation.
The wider predictive intervals are useful for assessing the observed scatter.
Run this after the preceding plotting example, which defines the bin edges,
normalization, labels, and total spectrum draws.

```python
replicated_draws = fit.stan_variable("counts_rep") / norm
pred_lo, pred_hi = np.quantile(replicated_draws, [0.025, 0.975], axis=0)

fig_ppc, ax_ppc = plt.subplots(figsize=(11, 5))
ax_ppc.fill_between(
    edges, np.r_[pred_lo, pred_lo[-1]], np.r_[pred_hi, pred_hi[-1]],
    step="post", color="C0", alpha=0.2, label="95% posterior predictive interval",
)
ax_ppc.stairs(total_draws.mean(axis=0), edges, color="C3", label="Total posterior mean")
ax_ppc.plot(centres, counts / norm, ".", color="black", label="Data", zorder=5)
ax_ppc.set(xlabel="Energy [keV]", ylabel=ylabel, xlim=(edges[0], edges[-1]))
ax_ppc.legend()
fig_ppc.tight_layout()
plt.show()
```

Look for systematic excesses or deficits across adjacent bins, or spectral
features missing from the model. A few points outside a 95% pointwise interval
can occur; these intervals are not a global goodness-of-fit test. Chain
diagnostics assess sampling reliability, while predictive checks assess whether
the model describes the spectrum. Repeat the checks for each group.

## Fix parameters

Stan cannot pin a declared parameter to a value. A fixable parameter is
therefore declared as an array whose length is a data flag, and the value used
by the model is defined in `transformed parameters`:

```stan
data                   { int<lower=0, upper=1> fit_c0; real<lower=0> c0_fixed; }
parameters             { array[fit_c0] real<lower=0> c0_free; }
transformed parameters { real<lower=0> c0 = fit_c0 ? c0_free[1] : c0_fixed; }
model                  { target += lognormal_lpdf(c0_free | m, s); }
```

With `fit_c0 = 0`, `c0_free` has length 0: it is not sampled, its prior adds
nothing, and `c0` takes the value `c0_fixed`. This is a fixed parameter in the
BAT sense: the evidence of the reduced model integrates only over the remaining
parameters, $Z=\int p(n\mid\theta,c_0=c_{0,\rm fixed})\,p(\theta)\,d\theta$. Do
not imitate a fixed value with a very narrow prior. The lognormal has no density
at zero, a near-delta prior gives the sampler a funnel, and a Bayes factor would
then depend on the chosen width.

`c0` and `cQ` are fixable. `build_stan_data(..., fixed=...)` sets the flags and
values; `NO_LINEAR = {"c0": 0.0, "cQ": 0.0}` gives the model without the linear
continuum. Use the same compiled model and the same `priors` as for the full
fit, so the shared parameters keep identical priors:

```python
from bosonic_dm.fitting.background_fit import (
    NO_LINEAR, build_stan_data, inits_from_prefit, run_background_fit,
)

data_nl = build_stan_data(spectrum, priors, fixed=NO_LINEAR)
fit_nl = run_background_fit(
    model, data_nl, seed=12345,
    inits=inits_from_prefit(prefit, fixed=NO_LINEAR),
)
```

Initial values for fixable parameters use the array names (`c0_free = [value]`);
`inits_from_prefit` writes them, leaving fixed parameters out. Stan still writes
`c0` and `cQ`, with constant values. `summarize_background_fit` and the chain
and corner plots skip fixed parameters, and `expected_linear` is left out of the
component medians when it is identically zero.

To make another parameter fixable, for example `tau`:

1. Add `int<lower=0, upper=1> fit_tau;` and `real<lower=0> tau_fixed;` to
   `data`.
2. Replace `real<lower=0> tau;` in `parameters` with
   `array[fit_tau] real<lower=0> tau_free;`.
3. Define `real<lower=0> tau = fit_tau ? tau_free[1] : tau_fixed;` at the top of
   `transformed parameters`.
4. Put the prior on `tau_free`.
5. Add `"tau"` to `FIXABLE_PARAMETERS` in `fitting/background_fit.py`.
   `build_stan_data`, `inits_from_prefit`, and `background_log_density` read the
   fixable parameters from that tuple.

## Compare models

The Bayes factor of model $a$ against model $b$ is the ratio of their marginal
likelihoods (evidences),

$
\mathrm{BF}_{ab}=\frac{Z_a}{Z_b},\qquad
Z=\int p(n\mid\theta)\,p(\theta)\,d\theta .
$

`fitting/model_comparison.py` estimates each $\ln Z$ by bridge sampling (Meng &
Wong 1996). It uses the iterative scheme and normal proposal of Gronau et al.
(2017), as in the R `bridgesampling` package and BAT.jl's `BridgeSampling`:

- The free parameters are taken on the log scale, where Stan samples them. Let
  $q(\theta)$ be the full log posterior (likelihood, prior, and Jacobian), as
  computed by `background_log_density`. This NumPy mirror of `lp__` is checked
  against `lp__` on 200 posterior draws before sampling, and a mismatch raises.
- In each chain, the first half of the draws fits a multivariate-normal proposal
  $g$. The second half ($N_1$ draws, $N_{\rm eff}$ effective) and $N_2=N_1$
  draws from $g$ enter the fixed-point iteration, with $\ell=q/g$,
  $s_1=N_{\rm eff}/(N_{\rm eff}+N_2)$, and $s_2=1-s_1$:

  $
  \hat r^{(t+1)}=
  \frac{N_2^{-1}\sum_j \ell(\tilde\theta_j)/[s_1\ell(\tilde\theta_j)+s_2\hat r^{(t)}]}
       {N_1^{-1}\sum_i 1/[s_1\ell(\theta_i)+s_2\hat r^{(t)}]},
  \qquad \hat Z=\lim_t \hat r^{(t)} .
  $

- The uncertainty `log_ml_sd` is the square root of the relative mean-squared
  error of Frühwirth-Schnatter (2004). Its posterior term uses the effective
  sample size, so autocorrelation is included.

```python
from bosonic_dm.fitting.model_comparison import (
    compare_background_models, loo_compare_background_models,
)
from bosonic_dm.yaml_io import write_yaml

bf = compare_background_models(fit, data, fit_nl, data_nl)  # a = full, b = no linear
print(f"ln BF = {bf.log_bf:.2f} ± {bf.log_bf_sd:.2f}: {bf.interpretation}")
write_yaml(bf.as_dict(), "bayes_factor.yaml")

print(loo_compare_background_models(fit, data, fit_nl, data_nl))
```

`bf.log_bf` is $\ln\mathrm{BF}_{ab}$, and positive values favour model $a$.
`bf.probability_a` is the posterior probability of $a$ at equal prior odds.
`bf.interpretation` applies the Kass & Raftery (1995) scale to
$2\ln\mathrm{BF}$. `compare_background_models` rejects two fits that do not
share the spectrum, mask, fixed inputs, and the priors of the common parameters,
or that fix the same parameters.

The LOO table compares the expected log predictive density from PSIS-LOO
(`arviz.compare`). Each fitted bin is one left-out observation. This is a
predictive comparison rather than an evidence ratio, and it depends only weakly
on the priors. A Pareto-$k$ warning means the importance-sampling estimate is
unreliable for some bins.

Before trusting a Bayes factor:

- **Prior widths.** The evidence of the full model includes an Occam penalty set
  by the `c0`/`cQ` priors: widening them lowers $Z_a$. Report how
  $\ln\mathrm{BF}$ changes with them: for example, refit the full model with
  `priors["log_rate_prior_sd"][2:]` set to 0.5, 1, and 2. The no-linear fit
  ignores these entries and can be reused.
- **Data-centred priors.** Priors centred on a `curve_fit` prefit of the same
  spectrum use the data twice and favour the larger model. Priors from
  independent information (simulation, sidebands, calibration) avoid this.
- **Draws.** Bridge sampling uses only half of each chain for the iteration. Use
  more draws than for parameter estimation (e.g. `iter_sampling=4000`), check
  that `log_bf_sd` is much smaller than $|\ln\mathrm{BF}|$, and repeat with
  another `seed`. The normal proposal is less efficient when the Ar39 shape
  posterior is far from Gaussian on the log scale, which shows up as a larger
  `log_ml_sd`.
- **Sampler diagnostics.** Both fits must pass them; divergent or stuck chains
  bias the evidence as well as the parameters.
- **Old fits.** Fits made before `lp__` kept all constants are rejected by the
  `lp__` check. Refit them with the current model.

## Numerical verification

Numerical checks against the Python definitions, including nonuniform bins,
restricted windows, fixed linear coefficients, and automatic differentiation
versus finite-difference gradients, run with:

```sh
pixi run test tests/test_stan_background.py
```

`tests/test_model_comparison.py` checks the NumPy log density against Stan's
`log_prob` for each variant and bridge sampling against analytically known
evidences, and it fits a toy spectrum with both variants.

These checks require an installed CmdStan toolchain; they skip if it is absent.
