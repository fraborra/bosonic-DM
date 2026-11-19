# Spectral fits

The binned background and signal fits are Python functions under
`bosonic_dm.fitting`, sampled with [CmdStanPy](https://mc-stan.org/cmdstanpy/).
They are not exposed as `bosonic-dm` subcommands. Run them from scripts or
notebooks, for example `notebooks/bkg-stan-fit.ipynb`.

> **Status:** the background fit is being validated. The signal fit is a first
> stage: it fixes the background and the signal response at nominal values, so
> its limit does not include background or response uncertainties. Neither fit
> is ready for a published result.

The Stan models are in `stan/`. [`stan/README.md`](../stan/README.md) gives the
background likelihood, its parameters, and a step-by-step example with chain
checks and posterior predictive plots.

## Workflow

```text
background Parquet dataset
  └─> prepare_background_spectra  per-group histograms, exposure, fit_mask
        └─> background fit         stan/background.stan, one group at a time
              └─> B_hat            fixed per-bin background prediction
                    └─> signal fit stan/signal.stan, one mass at a time
```

## Prepare the binned spectra

`bosonic_dm.background.prepare_background_spectra` builds, for each detector
group, the analysis and comparison histograms, their exposure-normalized rates,
the group exposure, and a `fit_mask`. Bins that overlap
`[centre - exclusion_width_keV, centre + exclusion_width_keV]` around any of
`peak_centres_keV` get `fit_mask = False`. Their counts are kept, but the fits
leave them out of the likelihood. `spectrum_in_range` cuts a prepared group to a
fit window without rebinning.

The group exposure counts only detector/runs with usability `on`. The background
Parquet dataset is not filtered by usability, so pass a usability table to keep
the counts consistent with that exposure:

```python
from bosonic_dm.cuts import build_usability_table

period_runs = background.select("period", "run").unique().collect().rows()
usable = build_usability_table(context.get_channelmap_for_run, period_runs)

spectra = prepare_background_spectra(
    background,
    groups_dict,
    eres_dict,
    peak_centres_keV=peak_centres,
    usable=usable,
)
```

`build_usability_table` keeps geds that are processable and have usability `on`
in each run. `usability_table_from_eres` builds the same table from an exposure
dictionary, but cannot check `processable`. `run_background_fit` logs a reminder
of this requirement on every call.

## Background fit

`stan/background.stan` fits one group's raw counts with a 2νββ Primakoff–Rosen
shape, a linear continuum, and a modified-beta Ar39 shape. The helper functions
in `bosonic_dm.fitting.background_fit` are used in this order:

| Function                                      | Purpose                                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `priors_from_prefit`                          | Lognormal priors centred on a `curve_fit` prefit (`fit_ar` entry).                         |
| `inits_from_prefit`                           | Per-chain initial values near the prefit, with multiplicative jitter.                      |
| `build_stan_data`                             | Stan data from a prepared spectrum; `fit_mask` becomes `use_bin`. `fixed` fixes `c0`/`cQ`. |
| `run_background_fit`                          | Sample with CmdStan (4 chains, 1000 warmup and 1000 sampling by default).                  |
| `summarize_background_fit`                    | Parameter summaries, sampler diagnostics, and the frozen `B_hat`.                          |
| `save_background_fit` / `load_background_fit` | Write and read the fit products below.                                                     |

The linear coefficients can be fixed for a run with
`build_stan_data(..., fixed={"c0": 0.0, "cQ": 0.0})` (`NO_LINEAR`), which gives
the model without the linear continuum from the same compiled Stan model. A
fixed parameter is removed from the sampled space and has no prior, as in BAT;
`stan/README.md` describes the mechanism and how to make other parameters
fixable.

Sampler checks are shared with the signal fit through
`bosonic_dm.fitting.diagnostics`. A fit passes when it has no divergences, fewer
than 5% of its post-warmup draws hit the maximum tree depth (12 in the
background fit), E-BFMI is at least 0.3, R-hat is below 1.01, and bulk and tail
effective sample sizes are at least 400. A fit with a non-finite E-BFMI, R-hat,
or effective sample size fails. Check the summary's diagnostics before using
`B_hat`.

`bosonic_dm.plotting.background_fit` plots normalized residuals with the
excluded peak windows, trace plots, and corner plots.

### Products

`save_background_fit` writes into the directory you pass:

| File                        | Content                                                             |
| --------------------------- | ------------------------------------------------------------------- |
| `posterior.nc`              | Full posterior as ArviZ InferenceData.                              |
| `bhat.parquet`              | Bin edges, `B_hat`, `used_in_fit`, and per-component medians.       |
| `expected_counts_draws.npy` | Thinned draws of the expected counts, shape `(draws, bins)`.        |
| `summary.yaml`              | Parameter summaries, diagnostics, sampler settings, and provenance. |

Pass `provenance` (group name, input dataset, peak list, fit window) so the
summary records how the fit was made. `summary.yaml` also records the fixed
parameters (`fixed_parameters`).

### Model comparison

`bosonic_dm.fitting.model_comparison` compares two variants of the background
fit on the same spectrum, typically the full model against `NO_LINEAR`:

| Function                             | Purpose                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------- |
| `compare_background_models`          | Bayes factor from bridge-sampling estimates of both log evidences.        |
| `loo_compare_background_models`      | PSIS-LOO comparison (leave-one-bin-out) with `arviz.compare`.             |
| `background_log_marginal_likelihood` | Log evidence of one fit, after checking the NumPy density against `lp__`. |
| `bridge_sampling`                    | Generic bridge-sampling estimator for draws on an unconstrained scale.    |

The Bayes factor depends on the priors of the parameters that differ between the
variants, here the `c0`/`cQ` widths. Priors centred on a prefit of the same data
favour the larger model. See
[`stan/README.md`](../stan/README.md#compare-models) for the method, the number
of draws to use, and the checks to make before quoting a result.

## Signal fit

The signal fit uses the saved background as fixed input. For group `g` and bin
`i`:

```text
counts[g, i] ~ Poisson(B_hat[g, i] + R * S[g, i](m))
S[g, i](m) = exposure_g * efficiency_g(m) * P[g, i](m)
```

`P` is a Gaussian centred at the mass `m`. Its width is moment matched: the
squared group FWHM is the exposure-weighted mean of the detector/run FWHM², so
the Gaussian has the variance of the exposure-weighted mix of the detector/run
Gaussians. This neglects the efficiency differences between detector/runs in the
weights and the spread of their centroids. The rate `R`, in counts/(kg yr), is
shared by all groups and has a uniform prior on `[0, rate_upper]`. Bins masked
in the background fit are also masked here.

`bosonic_dm.fitting.signal_template` builds the inputs:

- `load_background_prediction` reads `B_hat` from a background fit directory.
- `group_efficiency` aggregates the run-level simulation efficiencies into
  detector groups. Pass it the efficiency table
  (`<interaction>_efficiency.parquet`, read with
  `bosonic_dm.efficiency.load_efficiency_table`). The `"all"` selection matches
  `passes_analysis` in the background data.
- `group_fwhm` returns the moment-matched FWHM and, as its uncertainty, the
  uncertainty of the exposure-weighted mean FWHM. `group_centroid_offset` gives
  an optional calibration offset per group.
- `build_signal_window` cuts the fit window (default half-width 12.5 keV, 24
  bins of 1 keV) and builds each group's template. If the background spectra
  were prepared with a non-default `exposure_key`, pass the same key here so the
  group exposure and the resolution weights use the same field.

`bosonic_dm.fitting.signal_fit` then runs the fit:

- `suggest_rate_upper` and `build_signal_stan_data` set the prior bound and the
  Stan data.
- `run_signal_fit` samples `stan/signal.stan` (1000 warmup and 2000 sampling
  iterations by default).
- `summarize_signal_fit` reports the rate posterior and the credible upper
  limit, by default at 90%. The fit fails its checks if the limit exceeds half
  the prior bound.
- `save_signal_fit` writes `rate_draws.npy` and `summary.yaml`.

For many datasets at once, such as pseudo-experiments,
`rate_posterior_quantiles` and `rate_upper_limits` compute the same posterior
without sampling. They integrate the likelihood on a grid in `R >= 0`. A coarse
grid is doubled until the likelihood at its end is below 1e-8 of its maximum,
then 1000 points cover the range where it is above that fraction. The result
therefore does not depend on the prior bound, as long as the bound lies above
the grid. Counts can be non-integer (Asimov data) and have shape
`(datasets, groups, bins)` or `(groups, bins)`. `rate_upper_limits` returns one
limit per dataset, in counts/(kg yr); it matches the Stan limit within its Monte
Carlo error.

### Checking one mass

`scripts/signal_fit_one_mass.py` runs the chain above at one mass from the
pipeline outputs named by a config: the background fits in
`<data_root>/background_fits/`, `eres_dict.yaml` and the detector groups under
`inputs_root`, and the efficiency table in `<data_root>/dictionaries/`.

```bash
PYTHONPATH=src pixi run python3 scripts/signal_fit_one_mass.py \
    --config configs/nersc.yaml --mass 500 --outdir tmp/signal_500
```

It is blinded by default: the observed counts in the window are never fitted,
plotted or written. The script:

1. logs, per group, the exposure, efficiency, moment-matched FWHM, window
   containment, and the window sums of `B_hat` and of the signal template;
2. computes the upper limit on the Asimov dataset (counts = `B_hat`) with the
   grid posterior; Stan needs integer counts, so the Asimov limit is grid only;
3. draws one Poisson toy from `B_hat`, fits it with Stan and with the grid
   posterior, and checks that the two limits agree within 4 Monte Carlo standard
   errors;
4. writes `summary.yaml` and `rate_draws.npy` (the Stan fit of the toy),
   `checks.yaml`, and `window.png` (toy, `B_hat`, and `B_hat + R90 S`).

`--inject R` adds `R * S` to the toy; the posterior should recover `R`
(`injection_pull` in `checks.yaml`, within about ±3). This checks that Python
and Stan see the same data (group and bin order) and that the fit is unbiased
for `R > 0`. It does not check the units of `S`: the same template generates and
fits the toy, so a wrong exposure or efficiency cancels. `--unblind` fits the
observed counts instead of a toy.

The reported limit is conditional on the fixed `B_hat` and the nominal response.
A mass scan, pseudo-experiments, and the conversion to a coupling are not
implemented yet.
