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

| Function                                      | Purpose                                                                   |
| --------------------------------------------- | ------------------------------------------------------------------------- |
| `priors_from_prefit`                          | Lognormal priors centred on a `curve_fit` prefit (`fit_ar` entry).        |
| `inits_from_prefit`                           | Per-chain initial values near the prefit, with multiplicative jitter.     |
| `build_stan_data`                             | Stan data from a prepared spectrum; `fit_mask` becomes `use_bin`.         |
| `run_background_fit`                          | Sample with CmdStan (4 chains, 1000 warmup and 1000 sampling by default). |
| `summarize_background_fit`                    | Parameter summaries, sampler diagnostics, and the frozen `B_hat`.         |
| `save_background_fit` / `load_background_fit` | Write and read the fit products below.                                    |

Sampler checks are shared with the signal fit through
`bosonic_dm.fitting.diagnostics`. A fit passes when it has no divergences and no
maximum-tree-depth hits, E-BFMI is at least 0.3, R-hat is below 1.01, and bulk
and tail effective sample sizes are at least 400. Check the summary's
diagnostics before using `B_hat`.

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
summary records how the fit was made.

## Signal fit

The signal fit uses the saved background as fixed input. For group `g` and bin
`i`:

```text
counts[g, i] ~ Poisson(B_hat[g, i] + R * S[g, i](m))
S[g, i](m) = exposure_g * efficiency_g(m) * P[g, i](m)
```

`P` is a Gaussian centred at the mass `m`, with the group's exposure-weighted
FWHM. The rate `R`, in counts/(kg yr), is shared by all groups and has a uniform
prior on `[0, rate_upper]`. Bins masked in the background fit are also masked
here.

`bosonic_dm.fitting.signal_template` builds the inputs:

- `load_background_prediction` reads `B_hat` from a background fit directory.
- `group_efficiency` aggregates the run-level simulation efficiencies
  (`<interaction>_efficiency.yaml`) into detector groups. The `"all"` selection
  matches `passes_analysis` in the background data.
- `group_fwhm` and `group_centroid_offset` give the resolution and an optional
  calibration offset per group.
- `build_signal_window` cuts the fit window (default half-width 12.5 keV, 24
  bins of 1 keV) and builds each group's template.

`bosonic_dm.fitting.signal_fit` then runs the fit:

- `suggest_rate_upper` and `build_signal_stan_data` set the prior bound and the
  Stan data.
- `run_signal_fit` samples `stan/signal.stan` (1000 warmup and 2000 sampling
  iterations by default).
- `summarize_signal_fit` reports the rate posterior and the credible upper
  limit, by default at 90%. The fit fails its checks if the limit exceeds half
  the prior bound.
- `save_signal_fit` writes `rate_draws.npy` and `summary.yaml`.

The reported limit is conditional on the fixed `B_hat` and the nominal response.
A mass scan, pseudo-experiments, and the conversion to a coupling are not
implemented yet.
