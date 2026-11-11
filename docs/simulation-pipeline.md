# Simulation pipeline

The simulation workflow converts per-energy simulation products into Parquet
datasets, efficiency dictionaries, and summary plots.

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction dark-compton
```

## Input discovery

For each configured energy, the interaction's `job_template` is expanded and the
pipeline searches for:

```text
<simulation_root>/generated/tier/evt/<job-name>/
  l200cfg01-<job-name>-job_*-tier_evt.lh5

<simulation_root>/generated/tier/stp/<job-name>/
  l200cfg01-<job-name>-job_*-tier_stp.lh5
```

The per-job evt files are the only event input: they hold the events and the
period/run every event was assigned to. The merged cvt file is not read. The STP
files give the primaries; an energy needs the same job numbers in both tiers,
otherwise its efficiency is skipped with a warning.

The geometry used for vertex counting is taken from the first configured energy:

```text
<simulation_root>/generated/pars/geom/
  l200cfg01-<first-job-name>-tier_stp-geom.gdml
```

Efficiency calculation additionally needs:

- `<inputs_root>/dictionaries/eres_per_det_tot.yaml`.
- `<reference_root>/<version>/config.json`.
- The detector grouping configured by `analysis.detector_groups` for grouped
  plots.

Some advanced plots also use
`<inputs_root>/dictionaries/rawid_by_det_type.yaml`.

## Stages

The execution order is:

```text
build-dataset ─> efficiencies ─> plots
```

Counting the primaries is part of `efficiencies`; there is no `count-vertices`
stage and no primary-counts file.

### `build-dataset`

Builds one Parquet partition per energy from the evt files of all its jobs,
keeping the multiplicity-one events:

```text
<data_root>/parquet/<interaction>/
  sim_e=<energy>/
    data.parquet
```

### `efficiencies`

Combines primary counts, Parquet events, channel metadata, energy resolution,
the configured FEP window, selections, and LAr-veto setting. The result is:

```text
<data_root>/dictionaries/<interaction>_efficiency.parquet
```

Missing energy inputs are skipped and recorded; the pipeline does not invent a
denominator for incomplete energy points.

#### Primary counts

The stage first needs the primaries generated in each detector **for each
period/run**. It maps the STP vertices to detectors with the GDML geometry. The
simulation assigns the primaries of a job to runs in contiguous `evtid` blocks;
the blocks are read from the evt file of the same job (`read_run_ranges`) and
applied to the vertex `evtid`s. `evtid` restarts in every job, which is why the
per-job files are used.

The counts are stored only in the `n_primaries` column of the efficiency table.
Counting is slow, so the stage reuses what the table already holds:

| Situation                                    | Primary counts                         |
| -------------------------------------------- | -------------------------------------- |
| Cut setup unchanged                          | Stage is `cached`; nothing is counted. |
| FEP window, selections or LAr veto changed   | Reused from the table.                 |
| Energy not yet in the table                  | Counted for that energy only.          |
| `--overwrite`, or a table in an older format | Counted again for every energy.        |

The manifest lists the reused and the counted energies under
`stages.efficiencies.primary_counts`. Use `--overwrite` after changing the
simulation files or the calibration input: a detector/run that is new to the
calibration input has no stored count, gets `n_primaries = 0` and the status
`missing-primaries`, and the pipeline warns about it.

The table has one row per simulated energy, detector, period, run, and
selection:

| Columns                                                       | Content                                                 |
| ------------------------------------------------------------- | ------------------------------------------------------- |
| `sim_e`, `detector`, `period`, `run`, `selection`             | Row keys; `sim_e` is the simulated energy in keV.       |
| `status`                                                      | `valid`, or the reason the result is unavailable.       |
| `psd_available`                                               | Detector-level flag, repeated on each of its rows.      |
| `n_primaries`                                                 | Primaries generated in that detector for that run.      |
| `exposure_kg_yr`, `fwhm_keV`, `fwhm_unc_keV`                  | Exposure and resolution of that detector in that run.   |
| `n_events`                                                    | Selected events inside the run's FEP window.            |
| `efficiency_mle`, `efficiency`, `efficiency_stat_unc`         | Efficiency and its statistical uncertainty.             |
| `efficiency_syst_fwhm`                                        | Half the change when the window follows FWHM ± unc.     |
| `effective_exposure_kg_yr` and its `_stat_unc` / `_syst_fwhm` | Exposure times efficiency, with the same uncertainties. |

Result columns are null when `status` is not `valid`; such rows keep the run
exposure for diagnosis. Read the table with Polars, filtering by energy to keep
memory low:

```python
from bosonic_dm.efficiency import load_efficiency_table

efficiencies = load_efficiency_table(
    "data/v1/dictionaries/dark-compton_efficiency.parquet",
    energy_range_keV=(400, 600),
)
```

Detector and group averages are not stored. Compute them with
`aggregate_run_aware_efficiencies` (detector type or detector group), or with
`bosonic_dm.fitting.signal_template.group_efficiency` for the signal fit.

A run's efficiency uses only the events the simulation assigned to that run,
inside the FEP window set by that run's resolution, divided by the primaries
generated in the detector for that run.

#### Combining runs and detectors

The run-level `efficiency` and `efficiency_stat_unc` are the Jeffreys posterior
mean and standard deviation of that run alone. They are **not** averaged to get
a detector or group value: that would put in one prior per run and pull the
result towards 0.5 by an amount that does not shrink as runs are added
(D'Agostini, _Bayesian Reasoning in Data Analysis_, Sec. 8.11). The counts are
combined first (`combine_run_efficiencies`):

1. Each run's events and primaries are weighted by its exposure divided by its
   primaries. The ratio of the weighted sums is the exposure-weighted mean of
   the run-level `n_events / n_primaries`.
2. One Jeffreys prior is applied to the combined counts, using the effective
   number of primaries `(sum X)^2 / sum(X^2 / n)`, with `X` the exposure and `n`
   the primaries of a run.

When primaries are proportional to exposure, this is the plain sum of events
over the plain sum of primaries, and the result is exact. Otherwise the
effective primaries are fewer than the generated ones and the posterior is an
approximation. The FWHM systematic is the exposure-weighted mean of the
run-level values.

Earlier versions wrote `<interaction>_efficiency.yaml` and
`<interaction>_primary-counts.yaml`. Both are obsolete: they hold one count per
detector and efficiencies computed on the whole sample for every run. The next
run of the `efficiencies` stage counts the primaries per run and writes the
table; the two YAML files can then be deleted. Existing dataset partitions stay
valid.

### `plots`

Creates efficiency, effective-exposure, and FEP-survival summaries grouped by
detector type and detector group. Interaction flags can enable energy spectra,
LAr-survival, and AoE-survival plots.

## Running selected stages

`--stage` accepts one or more stage names. Dependencies are resolved
automatically, so this command requests the complete chain needed for plots:

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction dark-compton \
  --stage plots
```

Use `--overwrite` to rebuild existing products. Without it, existing energy
partitions can be reused, and efficiency or plot products are reused only when
their manifest records match the relevant cut and plot setup.

## Partial runs

When only some energies are available, the workflow still produces results for
those energies. A stage may be `completed`, `cached`, `partial`, `blocked`, or
`disabled`. Inspect `<data_root>/<interaction>_manifest.yaml` for the exact
status and warnings from the last run.
