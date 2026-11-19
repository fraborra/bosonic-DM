# Configuration reference

The `bosonic-dm` command requires a YAML configuration with `schema_version: 1`.
Start from `configs/example-local.yaml`; `configs/default.yaml` shows a more
complete set of analysis values.

## Path behavior

Environment variables in configured paths are expanded. Relative paths are
resolved from the directory in which the command is run, not from the YAML
file's directory. The loader creates `data_root`, its `parquet/` and
`dictionaries/` children, `plots_root`, and `temporary_root` when needed.

`parquet_root` and `dictionaries_root` are derived automatically:

```text
<data_root>/parquet/
<data_root>/dictionaries/
```

## Complete example

```yaml
schema_version: 1

production:
  version: v2.1.5
  reference_root: /path/to/legend/production
  metadata_override: null

paths:
  simulation_root: /path/to/simulation
  data_root: data/v1
  inputs_root: data/inputs
  plots_root: plots
  temporary_root: tmp

analysis:
  simulated_energies_keV:
    start: 100
    stop: 1000
    step: 100
  fep_window:
    half_width_fwhm: 2.0
  selections: [all, valid-psd, sse, mse]
  apply_lar_veto: true
  efficiency_run_substitutes:
    p09-r004: p09-r005
  detector_groups: dictionaries/detector-grouping/groups_dict.yaml

interactions:
  axio-electric:
    job_template: electron_{energy}keV_hpge_bulk
    make_lar_survival_plots: false
    make_energy_spectra_plots: false
    make_aoe_survival_plots: false
  dark-compton:
    job_template: fromfile_dark_compton_{energy}keV_hpge_bulk
    make_lar_survival_plots: true
    make_energy_spectra_plots: false
    make_aoe_survival_plots: false

background:
  # If omitted, the PET path is read from the production config.json.
  pet_glob: /path/to/pet/phy/*.lh5
  comparison_cut_profile: without-bb-like
  energy_ranges_keV: [[20, 300], [20, 1000]]
  bin_widths_keV: [1, 5, 10]

output:
  overwrite: false
  save_plots: true
  write_manifest: true
```

## Top-level sections

### `production`

| Key                 | Meaning                                           |
| ------------------- | ------------------------------------------------- |
| `version`           | Production version below `reference_root`.        |
| `reference_root`    | Root containing production versions and metadata. |
| `metadata_override` | Optional metadata path override; may be `null`.   |

### `paths`

| Key               | Meaning                                          | Default       |
| ----------------- | ------------------------------------------------ | ------------- |
| `simulation_root` | Simulation production root.                      | required      |
| `data_root`       | Generated datasets, dictionaries, and manifests. | required      |
| `inputs_root`     | Static analysis inputs.                          | `data/inputs` |
| `plots_root`      | Generated plots.                                 | `plots`       |
| `temporary_root`  | Temporary products.                              | `tmp`         |

### `analysis`

`simulated_energies_keV` accepts either an explicit list or an inclusive range:

```yaml
simulated_energies_keV: [200, 300, 500]
```

```yaml
simulated_energies_keV:
  start: 100
  stop: 1020
  step: 10
```

Energies must be positive. Duplicates are removed and the values are sorted. The
range's default step is 10 keV.

| Key                          | Meaning                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| `fep_window.half_width_fwhm` | Positive half-width of the full-energy-peak selection in units of detector FWHM.     |
| `selections`                 | Efficiency selections to compute.                                                    |
| `apply_lar_veto`             | Apply the LAr veto to simulation efficiencies and the background analysis selection. |
| `detector_groups`            | Detector-group dictionary; relative values are resolved below `inputs_root`.         |
| `efficiency_run_substitutes` | Runs missing from the simulation, mapped to the run whose efficiency they use.       |

`efficiency_run_substitutes` maps `period-run` labels, e.g.
`p09-r004: p09-r005`. The replaced run keeps its own exposure and FEP window,
but its events and primaries come from the source run, which counts only once
when runs are combined. The background fit is not affected: it uses data counts
and exposure, which the replaced run has. A source run cannot itself be
replaced. Changing the map recomputes the efficiencies; stored primary counts
are reused.

### `interactions`

Each mapping key defines an interaction accepted by `--interaction`.
`job_template` is required and must contain `{energy}`. The three optional plot
flags default to `false`.

The simulation pipeline looks for the expanded job name below:

```text
<simulation_root>/generated/tier/evt/<job-name>/
<simulation_root>/generated/tier/stp/<job-name>/
```

### `background`

| Key                      | Meaning                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `pet_glob`               | Optional PET file glob. An empty value uses the production `config.json`. Environment variables are expanded. |
| `comparison_cut_profile` | Alternate cut profile stored and plotted alongside the default selection.                                     |
| `energy_ranges_keV`      | Energy intervals used for diagnostic spectrum plots.                                                          |
| `bin_widths_keV`         | Bin widths used for each configured interval.                                                                 |
| `fit`                    | Optional Stan fit settings, below. Every key has a default.                                                   |

#### `background.fit`

| Key                        | Default                                                            | Meaning                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spectrum_range_keV`       | `[60, 2000]`                                                       | Energy range of the spectrum that is built.                                                                                                                 |
| `spectrum_bin_width_keV`   | `1`                                                                | Spectrum bin width.                                                                                                                                         |
| `fit_range_keV`            | `[100, 1100]`                                                      | Energy range used in the fit.                                                                                                                               |
| `peaks_keV`                | `standard`                                                         | `standard` (the built-in table of 20 gamma lines) or a list of centres in keV; `[]` masks none. Lines whose mask falls outside `fit_range_keV` are dropped. |
| `peak_exclusion_width_keV` | `5.0`                                                              | Half-width of the mask around each line.                                                                                                                    |
| `prefit_dictionary`        | `dictionaries/prefit_dictionary.yaml`                              | Prefit file, relative to `inputs_root`.                                                                                                                     |
| `prefit_key`               | `fit_ar`                                                           | Entry of the prefit file to use.                                                                                                                            |
| `model`                    | `no_linear`                                                        | `no_linear` or `full`.                                                                                                                                      |
| `shape_sd`                 | `0.5`                                                              | Prior width on the shape parameters.                                                                                                                        |
| `rate_prior_sd`            | `{n2nu: 0.5, ar39: 2.0}`                                           | Prior widths on the rates.                                                                                                                                  |
| `sampler`                  | `{chains: 4, iter_warmup: 1000, iter_sampling: 1000, seed: 12345}` | Stan sampler settings.                                                                                                                                      |

### `output`

| Key              | Meaning                         | Default |
| ---------------- | ------------------------------- | ------- |
| `overwrite`      | Rebuild existing products.      | `false` |
| `save_plots`     | Generate pipeline plots.        | `true`  |
| `write_manifest` | Write cumulative run manifests. | `true`  |

The command-line `--overwrite` flag takes precedence over the configured value.
