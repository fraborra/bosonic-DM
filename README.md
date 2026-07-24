# bosonic-DM

Analysis tools for bosonic dark-matter searches with the LEGEND-200 experiment.
The package provides reusable Python functions and command-line pipelines for
simulation processing, efficiency estimation, plots, and background studies.

> **Status:** under active development. In particular, the background-analysis
> pipeline is not yet fully implemented. Treat its products as development
> outputs until the analysis is validated.

## Requirements and installation

The project requires Python 3.11 or newer and uses [Pixi](https://pixi.sh/) to
manage its environment. From the repository root:

```bash
pixi install
pixi run pip install --editable .
```

Use `pixi run` for all project commands. To run the test suite:

```bash
pixi run test
```

## Analysis configuration

The `bosonic-dm` pipeline is configured with a YAML file. It expands environment
variables in path values and creates the configured output directories as
needed. No site-specific configuration is committed to this repository, so
create one for your data location, for example `configs/local.yaml`:

```yaml
schema_version: 1

production:
  version: v2.1.5
  reference_root: /path/to/legend/reference
  # metadata_override: /optional/path/to/metadata

paths:
  simulation_root: /path/to/simulation
  data_root: data/v1
  # Optional; these default to data/inputs, plots, and tmp.
  inputs_root: data/inputs
  plots_root: plots
  temporary_root: tmp

analysis:
  simulated_energies_keV: [200, 300, 400]
  fep_window:
    half_width_fwhm: 1.0
  selections: []
  # Relative paths are resolved below inputs_root.
  detector_groups: dictionaries/detector-grouping/groups_dict.yaml

interactions:
  axio-electric:
    job_template: axio-electric_{energy}keV
  dark-compton:
    job_template: dark-compton_{energy}keV
    make_lar_survival_plots: true
    make_energy_spectra_plots: true
    make_aoe_survival_plots: true

background:
  pet_glob: /path/to/pet/*.lh5
  apply_lar_veto: true
  comparison_cut_profile: default
  energy_ranges_keV: [[0, 3000]]
  bin_widths_keV: [1]

output:
  overwrite: false
  save_plots: true
  write_manifest: true
```

Each interaction's `job_template` must contain `{energy}`. The simulation
pipeline searches for corresponding converted and step-tier LH5 files below
`simulation_root/generated/tier/`. Calibration and other analysis dictionaries
are stored under `data_root/dictionaries/`; input detector-group dictionaries
are supplied in `data/inputs/`.

## Main CLI

Run a single simulated interaction:

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction dark-compton
```

Run both simulation interactions and then the background pipeline:

```bash
pixi run bosonic-dm all --config configs/local.yaml
```

The `all` command accepts a subset with `--interactions`, while `simulation`
accepts exactly one `--interaction`:

```bash
pixi run bosonic-dm all \
  --config configs/local.yaml \
  --interactions axio-electric dark-compton
```

Run only the background pipeline:

```bash
pixi run bosonic-dm background --config configs/local.yaml
```

Pass `--overwrite` to replace existing products. `--stage` accepts one or more
stages. Simulation stages are `count-vertices`, `build-dataset`, `efficiencies`,
and `plots`; dependencies are added automatically. For example, requesting
`plots` runs all required preceding stages:

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction dark-compton \
  --stage plots \
  --overwrite
```

If enabled in `output.write_manifest`, each simulation run writes
`<data_root>/<interaction>_manifest.yaml`. The manifest records requested and
resolved stages, output paths, skipped work, and non-fatal warnings. In
particular, missing energy inputs are reported rather than used to produce an
efficiency with an incomplete denominator.

## Additional commands

### Generate Dark Compton macros

Generate GPS macro YAML files in `tmp/` for the supplied dark-matter masses
(keV):

```bash
pixi run generate-dark-compton --energies 200 300 500
```

Without `--energies`, it generates masses from 200 to 1000 keV in 100 keV steps.
The output files are `tmp/generators-dark_compton.yaml` and
`tmp/simconfig-dark_compton.yaml`.

### Assign detectors to simulated vertices

Map vertices in one or more LH5 files to LEGEND-200 HPGe detectors. Add `--save`
to write the detector field into the LH5 files, and `--counts-yaml` to save
aggregate counts:

```bash
pixi run assign-detectors \
  --gdml /path/to/l200.gdml \
  --lh5-file /path/to/input-1.lh5 /path/to/input-2.lh5 \
  --counts-yaml data/v1/dictionaries/dark-compton_primary-counts.yaml
```

### Migrate legacy YAML files

Rewrite legacy dictionaries containing NumPy/Python YAML tags using the safe
YAML format:

```bash
pixi run migrate-yaml data/v1/dictionaries/eres_per_det_tot.yaml --in-place
```

Migration is atomic and verifies that the rewritten file is readable by the safe
YAML loader.

### Check Dark Compton simulation statistics

```bash
pixi run check-dark-compton-stats
```

This is currently a site-specific diagnostic: it uses a hard-coded LEGEND
production reference path and reads Dark Compton products below `data/v1/`.

## Repository layout

- `src/bosonic_dm/` — core library and CLI entry points.
- `src/bosonic_dm/pipeline/` — simulation and background orchestration.
- `src/bosonic_dm/plotting/` — analysis plotting routines.
- `data/inputs/` — version-controlled input dictionaries.
- `notebooks/` — exploratory and legacy notebooks.
- `tests/` — automated tests.

The CLI is deliberately thin; analysis code is available from the Python API for
use in scripts and notebooks.

## License

See [LICENSE](LICENSE).
