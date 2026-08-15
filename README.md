# bosonic-DM

Analysis tools for bosonic dark-matter searches with the LEGEND-200 experiment.
The package provides reusable Python functions and command-line pipelines for
simulation processing, efficiency estimation, plots, and background studies.

> **Status:** under active development. The run-aware background dataset and
> diagnostic plots are implemented, but the products must still be validated on
> production data before they are used for inference.

## Requirements and installation

The project requires Python 3.11 or newer and uses [Pixi](https://pixi.sh/) to
manage its environment. From the repository root:

```bash
pixi install
pixi run pip install --editable .
```

Use `pixi run` for all project commands. To run the test suite in the Pixi test
environment:

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

### Background dataset workflow

The background command reads PET-tier LH5 files one at a time and writes a
run-aware, Hive-partitioned Parquet dataset:

```text
<data_root>/parquet/background/
  period=pXX/
    run=rXXX/
      <PET-source-name>.parquet
```

Each run is mapped with the channel map valid at its physics start key. Only
multiplicity-one events are flattened permanently; the remaining selections are
stored as boolean columns so they can be changed or compared from a notebook
without rereading LH5 data:

- `passes_baseline`: good channel, no pulser or forced trigger, no offline muon,
  and an HPGe coincidence.
- `passes_default`: baseline plus the standard BB-like selection.
- `passes_without_bb_like`: baseline without the BB-like requirement, while
  retaining the delayed-discharge veto used by the legacy background notebook.
- `passes_lar`: no SiPM coincidence.
- `passes_analysis`: default selection combined with the configured LAr-veto
  behavior.
- `passes_comparison`: the configured comparison profile combined with the same
  LAr-veto behavior.

For example, the dataset can be inspected lazily with Polars:

```python
from pathlib import Path

import polars as pl

dataset_root = Path("data/v1/parquet/background")
background = pl.scan_parquet(dataset_root, hive_partitioning=True)

selected = background.filter(pl.col("passes_analysis"))
cutflow = background.select(
    pl.len().alias("multiplicity_one"),
    pl.col("passes_baseline").sum().alias("baseline"),
    pl.col("passes_default").sum().alias("default"),
    pl.col("passes_lar").sum().alias("lar"),
    pl.col("passes_analysis").sum().alias("analysis"),
).collect()
```

When `output.save_plots` is enabled, the pipeline also writes spectrum overlays
and a partition summary below `<plots_root>/background/`. These are diagnostic
checks based on event counts; they are not exposure-normalized fit inputs. The
comparison sample is correlated with the default sample, so its current ratio
error bars must not be interpreted as an inference-ready uncertainty.

The run also writes `<data_root>/background_manifest.yaml`, listing discovered
PET files, written or reused partitions, plots, warnings, and stage status.
Existing partitions are reused based on their presence. Until cache
fingerprinting is implemented, use `--overwrite` whenever the PET production,
metadata, cut definitions, LAr setting, or comparison profile changes.

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

### Configuration

You can specify the energies for the simulation in your YAML configuration file
(e.g., `configs/nersc.yaml`) under `analysis.simulated_energies_keV`. You can
provide an exhaustive list, or to keep it clean, use a range dictionary with
`start`, `stop`, and `step`:

```yaml
analysis:
  simulated_energies_keV:
    start: 100
    stop: 1020
    step: 10
```

### Migrate legacy YAML files

Rewrite legacy dictionaries containing NumPy/Python YAML tags using the safe
YAML format:

```bash
pixi run migrate-yaml data/v1/dictionaries/eres_per_det_tot.yaml --in-place
```

Migration is atomic and verifies that the rewritten file is readable by the safe
YAML loader.

### Generate GPS Macros

Generates the `generators-*.yaml` and `simconfig-*.yaml` FromFile macros inside
the `tmp/` directory for either dark compton, axio-electric, or both
interactions.

**Command:**

```bash
pixi run generate-gps-macros [OPTIONS]
```

**Options:**

- `-e, --energies`: Explicit list of $m_{DM}$ values in keV (e.g.
  `-e 200 300 400`).
- `--mass-range START STOP STEP`: Inclusive mass range in keV (e.g.
  `--mass-range 100 1020 10`).
- `-t, --type`: Type of macros to generate. Choices: `dark_compton`,
  `axio_electric`, `both` (default is `both`).

If neither `-e` nor `--mass-range` is given, generates all default mass points
(200–1000 keV in 100 keV steps).

**Example:**

```bash
pixi run generate-gps-macros -t axio_electric -e 100 200 300
pixi run generate-gps-macros --mass-range 100 1020 10
```

### Generate Dark Compton Kinematic Files

Produces LH5 kinematic input files for the remage `FromFile` generator. Each
event contains a correlated electron–photon pair with Dark Compton kinematics.

**Command:**

```bash
pixi run generate-dark-compton-kin [OPTIONS]
```

**Options:**

- `-e, --energies`: Explicit list of $m_{DM}$ values in keV (e.g.
  `-e 200 300 400`).
- `--mass-range START STOP STEP`: Inclusive mass range in keV (e.g.
  `--mass-range 100 1020 10`).
- `--events`: Number of events per file (default: 10 000 000).
- `--outdir`: Output directory (default: `tmp/`).
- `--seed`: Base random seed (default: 0).
- `--chunk-size`: Events per LH5 write chunk (default: 500 000).

If neither `-e` nor `--mass-range` is given, generates all default mass points
(200–1000 keV in 100 keV steps).

**Examples:**

```bash
# Generate a single mass point
pixi run generate-dark-compton-kin -e 200 --events 10000000

# Generate a custom range
pixi run generate-dark-compton-kin --mass-range 100 1020 10

# Generate all default mass points
pixi run generate-dark-compton-kin
```

### Check Dark Compton Stats

Calculates the statistical uncertainties of the detection efficiencies for
individual detectors to estimate how many additional events need to be generated
to satisfy relative statistical uncertainty requirements.

**Command:**

```bash
pixi run check-dark-compton-stats
```

This is currently a site-specific diagnostic: it uses a hard-coded LEGEND
production reference path and reads Dark Compton products below `data/v1/`.

## Repository layout

- `src/bosonic_dm/`: Core library code.
  - `cli.py`: Command-line entry point for `simulation`, `background`, and
    `all`.
  - `config.py`, `models.py`, `yaml_io.py`: Configuration, result models, and
    YAML I/O.
  - `pipeline/`: High-level orchestration for simulation and background
    analyses.
  - `generators.py`: GPS macro generation.
  - `generate_dark_compton_kin.py`: LH5 kinematic file generation for Dark
    Compton FromFile remage generator.
  - `cuts.py`: Quality cuts and filtering logic.
  - `efficiency.py`: Efficiency computations.
  - `io.py`: Parquet and awkward data manipulation.
  - `resolution.py`: Energy resolution (FWHM) extraction.
  - `stats.py`: Statistical utilities.
  - `utils.py`: General utilities.
  - `plotting/`: Subpackage for plotting logic (AoE, spectra, resolution).
- `notebooks/`: Exploratory Jupyter notebooks.
- `data/v1/`: Parquet datasets and dictionaries.
- `tests/`: Automated tests.
- `tmp/`: Generated temporary files and macros.

The CLI is deliberately thin; analysis code is available from the Python API for
use in scripts and notebooks.

## License

See [LICENSE](LICENSE).
