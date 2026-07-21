# bosonic-DM

Python package for bosonic DM analysis for the LEGEND-200 experiment.

## Installation

This project is managed using [Pixi](https://pixi.sh/).

To install the environment and the package in editable mode, run:

```bash
pixi run pip install -e .
```

## CLI Usage Guide

The package provides command-line interfaces for specific tasks without needing
to run Jupyter notebooks.

### Run the simulation pipeline

The simulation command resolves stage dependencies automatically. Missing energy
inputs are recorded as skipped or partial work in the manifest instead of
producing efficiencies with missing denominators.

```bash
pixi run bosonic-dm simulation \
  --config configs/nersc.yaml \
  --interaction dark-compton
```

To run both simulation interactions, pass them to the `all` command:

```bash
pixi run bosonic-dm all \
  --config configs/nersc.yaml \
  --interactions axio-electric dark-compton
```

`simulation` currently runs one interaction at a time with `--interaction`,
while `all` accepts one or more interactions with `--interactions` and also runs
the background analysis. The planned CLI improvement is to make `simulation`
accept multiple interactions as well.

Run only the background pipeline with:

```bash
pixi run bosonic-dm background --config configs/nersc.yaml
```

Use `--stage` to select specific pipeline stages and `--overwrite` to replace
existing products. For example:

```bash
pixi run bosonic-dm simulation \
  --config configs/nersc.yaml \
  --interaction dark-compton \
  --stage plots \
  --overwrite
```

Calibration-derived resolution dictionaries are configured separately through
`paths.calibration_dictionaries_root`; simulation products are written below
`paths.data_root`.

### Migrate legacy YAML files

Rewrite legacy dictionaries containing NumPy/Python YAML tags:

```bash
pixi run migrate-yaml \
  data/v1/dictionaries/eres_per_det_tot.yaml \
  data/v1/dictionaries/rawid_by_det_type.yaml \
  --in-place
```

The migration is atomic and verifies that the rewritten file can be read back
using the safe YAML loader.

### Generate Dark Compton Macros

Generates the `generators-dark_compton.yaml` and `simconfig-dark_compton.yaml`
GPS macros inside the `tmp/` directory.

**Command:**

```bash
pixi run generate-dark-compton [OPTIONS]
```

**Options:**

- `-e, --energies`: List of dark matter mass ($m_{DM}$) values in keV to
  simulate. If not provided, it defaults to
  `200 300 400 500 600 700 800 900 1000`.

**Example:**

```bash
pixi run generate-dark-compton -e 200 300 500
```

### Check Dark Compton Stats

Calculates the statistical uncertainties of the detection efficiencies for
individual detectors to estimate how many additional events need to be generated
to satisfy relative statistical uncertainty requirements.

**Command:**

```bash
pixi run check-dark-compton-stats
```

## Project Structure

- `src/bosonic_dm/`: Core library code.
  - `cli.py`: Command-line entry point for `simulation`, `background`, and
    `all`.
  - `config.py`, `models.py`, `yaml_io.py`: Configuration, result models, and
    YAML I/O.
  - `pipeline/`: High-level orchestration for simulation and background
    analyses.
  - `cuts.py`: Quality cuts and filtering logic.
  - `efficiency.py`: Efficiency computations.
  - `io.py`: Parquet and awkward data manipulation.
  - `resolution.py`: Energy resolution (FWHM) extraction.
  - `stats.py`: Statistical utilities.
  - `utils.py`: General utilities.
  - `plotting/`: Subpackage for plotting logic (AoE, spectra, resolution).
- `notebooks/`: Exploratory Jupyter notebooks.
- `data/v1/`: Parquet datasets and dictionaries.
- `tmp/`: Generated temporary files and macros.

The intended flow is:

```text
CLI (cli.py) → configuration (config.py) → pipeline/
                                      ├── simulation.py → tables, efficiencies, plots
                                      └── background.py → spectra, summaries, plots
```

The CLI is deliberately thin: scientific logic belongs in the Python API and the
pipeline modules, so the same analysis can be called from scripts or notebooks.
