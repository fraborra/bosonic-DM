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
