# Development guide

## Source layout

```text
src/bosonic_dm/
  cli.py                 `bosonic-dm` entry point
  config.py              YAML loading and validated configuration objects
  constants.py           physical constants and detector-type names
  models.py              result data structures
  pipeline/              simulation and background orchestration, manifests
  plotting/              plotting functions, including fit plots
  fitting/
    background_fit.py    Stan background fit, B_hat, prior predictive
    background_shapes.py NumPy mirrors of the background shapes
    model_comparison.py  Bayes factors (bridge sampling) and PSIS-LOO
    signal_template.py   group FWHM, efficiency, and Gaussian templates
    signal_fit.py        Stan signal fit and credible upper limit
    diagnostics.py       sampler checks shared by both fits
    stan_utils.py        Stan file location and provenance records
  background.py          background dataset and binned spectra
  cuts.py                quality cuts, filtering, usability, group exposure
  efficiency.py          efficiency calculation, table, and aggregation
  geometry.py            detector/vertex mapping (`assign-detectors`)
  io.py                  LH5, Awkward, and Parquet helpers
  resolution.py          detector energy resolution
  stats.py               statistical utilities
  utils.py               mass-point CLI helpers and Dark Compton kinematics
  yaml_io.py             safe YAML reading, writing, and `migrate-yaml`
  generators.py          `generate-gps-macros`
  generate_dark_compton_kin.py  `generate-dark-compton-kin`
  check_efficiency_stats.py     `check-efficiency-stats`
```

Core modules do not import from `plotting/`; shared names such as the
detector-type map live in `constants.py`.

The Stan models live in `stan/` at the repository root. The fitting modules
locate them relative to the source tree (`fitting/stan_utils.py`), so use an
editable installation. CmdStanPy compiles a model on first use.

The CLI should remain a thin wrapper. Put reusable analysis logic in the package
so it can also be called from scripts, tests, and notebooks.

## Environment and tests

Install the editable package once after creating the environment:

```bash
pixi install
pixi run pip install --editable .
```

Run the complete test suite in the Pixi test environment:

```bash
pixi run test
```

Run Python commands through Pixi and use `python3`, for example:

```bash
pixi run python3 -m bosonic_dm.cli --help
```

Formatting and lint rules are configured in `pyproject.toml` and
`.pre-commit-config.yaml`. Consult `AGENTS.md` for repository-specific coding,
testing, statistics, and Git conventions before contributing.

## Documentation conventions

- Keep `README.md` focused on project identity, installation, a minimal run, and
  links into `docs/`.
- Put task-oriented walkthroughs in dedicated pages and reference material in
  `configuration.md`, `cli.md`, or `outputs.md`.
- Verify command names and defaults against `pyproject.toml` and the relevant
  `argparse` definition.
- Use paths such as `<data_root>` when a location is configurable.
- State validation limitations next to the affected product, especially for
  analysis plots and uncertainties.
- Link every new page from `docs/index.md` and, when it is a primary entry
  point, from the README.
