# Development guide

## Source layout

```text
src/bosonic_dm/
  cli.py                 command-line entry point
  config.py              YAML loading and validated configuration objects
  models.py              result data structures
  pipeline/              simulation and background orchestration
  plotting/              plotting functions, including background-fit plots
  fitting/               Stan background and signal fits, diagnostics, shapes
  background.py          background dataset and binned spectra
  cuts.py                quality cuts, filtering, and detector usability
  efficiency.py          efficiency calculations
  geometry.py            detector/vertex mapping
  io.py                  LH5, Awkward, and Parquet helpers
  resolution.py          detector energy resolution
  stats.py               statistical utilities
  yaml_io.py             safe YAML reading, writing, and migration
```

The Stan models live in `stan/` at the repository root. The fitting modules
locate them relative to the source tree, so use an editable installation.
CmdStanPy compiles a model on first use.

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
