# bosonic-DM

Analysis tools for bosonic dark-matter searches with the LEGEND-200 experiment.
The package provides reusable Python functions and command-line pipelines for
simulation processing, efficiency estimation, diagnostic plots, and background
studies.

> **Status:** under active development. The run-aware background dataset and
> diagnostic plots must be validated on production data before they are used for
> inference.

## Quick start

The project requires Python 3.11 or newer and uses [Pixi](https://pixi.sh/) to
manage its environment.

```bash
pixi install
pixi run pip install --editable .
cp configs/example-local.yaml configs/local.yaml
pixi run test
```

Edit `configs/local.yaml` for your production and simulation paths, then run one
of the analysis pipelines:

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction dark-compton

pixi run bosonic-dm background --config configs/local.yaml
```

Use `pixi run` for every project command. Site-specific data paths and
simulation inputs are intentionally not committed.

## Documentation

- [Documentation index](docs/index.md)
- [Getting started](docs/getting-started.md)
- [Configuration reference](docs/configuration.md)
- [Command-line reference](docs/cli.md)
- [Simulation pipeline](docs/simulation-pipeline.md)
- [Background pipeline](docs/background-pipeline.md)
- [Outputs and manifests](docs/outputs.md)
- [Development guide](docs/development.md)

## Repository layout

- `src/bosonic_dm/` — library and pipeline implementation.
- `configs/` — default and local configuration examples.
- `docs/` — user and contributor documentation.
- `notebooks/` — exploratory and legacy analysis notebooks.
- `tests/` — automated tests.

The CLI is deliberately thin; analysis functions are also available from the
Python API for use in scripts and notebooks.

## License

See [LICENSE](LICENSE).
