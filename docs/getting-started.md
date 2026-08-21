# Getting started

## Prerequisites

- Python 3.11 or newer.
- [Pixi](https://pixi.sh/) for dependency and environment management.
- Access to the LEGEND production metadata and the simulation or PET-tier files
  needed by the workflow you intend to run.

Run commands from the package repository, the directory containing
`pyproject.toml`.

## Install the environment

```bash
pixi install
pixi run pip install --editable .
```

The editable installation exposes the `bosonic-dm` command and the utility
commands declared by the package. Keep the `pixi run` prefix so commands use the
project environment.

To check the installation:

```bash
pixi run bosonic-dm --help
pixi run test
```

## Create a local configuration

```bash
cp configs/example-local.yaml configs/local.yaml
```

At minimum, review these paths in `configs/local.yaml`:

- `production.reference_root`: root of the LEGEND production reference.
- `production.version`: production version below that root.
- `paths.simulation_root`: root containing generated simulation tiers.
- `paths.data_root`: destination for Parquet datasets, dictionaries, and
  manifests.
- `paths.inputs_root`: static calibration and detector-group inputs.
- `background.pet_glob`: optional explicit glob for background PET files.

See the [configuration reference](configuration.md) for every setting and path
resolution rules.

## Run a first workflow

Run one simulated interaction:

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction dark-compton
```

Or build the background dataset:

```bash
pixi run bosonic-dm background --config configs/local.yaml
```

The commands log missing inputs as warnings where partial execution is safe.
After a run, inspect the manifest under `paths.data_root`; it records the stage
status, outputs, resolved stages, and warnings.

## Common first-run problems

### The production root does not exist

Configuration loading emits a warning, but it does not stop immediately. Stages
that require production metadata will remain blocked until
`production.reference_root/production.version` is available.

### No background PET files are found

Set `background.pet_glob` explicitly. If it is empty, the pipeline tries to
derive the PET-tier path from the production `config.json`.

### A simulation energy is skipped

Check that the configured `job_template` expands to the actual simulation job
name and that the corresponding CVT and STP files exist. Missing inputs are
reported per energy in the simulation manifest.

### An old product is reused

Use `--overwrite` when you intentionally need to rebuild products. See
[Outputs and manifests](outputs.md) for the current cache behavior.
