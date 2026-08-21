# Command-line reference

All commands should be run through Pixi from the package repository.

## Analysis pipeline

```text
pixi run bosonic-dm {simulation,background,all} --config FILE [OPTIONS]
```

Common options:

| Option                      | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| `--config FILE`             | Required YAML configuration.                                     |
| `--overwrite`               | Rebuild existing products instead of reusing compatible outputs. |
| `--stage STAGE [STAGE ...]` | Run selected stages. Dependencies are added automatically.       |

### Simulation

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction dark-compton
```

`--interaction` accepts a key declared under `interactions` in the
configuration. Available stages are `count-vertices`, `build-dataset`,
`efficiencies`, and `plots`.

Requesting a downstream stage also runs its dependencies:

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction axio-electric \
  --stage plots
```

### Background

```bash
pixi run bosonic-dm background --config configs/local.yaml
```

The background workflow currently exposes one stage, `build-dataset`. Diagnostic
plots run automatically when `output.save_plots` is enabled.

### All analyses

By default, `all` runs `axio-electric`, `dark-compton`, and then the background
workflow:

```bash
pixi run bosonic-dm all --config configs/local.yaml
```

Limit the simulation interactions with `--interactions`:

```bash
pixi run bosonic-dm all \
  --config configs/local.yaml \
  --interactions dark-compton
```

When combining `--stage` with `all`, the stage name is passed to both pipeline
types. At present, `build-dataset` is the only stage name shared by simulation
and background; use the individual subcommands for any other staged run.

## Simulation-input utilities

### Generate GPS macro YAML

```bash
pixi run generate-gps-macros -t axio_electric -e 100 200 300
pixi run generate-gps-macros -t both --mass-range 100 1020 10
```

- `-e, --energies`: explicit mass points in keV.
- `--mass-range START STOP STEP`: inclusive range in keV.
- `-t, --type`: `dark_compton`, `axio_electric`, or `both` (default).

If no mass selection is supplied, the command uses 200–1000 keV in 100 keV
steps. It writes `generators-*.yaml` and `simconfig-*.yaml` under `tmp/`. The
generated Dark Compton generator definitions currently contain a site-specific
kinematics path, so review them before submitting a simulation.

### Generate Dark Compton kinematics

```bash
pixi run generate-dark-compton-kin -e 200 --events 10000000
pixi run generate-dark-compton-kin --mass-range 100 1020 10
```

In addition to the common energy options, this command accepts:

- `--events` (default `10000000`).
- `--outdir` (default `tmp/`).
- `--seed` (default `0`, incremented for each mass point).
- `--chunk-size` (default `500000`).

It creates one `dark_compton_<mass>keV.lh5` FromFile input per mass point.

### Assign detectors to vertices

```bash
pixi run assign-detectors \
  --gdml /path/to/l200.gdml \
  --lh5-file /path/to/input-1.lh5 /path/to/input-2.lh5 \
  --counts-yaml data/v1/dictionaries/dark-compton_primary-counts.yaml
```

Use `--save` to write the detector field back to the LH5 input. `--vtx-group`
changes the vertex group from its `vtx` default, and `--verbose` enables debug
logging.

## Data maintenance utilities

### Migrate legacy YAML

Rewrite NumPy/Python-tagged dictionaries into safe YAML in place:

```bash
pixi run migrate-yaml data/v1/dictionaries/old.yaml --in-place
```

Or write one or more migrated files elsewhere:

```bash
pixi run migrate-yaml one.yaml two.yaml \
  --output-dir migrated \
  --overwrite
```

The migration writes atomically and verifies that the result can be read by the
safe YAML loader.

### Check Dark Compton statistics

```bash
pixi run check-dark-compton-stats
```

This is a site-specific diagnostic. It currently assumes a hard-coded LEGEND
production path and reads Dark Compton products below `data/v1/`.
