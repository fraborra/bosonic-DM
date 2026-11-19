# Command-line reference

Run all commands through Pixi from the package repository.

## Analysis pipeline

```text
pixi run bosonic-dm {simulation,background} --config FILE [OPTIONS]
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
configuration. Available stages are `build-dataset`, `efficiencies`, and
`plots`. Primaries are counted inside `efficiencies`; the former
`count-vertices` stage no longer exists.

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

Stages are `build-dataset`, `fit` and `fit-plots`; all run by default. `fit`
runs the Stan background fit per detector group and `fit-plots` draws its
diagnostics. Requesting a downstream stage adds its dependencies.

`--group GROUP [GROUP ...]` limits `fit` and `fit-plots` to the named detector
groups (default: every group in `analysis.detector_groups`). Quote names with
spaces. An unknown group is an error.

```bash
pixi run bosonic-dm background \
  --config configs/local.yaml \
  --stage fit fit-plots \
  --group "ICPC group1" BEGe
```

Outputs:

- `<data_root>/background_fits/<group_slug>/`: fit products.
- `<plots_root>/background_fit/<group_slug>/`: fit plots.

A changed `background.fit` setup or a rebuilt dataset triggers a refit;
`--overwrite` forces one.

### Batch jobs (NERSC)

```bash
sbatch scripts/run_sim.sh dark-compton efficiencies
sbatch scripts/run_bkg.sh "ICPC group1" fit fit-plots
sbatch scripts/run_bkg.sh all
```

`run_bkg.sh` takes a group name or `all` (no `--group` flag), then optional
stages. Submit from the repository root. It requests 4 CPUs for 4 hours.

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
  --counts-yaml tmp/vertex-counts.yaml
```

Use `--save` to write the detector field back to the LH5 input. `--vtx-group`
changes the vertex group from its `vtx` default, and `--verbose` enables debug
logging. `--counts-yaml` writes one total per detector for a quick check. The
pipeline does not read it: it counts primaries per run itself and keeps them in
the efficiency table.

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

### Check efficiency statistics

Rank detectors by the statistical uncertainty of their simulated efficiency, to
decide whether more events must be simulated:

```bash
pixi run check-efficiency-stats \
  --config configs/local.yaml \
  --interaction dark-compton
```

The command reads the efficiency table written by the `efficiencies` stage
(`<data_root>/dictionaries/<interaction>_efficiency.parquet`); it needs no
production metadata. For every simulated energy it lists the detectors with the
largest relative and absolute statistical uncertainty, counts those above the
target, and estimates how many more primaries the worst one needs.

- `--selection` (default `all`): efficiency selection to check.
- `--top` (default `5`): detectors listed per energy.
- `--target` (default `0.01`): target relative statistical uncertainty.

This command replaces `check-dark-compton-stats`. Reinstall the editable package
once to register it: `pixi run pip install --editable .`
