# Simulation pipeline

The simulation workflow converts per-energy simulation products into Parquet
datasets, efficiency dictionaries, and summary plots.

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction dark-compton
```

## Input discovery

For each configured energy, the interaction's `job_template` is expanded and the
pipeline searches for:

```text
<simulation_root>/generated/tier/cvt/
  l200cfg01-<job-name>-tier_cvt.lh5

<simulation_root>/generated/tier/stp/<job-name>/
  l200cfg01-<job-name>-job_*-tier_stp.lh5
```

The geometry used for vertex counting is taken from the first configured energy:

```text
<simulation_root>/generated/pars/geom/
  l200cfg01-<first-job-name>-tier_stp-geom.gdml
```

Efficiency calculation additionally needs:

- `<inputs_root>/dictionaries/eres_per_det_tot.yaml`.
- `<reference_root>/<version>/config.json`.
- The detector grouping configured by `analysis.detector_groups` for grouped
  plots.

Some advanced plots also use
`<inputs_root>/dictionaries/rawid_by_det_type.yaml`.

## Stages

The execution order is:

```text
count-vertices ─┐
                ├─> efficiencies ─> plots
build-dataset ──┘
```

### `count-vertices`

Maps STP vertices to detectors using the GDML geometry, aggregates the primary
counts, and writes:

```text
<data_root>/dictionaries/<interaction>_primary-counts.yaml
```

### `build-dataset`

Builds one Parquet partition per available CVT energy:

```text
<data_root>/parquet/<interaction>/
  sim_e=<energy>/
    data.parquet
```

### `efficiencies`

Combines primary counts, Parquet events, channel metadata, energy resolution,
the configured FEP window, selections, and LAr-veto setting. The result is:

```text
<data_root>/dictionaries/<interaction>_efficiency.yaml
```

Missing energy inputs are skipped and recorded; the pipeline does not invent a
denominator for incomplete energy points. Run-level and aggregated selections
include usable exposure, nominal effective exposure, and propagated statistical
and FWHM-window uncertainty fields. Invalid categories retain run-level physical
exposure for diagnosis but have no aggregated usable or effective exposure.

### `plots`

Creates efficiency, effective-exposure, and FEP-survival summaries grouped by
detector type and detector group. Interaction flags can enable energy spectra,
LAr-survival, and AoE-survival plots.

## Running selected stages

`--stage` accepts one or more stage names. Dependencies are resolved
automatically, so this command requests the complete chain needed for plots:

```bash
pixi run bosonic-dm simulation \
  --config configs/local.yaml \
  --interaction dark-compton \
  --stage plots
```

Use `--overwrite` to rebuild existing products. Without it, existing energy
partitions can be reused, and efficiency or plot products are reused only when
their manifest records match the relevant cut and plot setup.

## Partial runs

The workflow is designed to preserve useful results when only some energies are
available. A stage may be `completed`, `cached`, `partial`, `blocked`, or
`disabled`. Inspect `<data_root>/<interaction>_manifest.yaml` for the exact
status and warnings from the last run.
