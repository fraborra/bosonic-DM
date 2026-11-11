# Outputs and manifests

All output roots come from the analysis configuration. With the example values,
the main tree is:

```text
data/v1/
  parquet/
    axio-electric/
      sim_e=<energy>/data.parquet
    dark-compton/
      sim_e=<energy>/data.parquet
    background/
      period=<period>/run=<run>/<source>.parquet
  dictionaries/
    <interaction>_efficiency.parquet
  <interaction>_manifest.yaml
  background_manifest.yaml

plots/
  <interaction>_*.png
  <interaction>/...
  background/...
```

Only products for workflows and energies that have valid inputs appear. The
efficiency product is a Parquet table with one row per energy, detector, run,
and selection; its columns are listed in
[Simulation pipeline](simulation-pipeline.md#efficiencies). It also holds the
primary counts (`n_primaries`); there is no separate primary-counts file.

Fit products are not part of this tree. `save_background_fit` and
`save_signal_fit` write to the directory passed by the caller and record their
inputs in a `summary.yaml`; see [Spectral fits](fitting.md#products).

## Simulation manifests

`<data_root>/<interaction>_manifest.yaml` is cumulative. Its stage records hold
each stage's status, outputs, and the cut and plot settings used to decide
whether efficiency and plot products can be reused. `last_run` records:

- requested and dependency-resolved stages;
- whether overwrite was enabled;
- non-fatal warnings.

## Background manifest

`<data_root>/background_manifest.yaml` records:

- the resolved PET glob and discovered input files;
- written and reused Parquet partitions;
- cut and plot setup snapshots;
- dataset and sanity-plot stage status;
- warnings from data processing and plotting.

## Stage statuses

| Status      | Meaning                                                              |
| ----------- | -------------------------------------------------------------------- |
| `completed` | The stage produced its expected current outputs.                     |
| `cached`    | Compatible existing outputs were reused.                             |
| `partial`   | Some configured inputs or energy points were unavailable.            |
| `blocked`   | Required inputs were unavailable and no current result was produced. |
| `disabled`  | The configuration intentionally disabled the output, such as plots.  |

Check both the stage status and the warnings. A file can exist and still not
match the current configuration.

## Overwrite behavior

The effective overwrite value comes from `output.overwrite`, unless the command
line includes `--overwrite`. Rebuild after changing source data or any setting
that a cached stage does not record as compatible.

To force a rebuild, use `--overwrite` rather than deleting individual files from
a partitioned dataset, then check the manifest and reported output paths.
