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
    <interaction>_primary-counts.yaml
    <interaction>_efficiency.yaml
  <interaction>_manifest.yaml
  background_manifest.yaml

plots/
  <interaction>_*.png
  <interaction>/...
  background/...
```

Only products for workflows and energies that have valid inputs will appear.

## Simulation manifests

`<data_root>/<interaction>_manifest.yaml` is cumulative. Its stage records track
outputs and compatibility metadata, while `last_run` records:

- requested and dependency-resolved stages;
- whether overwrite was enabled;
- non-fatal warnings.

The manifest's stage records capture each status plus the cut and plot settings
needed to decide whether efficiency and plot products can be reused.

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

Downstream users should check both the stage status and warnings. The existence
of a file alone does not guarantee that it matches the current configuration.

## Overwrite behavior

The effective overwrite value comes from `output.overwrite`, unless the command
line includes `--overwrite`. Rebuilding is appropriate after changing source
data or any setting that is not represented by a compatible cached stage.

Avoid deleting individual files from a partitioned dataset as a cache-control
mechanism. Prefer `--overwrite`, then verify the manifest and reported output
paths after the run.
