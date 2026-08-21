# Background pipeline

The background workflow reads PET-tier LH5 files one at a time and creates a
run-aware Parquet dataset for fast selection studies.

```bash
pixi run bosonic-dm background --config configs/local.yaml
```

## Input discovery

Set `background.pet_glob` to an explicit file pattern when possible:

```yaml
background:
  pet_glob: /path/to/pet/phy/*.lh5
```

If the value is empty or omitted, the workflow reads
`<reference_root>/<version>/config.json` and derives the physics PET-tier path.

## Dataset layout

The output is Hive-partitioned by period and run:

```text
<data_root>/parquet/background/
  period=pXX/
    run=rXXX/
      <PET-source-name>.parquet
```

Each run is mapped with the channel map valid at its physics start key. Only
multiplicity-one events are flattened permanently. Other selections remain
boolean columns so they can be varied from a notebook without rereading LH5:

- `passes_baseline`: good channel, no pulser or forced trigger, no offline muon,
  and an HPGe coincidence.
- `passes_default`: baseline plus the standard BB-like selection.
- `passes_without_bb_like`: baseline without BB-like, retaining the legacy
  delayed-discharge veto.
- `passes_lar`: no SiPM coincidence.
- `passes_analysis`: default selection combined with the configured LAr-veto
  behavior.
- `passes_comparison`: configured comparison profile combined with the same
  LAr-veto behavior.

## Reading the dataset

Polars can inspect all partitions lazily:

```python
from pathlib import Path

import polars as pl

dataset_root = Path("data/v1/parquet/background")
background = pl.scan_parquet(dataset_root, hive_partitioning=True)

selected = background.filter(pl.col("passes_analysis"))
cutflow = background.select(
    pl.len().alias("multiplicity_one"),
    pl.col("passes_baseline").sum().alias("baseline"),
    pl.col("passes_default").sum().alias("default"),
    pl.col("passes_lar").sum().alias("lar"),
    pl.col("passes_analysis").sum().alias("analysis"),
).collect()
```

## Diagnostic plots

When `output.save_plots` is true, spectrum overlays for every configured energy
range and bin width, plus a partition summary, are written below:

```text
<plots_root>/background/
```

These plots compare event counts. They are not exposure-normalized fit inputs.
The comparison sample is correlated with the default sample, so the current
ratio error bars must not be interpreted as inference-ready uncertainties.

## Reuse and rebuilding

The manifest stores the cut setup, selected PET files, written and reused
partitions, plot setup, and warnings. A cached dataset is reused when Parquet
fragments exist and the recorded cut setup matches. A changed setup or
`--overwrite` forces a rebuild. Plot compatibility is tracked separately.

The manifest is written to `<data_root>/background_manifest.yaml` when
`output.write_manifest` is enabled.
