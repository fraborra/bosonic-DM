# bosonic-DM documentation

## Start here

1. Follow [Getting started](getting-started.md) to create the Pixi environment,
   install the package, and run a first command.
2. Copy `configs/example-local.yaml` and use the
   [Configuration reference](configuration.md) to adapt it to your data.
3. Choose the [simulation](simulation-pipeline.md) or
   [background](background-pipeline.md) workflow.
4. Consult [Outputs and manifests](outputs.md) when inspecting or reusing
   generated products.
5. Use [Spectral fits](fitting.md) to fit the background spectra and set signal
   limits from Python.

## Reference

- [Command-line reference](cli.md) lists the main pipeline and utility commands.
- [Development guide](development.md) explains the source layout and project
  checks.
- [`stan/README.md`](../stan/README.md) documents the background Stan model.

## Scope and status

The package supports two simulated interactions, `axio-electric` and
`dark-compton`, plus a run-aware background-data workflow. The background plots
are diagnostic event-count comparisons. They are not exposure-normalized fit
inputs, and the current comparison-ratio uncertainties are not intended for
inference.

A Stan background fit and a first-stage signal fit are available from the Python
API. The background fit is under validation. The signal fit keeps the background
and signal response fixed, so its limits do not yet include those uncertainties.

The analysis is under active development. Check each run's manifest and warnings
before consuming an output, and validate products on the relevant production
data.
