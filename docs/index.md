# bosonic-DM documentation

This documentation covers installation, configuration, pipeline operation, and
the products created by the bosonic-DM analysis package.

## Start here

1. Follow [Getting started](getting-started.md) to create the Pixi environment,
   install the package, and run a first command.
2. Copy `configs/example-local.yaml` and use the
   [Configuration reference](configuration.md) to adapt it to your data.
3. Choose the [simulation](simulation-pipeline.md) or
   [background](background-pipeline.md) workflow.
4. Consult [Outputs and manifests](outputs.md) when inspecting or reusing
   generated products.

## Reference

- [Command-line reference](cli.md) lists the main pipeline and utility commands.
- [Development guide](development.md) explains the source layout and project
  checks.

## Scope and status

The package supports two simulated interactions, `axio-electric` and
`dark-compton`, plus a run-aware background-data workflow. The background plots
are diagnostic event-count comparisons. They are not exposure-normalized fit
inputs, and the current comparison-ratio uncertainties are not intended for
inference.

The analysis is under active development. Check each run's manifest and warnings
before consuming an output, and validate products on the relevant production
data.
