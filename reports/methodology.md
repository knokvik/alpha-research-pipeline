# Methodology Notes

This project treats statistical rigor as the primary deliverable.

## Validation

The default validation path is rolling walk-forward. Each fold trains only on prior dates, tests on a forward block, purges training labels that overlap the test period, and records fold metadata to `folds.json`.

## Feature Timing

All generated factors are shifted by at least one asset-level observation before they are merged with forward-return labels. Feature rows carry an `as_of_date`, and the assembly step rejects any feature whose `as_of_date` is after its market date.

## Multiple Testing

Every model or parameter setting is a strategy variant. The trial ledger persists the variant name, parameters, and metrics. Deflated Sharpe is computed against the distribution of tested variant Sharpes, not against the winning backtest alone.

## Data Limitations

The built-in demo uses synthetic data so the pipeline is fully reproducible. Real research must replace it with survivorship-bias-free, point-in-time data before making claims about live equity alpha.
