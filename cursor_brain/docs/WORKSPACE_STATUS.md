## Active Programs

- `workflow_runner.py`
  Core workflow heuristics, batch analysis, failure classification, and next-round planning.
- `automation_pipeline.py`
  Local CLI for snapshot, preflight, batch initialization, post-processing, and research-memory utilities.
- `run_batch_postprocess.py`
  Standardized batch post-processing and status generation.
- `prepare_batch_artifacts.py`
  Standard artifact layout creation helper.
- `expression_validator.py`
  Syntax, operator, field, and parameter validation.
- `expression_fingerprint.py`
  Structural fingerprinting, field extraction, and theme/operator summaries.
- `filter_wq_alphas.py`
  Threshold-based alpha summary and filtering utilities.
- `alpha_inventory_builder.py`
  Local inventory shaping and metadata maintenance.
- `field_catalog_builder.py`
  Field catalog normalization and CSV export helpers.

## Public Release Status

This repository is prepared for public sharing as a sanitized project snapshot.

Included:

- source code
- public-safe configuration templates
- sanitized workflow notes
- mock example inputs
- mock standardized batch artifacts

Removed or replaced:

- credentials
- private research memory logs
- real batch inventories
- real alpha ids and author ids
- real expressions and family names
- private batch-level notes

## Operational Rules Kept In The Public Version

1. Each round should have one primary objective.
2. Each batch should end with a winner, a failure diagnosis, and one next action.
3. Frozen directions should remain explicit in workflow state.
4. Temporary caches and generated artifacts should stay out of version control.

## Current Gaps

The public snapshot preserves architecture and file formats, but does not ship enough local assets to replay a private production workflow end to end.

## Legacy

- `legacy_direct_api/brain_toolkit.py`
  Archived direct-API reference path. Normal workflows should prefer the current local scripts and structured tooling.
