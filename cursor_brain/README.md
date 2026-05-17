# Alpha Mining Automation System

Public project snapshot of a WorldQuant BRAIN alpha-mining workflow focused on:

- expression validation
- batch preflight and post-processing
- robustness-oriented alpha filtering
- local inventory management
- reusable workflow configuration
- research-memory style iteration support

This repository is intentionally cleaned for GitHub. Bulk platform outputs, credentials, caches, and most one-off round artifacts were removed so the remaining content emphasizes system design instead of raw workspace volume.
Configuration values, research notes, and example artifacts have also been sanitized for public release. The examples are mock public-safe records that preserve file shape without exposing private research assets.

## Project Background

The system was built around the practical constraints of WorldQuant-style alpha research rather than generic machine learning experimentation. The main goal is to shorten the loop from:

`field research -> expression drafting -> local validation -> batch simulation -> robustness diagnosis -> next-round action selection`

The workflow is designed to reason explicitly about common failure modes such as:

- `LOW_SUB_UNIVERSE_SHARPE`
- `LOW_2Y_SHARPE` / `IS_LADDER_SHARPE`
- `PROD_CORRELATION`
- `SELF_CORRELATION`
- concentration and turnover issues

## What The Repository Contains

### Core scripts

- `workflow_runner.py`: workflow heuristics, diagnostics, candidate evaluation, batch payload construction
- `automation_pipeline.py`: CLI entrypoint for snapshot, preflight, batch initialization, batch processing, and research-memory utilities
- `expression_validator.py`: expression parser plus operator / field compatibility checks
- `expression_fingerprint.py`: expression structure fingerprinting and theme/operator extraction
- `filter_wq_alphas.py`: threshold-based filtering and compact alpha summaries
- `prepare_batch_artifacts.py`: standard artifact directory creation helpers
- `run_batch_postprocess.py`: batch post-processing and artifact/status generation
- `alpha_inventory_builder.py`: local alpha inventory builder
- `field_catalog_builder.py`: field catalog CSV builder

### Supporting directories

- `configs/`: reusable workflow configuration examples
- `docs/`: workflow notes, integrated process documentation, and research assets
- `examples/`: curated mock inputs and one standardized mock batch artifact example
- `legacy_direct_api/`: older direct-API utilities kept for reference only

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- workflow_runner.py
|-- automation_pipeline.py
|-- expression_validator.py
|-- expression_fingerprint.py
|-- filter_wq_alphas.py
|-- prepare_batch_artifacts.py
|-- run_batch_postprocess.py
|-- alpha_inventory_builder.py
|-- field_catalog_builder.py
|-- operators.json
|-- configs/
|-- docs/
|-- examples/
`-- legacy_direct_api/
```

## Example Workflow

### 1. Generate a workflow snapshot

```bash
python automation_pipeline.py snapshot --config configs/eur_analyst_workflow_config.json
```

### 2. Run local preflight on candidate expressions

```bash
python automation_pipeline.py preflight --config configs/eur_analyst_workflow_config.json --expressions your_candidates.json --report-out preflight_report.json
```

### 3. Initialize a standard batch artifact directory

```bash
python automation_pipeline.py init-batch --config configs/eur_analyst_workflow_config.json --artifact-root batch_artifacts --round-name demo_round --multisim-id YOUR_MULTISIM_ID
```

### 4. Post-process a collected batch

```bash
python automation_pipeline.py process-batch --config configs/eur_analyst_workflow_config.json --batch-dir batch_artifacts/demo_round__YOUR_MULTISIM_ID
```

## Included Examples

- `examples/inputs/`: representative mock candidate/preflight input files
- `examples/artifacts/public_batch_demo__MULTISIM_PLACEHOLDER/`: one standardized mock batch directory showing:
  - `multisim_children.json`
  - `alpha_details/`
  - `batch_payload.json`
  - `postprocess_status.json`
  - `manifest.json`

These examples are kept to show the expected artifact layout without publishing the full private research workspace.

## Dependencies

External Python dependencies are intentionally minimal:

- `pandas`
- `requests`

Install with:

```bash
pip install -r requirements.txt
```

## Notes On Public Cleanup

The public-facing cleanup excludes:

- local credentials
- generated inventories
- field catalog dumps
- large batch artifact collections
- optimization logs
- cache files
- most one-off research round JSON files

Ignored paths are defined in `.gitignore`.

## Limitations

- Some legacy scripts still assume a local `credential.txt` when using direct API flows.
- The repository snapshot is designed for code review and project presentation, not as a drop-in runnable production environment.
- Example data is illustrative, sanitized, and intentionally incomplete.
"# alpha-research-workflow"  
