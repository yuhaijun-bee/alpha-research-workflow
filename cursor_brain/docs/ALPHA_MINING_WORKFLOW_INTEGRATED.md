# Alpha Mining Workflow Integrated

This document is a public-safe overview of the workflow embodied in this repository. It intentionally avoids real alpha identifiers, real production expressions, and private research conclusions.

## Workflow Goal

The system is designed to shorten the loop from:

`research idea -> candidate expressions -> local validation -> platform batch results -> diagnosis -> next-round action`

The key design decision is to optimize for workflow quality, not just candidate count. A good round ends with a clear winner, a clear failure diagnosis, and one constrained next action.

## Standard Round Structure

### 1. Freeze the round scope

Before generating candidates, fix:

- region
- universe
- delay
- target data family
- one primary objective for the round
- any frozen branches that must not be reopened

### 2. Start with interpretable skeletons

The first batch should favor a few explainable candidates over many complicated ones.

Good early-round properties:

- clear economic rationale
- valid field/operator path
- limited operator count
- one obvious hypothesis per candidate

### 3. Preflight locally

Each candidate should be checked for:

- syntax validity
- operator compatibility
- field catalog alignment
- root scope legality
- structural similarity to prior ideas

### 4. Collect batch results in a standard layout

Each batch directory should capture:

- `multisim_children.json`
- `alpha_details/`
- `workflow_config.json`
- `batch_payload.json`
- `postprocess_status.json`
- `manifest.json`

### 5. Diagnose the batch, not just the best headline metric

The workflow emphasizes platform-facing failure modes such as:

- `LOW_SUB_UNIVERSE_SHARPE`
- `LOW_2Y_SHARPE` / `IS_LADDER_SHARPE`
- `PROD_CORRELATION`
- `SELF_CORRELATION`
- concentration and turnover problems

### 6. Choose one next action

The next round plan should be constrained by the detected failure type. Examples:

- structure failure -> rewrite the core signal idea
- setting problem -> run local setting sweeps
- concentration problem -> add light exposure compression
- correlation problem -> branch into sibling decorrelation variants
- plateau -> stop local knob-turning and refresh the research space

## Failure Classification Principles

### Structure failure

Use when multiple major checks fail together or when nearby settings behave almost identically across many variants.

### Setting-sensitive weakness

Use when the signal is directionally valid and close to target, but small parameter or neutralization changes may still matter.

### Robustness ceiling

Use when repeated repairs fail to improve a specific robustness check despite modest neighborhood exploration.

### Operational or platform issue

Use when a result is invalidated by collection or platform-side problems rather than alpha quality.

## Research Memory Principles

The repository is designed to accumulate memory about:

- families worth keeping as references
- branches that should remain frozen
- lessons from failed attempts
- plateau signatures
- which action types helped for which failure modes

This public repository includes only sanitized templates for those assets.

## Public Release Notes

This version removes:

- real alpha ids
- real multisim ids
- real author identifiers
- private expressions and field combinations
- private batch logs and research notes

It keeps the workflow structure intact so the repository still communicates how the system operates.
