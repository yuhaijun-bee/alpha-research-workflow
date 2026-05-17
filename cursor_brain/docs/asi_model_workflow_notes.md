# ASI Model Workflow Notes

This file is a sanitized public note for a regional model-style workflow branch.

## Purpose

Regional model branches are useful for demonstrating how the workflow handles persistence-style signals, narrower data families, and plateau detection.

## Public-Safe Lessons

- Persistence shaping can matter more than adding extra operators.
- Once a branch survives core checks, protect the winner and test only narrow local variants.
- Correlation work should usually happen in sibling branches rather than by rewriting the current reference candidate.
- Repeatedly similar outputs across nearby variants are a strong signal that the branch is plateauing.

## What Was Removed

The private version of this note contained real family names, real alpha identifiers, and route-specific formulas. Those details were intentionally removed from the public repository.
