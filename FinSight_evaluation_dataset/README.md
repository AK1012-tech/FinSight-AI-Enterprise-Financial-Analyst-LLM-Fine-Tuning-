# FinSight AI - Evaluation Dataset

This evaluation package is intentionally separate from the 12-document training corpus.

## Contents

- `custom_financial_eval_50.jsonl` - 50 custom evaluation questions with reference answers.
- `custom_financial_eval_50.csv` - spreadsheet-friendly version.
- `heldout_documents/` - three fictional financial documents not used in training.
- `evaluation_manifest.json` - recommended official benchmark sources and splits.

## External benchmarks

Use the official/public test splits of:
- FinQA
- ConvFinQA
- TAT-QA
- FiQA

Do not place benchmark test examples into the training corpus.

## Evaluation design

The custom set covers factual retrieval, numerical reasoning, percentage calculations,
trend analysis, risk analysis, cross-document comparison, and financial interpretation.

Run the exact same evaluation set for:

Base Model -> Checkpoint 1 -> Checkpoint 2 -> Checkpoint 3

This makes the stage-by-stage comparison reproducible.

## Important

The three held-out source documents are fictional and are not part of the 12-document training corpus.
