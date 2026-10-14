# FinSight AI Notebook 3 Datasets

SFT: 200 synthetic financial instruction examples across eight categories.
DPO: 100 synthetic preference pairs with `prompt`, `chosen`, and `rejected`.

SFT JSONL also contains a `messages` field that can be adapted directly to a chat-template based SFT pipeline.

Training progression:
Checkpoint 1 (DAPT) -> SFT -> Checkpoint 2 -> DPO -> Checkpoint 3

All source companies and financial values are fictional/synthetic.
