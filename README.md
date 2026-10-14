# FinSight AI
### Financial LLM Post-Training: DAPT → SFT → DPO → Evaluation

FinSight AI is a portfolio project that explores how domain-adaptive continued pretraining, supervised fine-tuning, and preference optimization affect an open-weight language model's performance on financial question answering.

The project implements an end-to-end post-training workflow using synthetic financial documents, parameter-efficient fine-tuning, independent benchmark evaluation, and a small LLM-as-a-Judge assessment.

> **Experimental takeaway:** In the current evaluation run, SFT produced the clearest increase in numeric-match rate. DPO did not improve the automated metrics over SFT, although the small judge sample showed a lower observed hallucination-risk rate. Results are preliminary and limited by evaluation coverage and context truncation.

---

## Pipeline

```text
Synthetic financial PDFs
        │
        ▼
1. Data preparation
   PDF extraction · cleaning · EDA · document-level split
        │
        ▼
2. Domain-Adaptive Continued Pretraining (DAPT)
   Causal language modeling · 4-bit loading · LoRA/QLoRA
        │
        ▼
   Checkpoint 1
        │
        ▼
3. Supervised Fine-Tuning (SFT)
   Financial instructions · chat-template formatting · LoRA
        │
        ▼
   Checkpoint 2
        │
        ▼
4. Direct Preference Optimization (DPO)
   Chosen/rejected response pairs · preference alignment
        │
        ▼
   Checkpoint 3
        │
        ▼
5. Evaluation
   Financial QA benchmarks · custom held-out set
   Exact Match · Token F1 · numeric match · LLM-as-a-Judge
```

## Methodology

### 1. Financial corpus preparation
- Uses 12 synthetic PDFs across three fictional financial organizations and four document types: annual reports, quarterly reports, earnings-call transcripts, and investor presentations.
- Extracts PDF text, performs light cleaning, records document metadata, and runs basic corpus EDA.
- Splits at the **document level** to reduce train/validation leakage.
- Produces `dapt_train.jsonl`, `dapt_validation.jsonl`, and `corpus_metadata.csv`.

### 2. Domain-Adaptive Continued Pretraining
- Compares two Hugging Face base-model candidates: `Qwen/Qwen3-8B` and `mistralai/Mistral-7B-v0.3`.
- Trains on raw financial text with the causal next-token prediction objective.
- Uses Unsloth, 4-bit model loading, and LoRA/QLoRA to reduce GPU memory requirements.
- Selects the DAPT checkpoint using held-out validation perplexity. The subsequent SFT/DPO pipeline uses **Mistral-7B-v0.3**.

### 3. Supervised Fine-Tuning
- Uses a synthetic dataset of **200 financial instruction-response examples** across eight task categories, including financial QA, analysis, ratio calculation, earnings analysis, risk identification, summarization, comparison, and structured analyst responses.
- Uses a deterministic, stratified 90/10 train/evaluation split.
- Continues training from Checkpoint 1 with TRL `SFTTrainer` and a Mistral chat template.
- Saves **Checkpoint 2**.

### 4. Preference optimization with DPO
- Uses **100 synthetic preference pairs** containing a prompt, a preferred response, and a rejected response.
- Preferred responses emphasize grounded claims, correct calculations, and explicit uncertainty; rejected responses model common financial-answering errors.
- Uses a deterministic 80/20 train/evaluation split.
- Continues from Checkpoint 2 using TRL `DPOTrainer` with `beta=0.1`.
- Saves **Checkpoint 3**.

### 5. Evaluation
Compares the base model and the three saved checkpoint snapshots:

- Base Mistral-7B-v0.3
- Checkpoint 1: DAPT
- Checkpoint 2: SFT
- Checkpoint 3: DPO

Evaluation sources in the current notebook:
- FinQA
- ConvFinQA
- FiQA
- TAT-QA (loaded, but the current run excluded its 20 rows per stage because of empty/invalid question or reference fields)
- Custom 50-question held-out FinSight evaluation set

Metrics:
- **Exact Match** and **Token F1** for text overlap
- **Numeric match rate** using a 2% tolerance against reference numbers
- **Reference-in-prediction** as a diagnostic indicator
- Binary yes/no answer match
- **LLM-as-a-Judge** using `gpt-4.1-mini` on 8 sampled questions per stage, scoring correctness, numerical correctness, grounding, reasoning, completeness, hallucination risk, and overall quality

The judge sample is small and is treated as directional evidence, not a definitive quality benchmark.

---

## Current evaluation snapshot

The latest uploaded evaluation run used 20 sampled questions per public benchmark and all 50 custom questions. The notebook reported 130 rows per stage before excluding invalid TAT-QA rows.

| Stage | Exact Match | Token F1 | Numeric match rate | Reference in prediction |
|---|---:|---:|---:|---:|
| Base | 3.85% | 4.34% | 2.31% | 0.00% |
| Checkpoint 1 · DAPT | 3.08% | 3.63% | 3.08% | 0.77% |
| Checkpoint 2 · SFT | 0.00% | 3.24% | **8.46%** | **2.31%** |
| Checkpoint 3 · DPO | 0.00% | 3.15% | 7.69% | **2.31%** |

**Interpretation**
- DAPT made only modest changes in this evaluation.
- SFT had the strongest numeric-match rate in the current run.
- DPO did not improve the automated metrics over SFT.
- Exact Match is strict for explanatory answers, while numeric match is only a heuristic and does not establish full calculation correctness.

### LLM-as-a-Judge snapshot

The judge successfully returned **32/32 judgments** (8 per stage).

| Stage | Correctness | Numeric correctness | Reasoning quality / 5 | Hallucination-risk rate | Overall / 5 |
|---|---:|---:|---:|---:|---:|
| Base | 0% | 0% | 1.00 | 37.5% | 1.00 |
| Checkpoint 1 · DAPT | 0% | 0% | 1.00 | 25.0% | 1.00 |
| Checkpoint 2 · SFT | 25.0% | 25.0% | 1.38 | 25.0% | 1.38 |
| Checkpoint 3 · DPO | 25.0% | 0% | 1.25 | 12.5% | 1.25 |

These scores are based on only eight judged examples per stage. They suggest possible behavioral changes, but are too small to support broad claims about model quality or hallucination reduction.

---

## Repository structure

```text
.
├── 01_financial_data_preparation.ipynb
├── 02_domain_adaptive_pretraining.ipynb
├── 03_sft_and_dpo_finsight.ipynb
├── 04_evaluation_final.ipynb
├── data/
│   ├── synthetic_financial_pdfs/       # 12 training PDFs
│   ├── FinSight_Notebook3_Datasets.zip # SFT and DPO datasets
│   └── FinSight_evaluation_dataset.zip # custom held-out evaluation
└── README.md
```

The `data/` paths above describe the recommended local repository organization. Add the generated datasets if you intend to publish them; otherwise, document how to obtain or regenerate them.

## Run the notebooks

The workflow is designed for **Google Colab with a CUDA GPU**.

1. Run `01_financial_data_preparation.ipynb` to create the cleaned corpus and document-level split.
2. Run `02_domain_adaptive_pretraining.ipynb` to compare candidates and save Checkpoint 1.
3. Run `03_sft_and_dpo_finsight.ipynb` to create Checkpoints 2 and 3.
4. Run `04_evaluation_final.ipynb` to evaluate the saved checkpoints and generate CSV/JSON/Markdown reports.

Use the same base model and checkpoint manifests throughout. Notebook 4 expects the adapter directories and manifests produced by the earlier notebooks, plus the custom evaluation dataset.

### Main technologies
- Python, PyTorch
- Hugging Face Transformers and Datasets
- PEFT, LoRA/QLoRA, Unsloth
- TRL: `SFTTrainer` and `DPOTrainer`
- pandas, NumPy, scikit-learn-style dataset splitting and evaluation utilities
- OpenAI API for the optional LLM-as-a-Judge section

## Limitations and next steps

- The training corpus and SFT/DPO examples are synthetic and relatively small.
- The current evaluation is a sampled benchmark run, not a full benchmark sweep.
- TAT-QA rows were excluded in the latest run due to invalid/empty question or reference fields; fix the loader before reporting TAT-QA performance.
- The evaluation emitted context-length warnings for inputs exceeding the configured 1,024-token limit. Implement controlled truncation or a validated longer context window before treating the metrics as final.
- Numeric match can count an answer as a match when a number happens to overlap; it is not a complete measure of financial reasoning.
- LLM-as-a-Judge uses only eight examples per stage and should be expanded for more reliable conclusions.
- The current results do not establish that the fine-tuned model is production-ready.

## Responsible use

FinSight AI is an experimental portfolio project, not a financial-advice system. Outputs should not be used for investment, lending, audit, or other high-impact decisions without domain-expert review and rigorous validation.

---

## Project goal

Demonstrate a reproducible post-training workflow and measure how **DAPT, SFT, and DPO** change a financial LLM's behavior, including the evaluation problems and limitations that must be addressed before deployment.
