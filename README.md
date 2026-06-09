已在你原有内容基础上，在多个关键位置强化了"当前数据仅为部分"的说明：

---

# UR-TabQA: Toward Verifiable Uncertainty Reduction in Industrial Table Question Answering

## Overview

We introduce **UR-TabQA**, a large-scale bilingual benchmark designed to advance table question answering (TableQA) research under real-world industrial settings. Unlike prior benchmarks built on Wikipedia tables or narrow domains, UR-TabQA frames industrial TableQA as an **uncertainty reduction** process: progressively narrowing the candidate answer space through multi-table grounding, column selection, row filtering, table operations, and answer generation.

UR-TabQA contains **1,932 tables** (1,101 Chinese + 831 English) spanning **30 industrial sub-domains** across 7 primary domains, covering four critical table types: single tables, multiple tables, complex structured tables (hierarchical/nested headers), and extremely large-scale tables. Beyond final answers, the benchmark provides **multi-step dual-mode reasoning annotations** and **executable Python code**, making the uncertainty-reduction process fully observable and verifiable.

We evaluate **24 competitive baseline models** on UR-TabQA. Experimental results show that even the top-performing closed-source model, GPT-5.4, achieves only **73.45%** overall accuracy, underscoring the substantial challenge posed by real-world industrial TableQA.

> ⚠️ **Important Notice on Data Availability:** The data currently released in this repository constitutes **only a partial subset** of the complete UR-TabQA benchmark. The full dataset — including all 1,932 tables, 5,523 question-answer pairs, and both SFT datasets — will be released in its entirety upon paper acceptance. **Please do not draw conclusions about the full benchmark scale or coverage based solely on the data currently available here.**

---

## The TabEIRL Framework

We propose **TabEIRL** (Table Effective Information Reinforcement Learning), a table-specific two-stage reinforcement learning framework driven by uncertainty reduction.

TabEIRL combines two complementary training signals:

**1. Entropy Reduction-Based Information Effectiveness.** At each reasoning step, we compute the relative entropy reduction over the candidate cell space. Tokens within a reasoning step receive step-specific entropy signals rather than a single global reward, encouraging the model to progressively and correctly narrow the answer space.

**2. Progressive Solving Reward (PSR).** A piecewise outcome reward evaluates code extraction, executability, and answer correctness. Additionally, an inter-group CodeBLEU-based similarity reward guides executable-but-incorrect code toward structurally and semantically correct implementations, providing a meaningful optimization signal before final correctness is achieved.

The combined reward jointly optimizes uncertainty-reduction reasoning quality and answer correctness. Applied to Qwen3-8B, TabEIRL surpasses all 19 open-source baselines and several larger models, achieving **68.07%** on UR-TabQA — the best result among open-source models — while also generalizing robustly to four external TableQA benchmarks (WTQ, AITQA, MiMoTable, HiTab).

---

## Dataset & Parameter Preview

> ⚠️ **Partial Release Notice:** The data provided in this repository is a **representative preview subset** of the full UR-TabQA benchmark. The complete dataset, comprising all 1,932 tables and 5,523 annotated samples across all 30 industrial sub-domains, will only be available upon paper acceptance. The statistics below describe the **full benchmark** as reported in the paper, not the current partial release.

The dataset is organized in the `data` directory, split into `zh` (Chinese) and `en` (English) subdirectories. Key statistics of the **complete benchmark** are summarized below:

| Property | Full Benchmark |
|---|---|
| Number of Tables | 1,932 |
| Chinese / English Tables | 1,101 / 831 |
| Number of Industrial Domains | 30 (across 7 primary domains) |
| Number of Questions | 5,523 |
| Avg. Questions per Table | 2.86 |
| Avg. Rows per Table | 138.3 |
| Avg. Cells per Table | 1,359.3 |
| Extremely Large-Scale Tables (>50K cells) | 128 |
| SFT Dataset (Thinking Mode) Avg. Response Length | 9,366 tokens |
| SFT Dataset (No-Thinking Mode) Avg. Response Length | 1,321 tokens |

The current partial release includes a representative sample from each of the 7 primary domains and all 4 table structural types, intended to allow reviewers and early adopters to verify data format, annotation quality, and benchmark structure. Both `zh` and `en` subdirectories follow the same directory structure as the full release:

- **Train/test splits** (8:2 ratio)
- **Two SFT datasets**: transformed-thinking mode and no-thinking mode, each containing ⟨table, question, reasoning process⟩ triples
- **RL dataset**: ⟨table, question, answer⟩ triples
- **Authentic tables** spanning industrial sub-domains

The `tabeirl_trainer.yaml` file provides key parameter configurations for the RL training process within TabEIRL (e.g., clipping hyperparameters `ε_high = 0.28`, `ε_low = 0.2`, prompt/response length up to 16,384 tokens), serving as a practical reference for model adaptation and reproduction.

---

## Data & Code Availability

The complete UR-TabQA dataset and full codebase will be made publicly available upon acceptance of our paper. We are committed to supporting fully reproducible research and will release the entire benchmark, all annotations, and the complete TabEIRL training and evaluation pipeline following the review process.
