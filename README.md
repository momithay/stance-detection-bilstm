# stance-detection-bilstm

# Stance Detection in Tweets with Conditional BiLSTM Encoding

A target-aware stance classification system that predicts whether a tweet is **FAVOR**, **AGAINST**, or **NONE** with respect to a given target (e.g. a topic or public figure), using GloVe-Twitter embeddings and a bidirectional LSTM with conditional (tweet + target) encoding — evaluated on both a standard benchmark split and an independently hand-labeled test set to check real-world generalization.

---

## Table of Contents
- [Overview](#overview)
- [Problem Framing](#problem-framing)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Text Preprocessing](#text-preprocessing)
- [Model Architecture](#model-architecture)
- [Experimentation & Ablation Study](#experimentation--ablation-study)
- [Evaluation Methodology](#evaluation-methodology)
- [Results](#results)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

**Stance detection** asks a more nuanced question than standard sentiment analysis: not just *"is this tweet positive or negative?"* but *"is the author for or against this specific target?"* — which requires the model to reason jointly about the tweet text **and** the target it's being evaluated against. This project implements a **conditional encoding BiLSTM** (inspired by Augenstein et al.'s stance detection work) that encodes the tweet conditioned on the target representation, rather than treating stance as plain text classification.

## Problem Framing

Given a `(tweet, target)` pair, predict one of three stance labels:

- **FAVOR** — the tweet expresses support for the target
- **AGAINST** — the tweet expresses opposition to the target
- **NONE** — the tweet is neutral or the stance can't be inferred

This is harder than generic sentiment classification because the same tweet can be FAVOR toward one target and AGAINST another — the model has to learn target-conditioned representations, not just a bag of sentiment-bearing words.

## Dataset

- **Primary data**: the [SemEval-2016 Task 6 Stance Dataset](https://www.saifmohammad.com/WebPages/StanceDataset.htm) — tweets labeled with `Target`, `Tweet`, `Stance`, and `Sentiment`, covering 5 distinct targets.
- **Independent test set**: a **manually collected and labeled** secondary dataset, used specifically to stress-test whether the model generalizes beyond the benchmark distribution rather than overfitting to SemEval-specific tweet phrasing.
- **Class balance**: targets are evenly distributed, but stance labels are imbalanced (AGAINST is the majority class) — handled through evaluation choices (macro/weighted F1) rather than resampling, to keep the benchmark comparable to published results.

## Exploratory Data Analysis

Before modeling, the notebook includes a full EDA pass:
- Target and stance label distributions (bar plots, pivot tables)
- Stance-vs-sentiment cross tabulation
- Word clouds over the tweet corpus to surface dominant vocabulary
- Tweet-level statistics: average word length, hashtag counts, most frequent @mentions
- Missing-value and schema checks across train/test splits

## Text Preprocessing

Tweets are noisy, informal text, so a dedicated normalization pipeline was built to maximize overlap with the pretrained embedding vocabulary:

1. **OOV (out-of-vocabulary) analysis** — built a vocabulary from the raw training tweets and checked coverage against GloVe-Twitter embeddings, surfacing which tokens were missing.
2. **Contraction expansion** — a custom contraction dictionary (`"don't" → "do not"`, `"won't" → "will not"`, etc.) to recover standard word forms.
3. **Tokenization** — NLTK's `TweetTokenizer`, which is Twitter-aware (handles hashtags, mentions, emoticons, elongated words) rather than a generic whitespace tokenizer.
4. **Re-validation** — recomputed OOV rate after preprocessing to confirm the cleanup measurably improved embedding coverage before moving to modeling.
5. **Vectorization** — Keras `TextVectorization` layer (max 20,000 tokens, sequence length 200) built directly into the model graph, so raw text strings can be fed straight into the model at inference time.

## Model Architecture

**Conditional encoding**: the tweet and the target are encoded through separate input branches that share the pretrained embedding matrix, allowing the network to represent the tweet *in the context of* the target rather than independently.

```
Tweet (string) ──▶ TextVectorization ──▶ Embedding(GloVe-Twitter-100d) ──▶ BiLSTM stack ──┐
                                                                                            ├──▶ Dense → Softmax(3)  [FAVOR / AGAINST / NONE]
Target (string) ─▶ TextVectorization ──▶ Embedding(GloVe-Twitter-100d) ──▶ BiLSTM/Cond ───┘
```

- **Embeddings**: pretrained **GloVe Twitter 100d** vectors — chosen over generic GloVe/Word2Vec because they're trained on Twitter's own informal register, giving much better coverage of hashtags, abbreviations, and slang.
- **Sequence encoder**: configurable, parameterized `build_model()` function supporting:
  - Stacked LSTM layers (`num_layers`)
  - Bidirectional wrapping (`bi=True`)
  - L2 kernel regularization, dropout & recurrent dropout
  - Optional attention and batch normalization
  - Frozen vs. trainable embeddings (`trainable` flag) for fine-tuning

This parameterization was the backbone of a structured ablation study (below) rather than hand-tuning a single fixed architecture.

## Experimentation & Ablation Study

Models were built up incrementally, with each variant's training/validation curves logged and compared:

| Stage | Change | Purpose |
|---|---|---|
| **Skeleton** | 1-unit, 1-layer BiLSTM | Sanity-check the pipeline end-to-end |
| **Baseline** | 64-unit BiLSTM, GloVe embeddings | Establish a real performance floor |
| **Width** | 128 units | Test capacity vs. overfitting trade-off |
| **Depth** | 2 stacked BiLSTM layers | Test representational depth |
| **Dropout** | Dropout + recurrent dropout added | Address overfitting seen in width/depth runs |
| **Fine-tuning** | Unfreeze embeddings, resume training at a much lower LR (`1e-5`) after initial frozen-embedding convergence | Recover task-specific signal from the embedding space itself |

TensorBoard was used throughout for live training/validation monitoring, and a custom `plotter()` utility overlaid accuracy/loss curves across all experiment variants for direct comparison.

## Evaluation Methodology

Rather than reporting a single accuracy number, the model is evaluated with:
- **Classification report** (per-class precision / recall / F1)
- **Confusion matrix** (heatmap visualization)
- **F1 — micro, macro, and weighted** — to separately surface overall performance and performance on minority classes (FAVOR/NONE), which raw accuracy would obscure given the AGAINST-majority class imbalance
- **Two-stage evaluation**: 1) the official SemEval test split, and 2) a fully independent, manually labeled dataset — a deliberate generalization check, since strong benchmark performance alone doesn't guarantee the model has learned genuine stance reasoning rather than benchmark-specific artifacts.

## Results

**SemEval provided test set** (n = 1,249):

| Stance | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| AGAINST | 0.68 | 0.66 | 0.67 |
| FAVOR | 0.40 | 0.37 | 0.39 |
| NONE | 0.35 | 0.42 | 0.38 |
| **Accuracy** | | | **0.545** |
| **Macro F1** | | | **0.479** |
| **Weighted F1** | | | **0.548** |

**Independent (manually labeled) test set** (n = 64):

| Stance | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| AGAINST | 0.48 | 0.65 | 0.55 |
| FAVOR | 0.57 | 0.53 | 0.55 |
| NONE | 0.27 | 0.20 | 0.23 |
| **Accuracy** | | | **0.48** |
| **Macro F1** | | | **0.44** |

The drop from the benchmark to the independent set is itself an informative result — it quantifies the generalization gap between benchmark performance and real-world tweets, which is exactly the kind of result a benchmark-only evaluation would have hidden.

## Key Engineering Decisions

- **Conditional encoding over plain text classification** — treating stance detection as `(tweet, target)` → label rather than `tweet` → label, reflecting the actual structure of the task.
- **Domain-matched embeddings** — GloVe *Twitter* embeddings specifically, plus a preprocessing pipeline built around measurably reducing OOV rate against that embedding space rather than generic NLP cleaning.
- **Structured ablation over one-shot training** — every architectural change (width, depth, dropout, fine-tuning) was isolated and compared against the previous best, with training curves logged for each, rather than jointly tuning everything at once.
- **Two-tier evaluation** — deliberately built and labeled an independent test set to validate generalization beyond the benchmark, and reported the (sizeable) performance gap honestly rather than only citing benchmark numbers.
- **Metric choice matched to class imbalance** — macro/weighted F1 and full confusion matrices reported alongside accuracy, since accuracy alone is misleading on an AGAINST-majority dataset.

## Tech Stack

`Python` · `TensorFlow / Keras` · `GloVe-Twitter Embeddings` · `NLTK` (`TweetTokenizer`) · `scikit-learn` · `pandas` / `NumPy` · `Seaborn` / `Matplotlib` / `WordCloud` · `TensorBoard` · Google Colab (GPU runtime)

## Repository Structure

```
.
├── stance-detection-bilstm.ipynb   # EDA, preprocessing, model ablations, evaluation
├── independent_data.csv            # manually collected & labeled generalization test set
├── .gitignore
└── README.md
```

> **Note on data**: the SemEval-2016 Stance Dataset is available from its original source (linked above) under its own terms of use and is not redistributed here.

## How to Run

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the [SemEval-2016 Stance Dataset](https://www.saifmohammad.com/WebPages/StanceDataset.htm) and the pretrained [GloVe Twitter embeddings](https://nlp.stanford.edu/projects/glove/) (100d), placing them under `data/`.
3. Open `notebooks/stance_detection_bilstm.ipynb` and run cells top-to-bottom (originally developed in Google Colab with data mounted from Drive — update the working-directory cell if running locally).
4. TensorBoard logs are written during training and can be viewed with:
   ```bash
   tensorboard --logdir <logdir>/models
   ```

## Limitations & Future Work

- **Transformer-based encoders**: swapping the BiLSTM for a fine-tuned transformer (e.g., BERTweet, which is pretrained specifically on tweets) would likely close a meaningful part of the benchmark-vs-independent-set gap.
- **Attention mechanism**: the `build_model()` function already exposes an `attention` flag — a fully worked-through attention variant (visualizing which tweet tokens drive the stance decision) is a natural next step for interpretability.
- **Cross-target generalization**: formally testing "leave-one-target-out" splits would better measure whether the model learns transferable *stance reasoning* versus target-specific vocabulary shortcuts.
- **Larger independent test set**: 64 examples is enough to reveal a generalization gap but too small for statistically robust per-class conclusions — expanding this set would sharpen the analysis.
