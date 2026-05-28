# Are Label Embeddings Discriminative Enough?
### Probing Semantic Label Similarity as a Source of Fine-Grained Failure in Zero-Shot Intent Detection

**Author:** Kasfi Ahamed, Deakin University (SIT770, 2026)

---

## Overview

This repository contains the code and figures for the SIT770 research proposal.
The study investigates whether sentence embeddings of bare intent label names
are sufficiently discriminative for fine-grained zero-shot intent detection,
and whether label-pair cosine similarity structurally predicts classification failure.

---

## Research Questions

- **RQ1:** Do intent label pairs with higher cosine similarity exhibit systematically lower pairwise zero-shot accuracy?
- **RQ2:** Does the label similarity matrix structurally predict the empirical confusion matrix (RSA-style)?
- **RQ3:** Do LLM-generated intent descriptions increase embedding separation and reduce the similarity-confusion correlation?

---

## Key Results (Pilot Study, 12-intent subset, BANKING77)

| Metric | Value |
|---|---|
| Pearson R² | 0.464 |
| Spearman ρ | −0.77 |
| p-value | < 0.001 |
| Accuracy range | 63.7% – 100.0% |
| Similarity range | 0.02 – 0.71 |

---

## Figures

### Figure 1 - Intent Label Embedding Similarity (Heatmap)
![Intent Label Similarity Heatmap](https://raw.githubusercontent.com/Kasfi-Ahamed/Are_Lebel_Embeddings_Discriminative_Enough/main/Outputs/intent_label_similarity.png)

*Cosine similarity between sentence embeddings of 12 BANKING77 intent label names.
Semantically proximate pairs (card\_arrival/card\_delivery\_estimate, transfer\_not\_received/transfer\_timing) cluster at high values.*

---

### Figure 2 - Label Similarity vs. Zero-Shot Accuracy (Scatter)
![Label Similarity vs Zero-Shot Accuracy](https://raw.githubusercontent.com/Kasfi-Ahamed/Are_Lebel_Embeddings_Discriminative_Enough/main/Outputs/scatter_similarity_vs_accuracy.png)

*Label-pair cosine similarity vs. zero-shot pairwise accuracy (R² = 0.464, Spearman ρ = −0.77, p < 0.001).
Orange points are the five highlighted pairs from Table 1 in the paper.*

---

## Results Table

Representative intent label pairs from the 12-intent pilot subset, ordered by cosine similarity.
The five upper rows correspond to the highlighted orange points in Figure 2.

| Label Pair | Cos Sim | Zero-Shot Accuracy |
|---|---|---|
| card\_arrival / card\_delivery\_estimate | 0.71 | 63.7% |
| card\_arrival / card\_linking | 0.58 | 87.5% |
| card\_arrival / card\_not\_working | 0.55 | 85.0% |
| card\_linking / card\_not\_working | 0.55 | 83.8% |
| card\_arrival / lost\_or\_stolen\_card | 0.53 | 88.8% |
| | | |
| card\_not\_working / transfer\_timing *(contrast)* | 0.02 | 100.0% |

> Higher cosine similarity between label names → Lower zero-shot classification accuracy
> (R² = 0.464, Spearman ρ = −0.77, p < 0.001)


---

## Notebook

[View the full experiment notebook](https://github.com/Kasfi-Ahamed/Are_Lebel_Embeddings_Discriminative_Enough/blob/main/Notebook/NLP_research_updated.ipynb)

The notebook covers:
1. BANKING77 dataset download (direct from PolyAI GitHub, no manual download needed)
2. Sentence embedding of 12 intent label names using `all-MiniLM-L6-v2`
3. Zero-shot pairwise accuracy computation across all 66 label pairs
4. Figure 1, cosine similarity heatmap
5. Figure 2, similarity vs. accuracy scatter plot
6. Results Table

---

## Repository Structure

```
├── Notebook/
│   └── NLP_research_updated.ipynb        # Full experiment notebook
└── Outputs/
    ├── intent_label_similarity.png        # Figure 1  heatmap
    └── scatter_similarity_vs_accuracy.png # Figure 2  scatter plot
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Kasfi-Ahamed/Are_Lebel_Embeddings_Discriminative_Enough.git
cd Are_Lebel_Embeddings_Discriminative_Enough

# 2. Install dependencies
pip install sentence-transformers scipy scikit-learn matplotlib requests

# 3. Open and run the notebook
jupyter notebook Notebook/NLP_research_updated.ipynb
```

Data is downloaded automatically inside the notebook, no manual setup required.

---

## Model and Dataset

| Item | Detail |
|---|---|
| **Encoder** | `all-MiniLM-L6-v2` (sentence-transformers) |
| **Dataset** | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets), 77 banking intents, 3,080 test samples |
| **Scale-up** | Full study extends to all BANKING77 pairs and CLINC150 |

---

## Citation

```
Ahamed, K. (2026). Are Label Embeddings Discriminative Enough?
Probing Semantic Label Similarity as a Source of Fine-Grained Failure
in Zero-Shot Intent Detection. SIT770, Deakin University.
https://github.com/Kasfi-Ahamed/Are_Lebel_Embeddings_Discriminative_Enough
```
