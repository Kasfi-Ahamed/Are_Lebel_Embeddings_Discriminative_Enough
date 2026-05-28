# Are Label Embeddings Discriminative Enough?
### Probing Semantic Label Similarity as a Source of Fine-Grained Failure in Zero-Shot Intent Detection

**Kasfi Ahamed**  
School of Information Technology, Deakin University  
SIT770 Applied Machine Learning, 2026

---

## Overview

Zero-shot intent detection assigns utterances to the nearest intent label in embedding space, implicitly assuming that label embeddings are sufficiently discriminative. This repository presents an empirical investigation of that assumption across two standard benchmarks: BANKING77 (77 intents, 3,080 test samples) and CLINC150 (150 intents, 2,790 test samples).

Three research questions are addressed:

**RQ1.** Do intent label pairs with higher cosine similarity exhibit systematically lower pairwise zero-shot accuracy?

**RQ2.** Does the label similarity matrix structurally predict the empirical confusion matrix (Representational Similarity Analysis)?

**RQ3.** Do LLM-generated descriptions increase embedding separation, and does their effectiveness depend on label-space density?

---

## Key Findings

**RQ1 - Label similarity predicts pairwise accuracy.**  
Spearman correlations between label cosine similarity and pairwise zero-shot accuracy are negative across all six experimental conditions (rho in [-0.08, -0.32], all p < 10^-8). The relationship is confirmed across four sentence encoders with distinct architectures and training objectives, and a lexical-overlap control (Jaccard similarity) confirms the effect is semantic rather than superficial.

**RQ2 - Label geometry predicts confusion patterns.**  
The label similarity matrix is a significant structural predictor of the empirical confusion matrix via RSA (rho_RSA in [+0.15, +0.31]). Permutation z-scores of 14.0 (BANKING77) and 12.5 (CLINC150) place the observed correlations far beyond the null distribution, ruling out statistical artefacts. Within-domain RSA (rho = +0.208) is 1.75 times stronger than cross-domain RSA (rho = +0.119), mechanistically explaining why single-domain datasets show stronger RSA than multi-domain ones.

**RQ3 - Description effectiveness is moderated by label-space density.**  
Gemini-2.5-Flash descriptions improve accuracy by +3.1pp on a deliberately confusable 12-intent pilot subset (mean inter-class similarity s-bar = 0.288), but degrade accuracy by -1.8pp on full BANKING77 (s-bar = 0.204) and -5.2pp on CLINC150 (s-bar = 0.125). Descriptions restructure label geometry in ways that do not transfer uniformly to larger taxonomies, and can amplify rather than resolve the similarity-confusion relationship.

---

## Results

**Table 1. Main results across all six experimental conditions.**  
All p-values two-tailed; 95% CI bootstrapped (2,000 resamples).  
Dagger: p > 0.05 due to insufficient statistical power (n = 66 pairs).

| Dataset | Condition | Acc | RQ1 rho | R2 | RSA rho |
|---|---|---|---|---|---|
| Pilot (12 int.) | Bare Labels | .744 | -.319 | .079 | +.640 |
| Pilot (12 int.) | LLM Descriptions | .775 | -.093 (dagger) | .009 | +.514 |
| BANKING77 | Bare Labels | .586 | -.120 | .011 | +.312 |
| BANKING77 | LLM Descriptions | .569 | -.096 | .008 | +.355 |
| CLINC150 | Bare Labels | .699 | -.082 | .004 | +.151 |
| CLINC150 | LLM Descriptions | .647 | -.253 | .074 | +.142 |

**Table 2. Multi-encoder replication on BANKING77 bare labels.**  
All four encoders: RQ1 rho < 0, p <= 0.0004. RSA rho stable within 0.009 across architectures.

| Encoder | Acc | RQ1 rho | p | RSA rho |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | .586 | -.120 | < 10^-10 | .312 |
| all-mpnet-base-v2 | .588 | -.065 | < 10^-3 | .318 |
| paraphrase-MiniLM-L6-v2 | .509 | -.123 | < 10^-10 | .320 |
| multi-qa-MiniLM-L6-cos-v1 | .571 | -.105 | < 10^-8 | .321 |

---

## Figures

**Figure 1. Label-pair cosine similarity versus pairwise zero-shot accuracy.**  
All four conditions (BANKING77 and CLINC150, bare labels and LLM descriptions). Regression line and Spearman rho annotated per panel. Negative slope is consistent across all datasets and conditions.

![RQ1 scatter — label similarity vs pairwise accuracy](Outputs/figures/fig2_rq1_scatter_4panel.png)

**Figure 2. Similarity binning analysis (BANKING77, bare labels).**  
Mean pairwise accuracy by label similarity bin. Error bars show 95% bootstrapped CI. Monotone decline from 60.5% in [0.0, 0.2) to 56.2% in [0.4, 0.6) confirms the effect is continuously graded, not driven by outlier pairs.

![Similarity binning analysis](Outputs/figures/fig13_similarity_binning_analysis.png)

**Figure 3. RSA permutation null distributions.**  
BANKING77 (left) and CLINC150 (right). Red dashed line marks the observed rho. z-scores of 14.0 and 12.5 confirm the result is not a statistical artefact.

![RSA permutation null distributions](Outputs/figures/fig8_rsa_permutation_null_distribution.png)

**Figure 4. Label-space density versus description effectiveness.**  
Mean inter-class cosine similarity s-bar versus accuracy change from Gemini-2.5-Flash descriptions across three dataset scales. Monotone relationship confirms density moderates effectiveness.

![Density vs description effectiveness](Outputs/figures/fig15_density_vs_description_effectiveness.png)

**Figure 5. Lexical overlap control.**  
Cosine similarity and Jaccard similarity of label word sets as competing predictors of pairwise accuracy. Adding Jaccard to a regression yields R2 = 0.0108 — identical to the cosine-only model — confirming the embedding effect is semantic rather than lexical.

![Lexical overlap control](Outputs/figures/fig11_lexical_overlap_vs_embedding_similarity.png)

**Figure 6. Inter-class similarity distributions.**  
Bare labels versus LLM descriptions. Descriptions shift the distribution leftward, reducing mean inter-class similarity, but at a cost that scales with label-space density.

![Inter-class similarity distributions](Outputs/figures/fig4_similarity_distributions.png)

---

## Repository Structure

```
Are_Lebel_Embeddings_Discriminative_Enough/
├── README.md
├── requirements.txt
├── label_embeddings_experiment.ipynb   # Main experiment notebook
├── Notebook/                           # Notebook development files
├── Outputs/
│   ├── figures/                        # All generated figures
│   └── data/                           # CSV result tables
└── paper/
    └── Are_Label_Embeddings_paper.pdf  # Full research paper
```

---

## Reproducibility

**Environment setup:**

```bash
pip install -r requirements.txt
```

**Datasets:** BANKING77 is loaded automatically from the PolyAI public CSV release. CLINC150 is loaded from the CLINC OOS-Eval repository. No manual download is required.

**API key:** A Gemini API key is required for LLM description generation (Section 9 of the notebook). Free-tier keys are available at [aistudio.google.com](https://aistudio.google.com). Set the key in Cell 1 of the notebook before running Section 9 or 10c.

**Offline mode:** All pre-computed LLM descriptions are included in the notebook. If no API key is provided, the notebook falls back to these pre-computed descriptions automatically.

**Encoder:** All experiments use frozen sentence encoders from the `sentence-transformers` library. No fine-tuning is applied at any stage.

**Runtime:** Full experiment pipeline (Sections 10–13g) takes approximately 20–40 minutes on a standard CPU.

---

## Experimental Method

The experiment follows a three-stage label discriminativeness audit framework:

**Stage 1 — Per-pair analysis (RQ1).** All K labels are embedded using a frozen sentence encoder. For every pair (i, j), label cosine similarity s_ij and pairwise zero-shot accuracy acc_ij are computed. Spearman rho with bootstrapped 95% CI (2,000 resamples) is reported across all K-choose-2 pairs.

**Stage 2 — RSA diagnostic (RQ2).** Spearman rho between the upper-triangular vectors of the label similarity matrix S and the symmetrised empirical confusion-rate matrix C is computed. A 1,000-permutation null distribution validates that the result is not a statistical artefact. For CLINC150, the analysis is repeated separately for intra-domain and inter-domain pairs.

**Stage 3 — LLM description intervention (RQ3).** Bare label names are replaced with discriminative descriptions generated by Gemini-2.5-Flash using the prompt: "Write one discriminative sentence (15-25 words) that precisely describes what a user is asking and clearly distinguishes this intent from semantically similar ones." Labels are re-embedded and the full pipeline is repeated.

---

## Citation

If you use this work, please cite:

```
Ahamed, K. (2026). Are Label Embeddings Discriminative Enough?
Probing Semantic Label Similarity as a Source of Fine-Grained Failure
in Zero-Shot Intent Detection. SIT770 Applied Machine Learning,
Deakin University.
```

---

## Acknowledgements

This work was completed as part of SIT770 (Applied Machine Learning) at Deakin University. The author thanks the open-source communities behind `sentence-transformers`, BANKING77, and CLINC150.
