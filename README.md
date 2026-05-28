# Are Label Embeddings Discriminative Enough?
### Probing Semantic Label Similarity as a Source of Fine-Grained Failure in Zero-Shot Intent Detection

**Kasfi Ahamed**  
School of Information Technology, Deakin University  
SIT770 Applied Machine Learning, 2026

---

## Overview

Zero-shot intent detection assigns utterances to the nearest intent label in embedding space, implicitly assuming that those label embeddings are sufficiently discriminative. This repository presents an empirical investigation of that assumption across two standard benchmarks: BANKING77 (77 intents, 3,080 test samples) and CLINC150 (150 intents, 2,790 test samples).

Three research questions are addressed:

**RQ1.** Do intent label pairs with higher cosine similarity exhibit systematically lower pairwise zero-shot accuracy?

**RQ2.** Does the label similarity matrix structurally predict the empirical confusion matrix via Representational Similarity Analysis (RSA)?

**RQ3.** Do LLM-generated descriptions increase embedding separation, and does their effectiveness depend on label-space density?

---

## Repository Structure

```
Are_Lebel_Embeddings_Discriminative_Enough/
├── README.md
├── Notebook/
│   └── label_embeddings_experiment.ipynb   # Full experiment notebook
└── Outputs/
    └── figures/                            # All generated figures (16 PNG files)
```

---

## Key Findings

**RQ1 — Label similarity predicts pairwise accuracy.**  
Spearman correlations between label cosine similarity and pairwise zero-shot accuracy are negative across all six experimental conditions (rho in [-0.08, -0.32], all p < 10^-8). The relationship is confirmed across four sentence encoders with distinct architectures and training objectives. A lexical-overlap control using Jaccard similarity of label word sets confirms the effect is semantic rather than superficial: the partial Spearman rho controlling for Jaccard is -0.078 (p = 2.3 x 10^-5).

**RQ2 — Label geometry predicts confusion patterns.**  
The label similarity matrix is a significant structural predictor of the empirical confusion matrix via RSA (rho_RSA in [+0.15, +0.31]). Permutation z-scores of 14.0 (BANKING77) and 12.5 (CLINC150) place the observed correlations far beyond the null distribution. Within-domain RSA on CLINC150 (rho = +0.208, 675 pairs) is 1.75 times stronger than cross-domain RSA (rho = +0.119, 10,500 pairs), mechanistically explaining the weaker aggregate RSA on multi-domain datasets.

**RQ3 — Description effectiveness is moderated by label-space density.**  
Gemini-2.5-Flash descriptions improve accuracy by +3.1pp on a deliberately confusable 12-intent pilot (mean inter-class similarity s-bar = 0.288), but degrade accuracy by -1.8pp on full BANKING77 (s-bar = 0.204) and -5.2pp on CLINC150 (s-bar = 0.125). Descriptions restructure label geometry in ways that amplify rather than resolve the similarity-confusion relationship on larger taxonomies.

---

## Results

**Table 1. Main results across all six experimental conditions.**  
All p-values two-tailed; 95% CI bootstrapped (2,000 resamples). Dagger: p > 0.05 due to insufficient power (n = 66 pairs).

| Dataset | Condition | Acc | RQ1 rho | R2 | RSA rho |
|---|---|---|---|---|---|
| Pilot (12 int.) | Bare Labels | .744 | -.319 | .079 | +.640 |
| Pilot (12 int.) | LLM Descriptions | .775 | -.093 † | .009 | +.514 |
| BANKING77 | Bare Labels | .586 | -.120 | .011 | +.312 |
| BANKING77 | LLM Descriptions | .569 | -.096 | .008 | +.355 |
| CLINC150 | Bare Labels | .699 | -.082 | .004 | +.151 |
| CLINC150 | LLM Descriptions | .647 | -.253 | .074 | +.142 |

**Table 2. Multi-encoder replication on BANKING77 bare labels.**  
All four encoders yield RQ1 rho < 0 (p <= 0.0004). RSA rho stable within 0.009 across architectures.

| Encoder | Acc | RQ1 rho | p | RSA rho |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | .586 | -.120 | < 10^-10 | .312 |
| all-mpnet-base-v2 | .588 | -.065 | < 10^-3 | .318 |
| paraphrase-MiniLM-L6-v2 | .509 | -.123 | < 10^-10 | .320 |
| multi-qa-MiniLM-L6-cos-v1 | .571 | -.105 | < 10^-8 | .321 |

**Table 3. Label-space density versus accuracy change from LLM descriptions.**

| Dataset | Mean sim (s-bar) | Acc (bare) | Delta Acc |
|---|---|---|---|
| Pilot (12 intents) | 0.288 | .744 | +3.1 pp |
| BANKING77 (77 int.) | 0.204 | .586 | -1.8 pp |
| CLINC150 (150 int.) | 0.125 | .699 | -5.2 pp |

---

## Figures

**Figure 1. Label-pair cosine similarity versus pairwise zero-shot accuracy.**  
All four conditions (BANKING77 and CLINC150, bare labels and LLM descriptions). Regression line and Spearman rho annotated per panel. Negative slope is consistent across all datasets and conditions.

![RQ1 scatter](Outputs/figures/fig2_rq1_scatter_4panel.png)

**Figure 2. RQ1 replicated across four sentence encoders (BANKING77, bare labels).**  
All four encoders show a negative Spearman rho, all significant at p <= 0.0004. RSA rho stable within 0.009 across architectures.

![Multi-encoder replication](Outputs/figures/fig7_rq1_multi_encoder_replication.png)

**Figure 3. RSA permutation null distributions.**  
BANKING77 (left) and CLINC150 (right). Red dashed line marks the observed rho. z-scores of 14.0 and 12.5 confirm the result is not a statistical artefact.

![RSA permutation null distributions](Outputs/figures/fig8_rsa_permutation_null_distribution.png)

**Figure 4. Similarity binning analysis (BANKING77, bare labels).**  
Mean pairwise accuracy by label similarity bin with 95% bootstrapped CI. Monotone decline from 60.5% in [0.0, 0.2) to 56.2% in [0.4, 0.6) confirms the effect is continuously graded, not driven by outlier pairs.

![Similarity binning analysis](Outputs/figures/fig13_similarity_binning_analysis.png)

**Figure 5. Label-space density versus description effectiveness.**  
Mean inter-class cosine similarity s-bar versus accuracy change from Gemini-2.5-Flash descriptions across three dataset scales. The monotone relationship confirms density moderates effectiveness.

![Density vs description effectiveness](Outputs/figures/fig15_density_vs_description_effectiveness.png)

**Figure 6. Lexical overlap control.**  
Cosine similarity and Jaccard similarity as competing predictors of pairwise accuracy. Adding Jaccard to a regression yields R2 = 0.0108, identical to the cosine-only model, confirming the effect is semantic rather than lexical.

![Lexical overlap control](Outputs/figures/fig11_lexical_overlap_vs_embedding_similarity.png)

**Figure 7. CLINC150 intra- versus inter-domain RSA.**  
Within-domain label geometry is 1.75 times more predictive of confusion than cross-domain pairs, mechanistically explaining the weaker aggregate RSA on multi-domain datasets.

![Intra vs inter domain RSA](Outputs/figures/fig10_clinc150_intra_inter_domain_rsa.png)

**Figure 8. BANKING77 label similarity heatmap (bare labels).**  
Cosine similarity matrix across all 77 intent label embeddings. High-similarity clusters correspond to the confusion hotspots identified in the error analysis.

![BANKING77 similarity heatmap](Outputs/figures/fig1_banking77_heatmap_bare.png)

---

## Reproducibility

**Setup:**

```bash
pip install sentence-transformers datasets scipy numpy pandas matplotlib seaborn google-generativeai
```

**Datasets:** BANKING77 loads automatically from the PolyAI public CSV release. CLINC150 loads from the CLINC OOS-Eval repository. No manual download is required; the notebook handles both with silent fallback between sources.

**LLM descriptions:** A Gemini API key is required only to regenerate descriptions (Sections 9 and 10c of the notebook). Pre-computed descriptions for all 77 BANKING77 and 150 CLINC150 intents are embedded in the notebook and used automatically if no key is provided.

**Runtime:** Full pipeline (Sections 10 through 13g) takes approximately 20-40 minutes on a standard CPU.

**Encoder:** All experiments use frozen sentence encoders from `sentence-transformers`. No fine-tuning is applied at any stage.

---

## Experimental Method

The study follows a three-stage label discriminativeness audit framework.

**Stage 1 — Per-pair analysis (RQ1).** All K labels are embedded using a frozen sentence encoder. For every pair (i, j), label cosine similarity s_ij and pairwise zero-shot accuracy acc_ij are computed. Spearman rho with bootstrapped 95% CI (2,000 resamples) is reported across all K-choose-2 pairs.

**Stage 2 — RSA diagnostic (RQ2).** Spearman rho between the upper-triangular vectors of the label similarity matrix S and the symmetrised empirical confusion-rate matrix C is computed. A 1,000-permutation null distribution validates statistical significance. For CLINC150, the analysis is repeated separately for intra-domain and inter-domain pairs.

**Stage 3 — LLM description intervention (RQ3).** Bare label names are replaced with discriminative descriptions generated by Gemini-2.5-Flash using the prompt: "Write one discriminative sentence (15-25 words) that precisely describes what a user is asking and clearly distinguishes this intent from semantically similar ones." Labels are re-embedded and the full pipeline is repeated.

---

## Citation

```
Ahamed, K. (2026). Are Label Embeddings Discriminative Enough?
Probing Semantic Label Similarity as a Source of Fine-Grained Failure
in Zero-Shot Intent Detection. SIT770 Applied Machine Learning,
Deakin University.
```

---

## Acknowledgements

This work was completed as part of SIT770 (Applied Machine Learning) at Deakin University. The author thanks the open-source communities behind `sentence-transformers`, BANKING77, and CLINC150.
