# Main Research

This folder contains the primary experiment notebook, generated outputs, and dependency file for the final study.

## Contents

```text
Main Research/
├── Notebook/
│   └── label_embeddings_experiment.ipynb
├── outputs/
│   ├── figures/    # 16 generated PNG figures
│   └── data/       # CSV result tables
└── requirements.txt
```

## Notebook Structure

Sections 1-4 define environment setup, label schemas, and sentence encoder configuration. Section 5 handles dataset loading for BANKING77 full, BANKING77 pilot, and CLINC150. Section 6 implements the zero-shot classification engine used across all evaluations. Sections 7-9 present pairwise analysis for RQ1, RSA analysis for RQ2, and LLM description generation. Section 10 executes the four main experimental configurations, with Section 10b providing multi-encoder replication across four encoders and Section 10c implementing the selective description strategy. Sections 11-13g provide results tables, full figure generation, error analysis, threshold analysis, intra/inter-domain RSA, lexical overlap control, similarity binning, AUC analysis, and density analysis.

## Setup

`pip install -r requirements.txt`

A Gemini API key must be set in Cell 1 only when regenerating LLM descriptions. Pre-computed descriptions are included.

## Generated Figures

```text
outputs/figures/fig1_banking77_heatmap_bare.png
outputs/figures/fig1b_banking77_heatmap_desc.png
outputs/figures/fig2_rq1_scatter_4panel.png
outputs/figures/fig3_rsa_scatter_4panel.png
outputs/figures/fig4_similarity_distributions.png
outputs/figures/fig5_clinc150_heatmap_bare.png
outputs/figures/fig5b_clinc150_heatmap_desc.png
outputs/figures/fig6_clinc150_domain_accuracy.png
outputs/figures/fig7_rq1_multi_encoder_replication.png
outputs/figures/fig8_rsa_permutation_null_distribution.png
outputs/figures/fig9_similarity_threshold_analysis.png
outputs/figures/fig10_clinc150_intra_inter_domain_rsa.png
outputs/figures/fig11_lexical_overlap_vs_embedding_similarity.png
outputs/figures/fig12_selective_description_strategy.png
outputs/figures/fig13_similarity_binning_analysis.png
outputs/figures/fig15_density_vs_description_effectiveness.png
```
