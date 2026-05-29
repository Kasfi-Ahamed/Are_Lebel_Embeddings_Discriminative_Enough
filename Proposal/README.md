# Proposal

This folder contains the proposal-stage notebook and supporting outputs for the SIT770 research project on label embedding discriminativeness in zero-shot intent detection. The proposal study used a 12-intent pilot subset of BANKING77 and reported a Spearman rho of -0.77 (R2 = 0.46) on representative highlighted pairs. The final study replaces this with a more conservative all-pair estimate over all 66 pairs in the subset (rho = -0.319, R2 = 0.079).

## Contents

```text
Proposal/
├── Notebook/
├── Outputs/
└── README.md
```

## Relationship to Main Research

The pilot result in this folder informed the design of the full-scale analysis implemented in the main experiment notebook. The discrepancy between the proposal rho estimate (-0.77) and the final all-pair estimate (-0.319) is explained in Section 6.1 of the main notebook.
