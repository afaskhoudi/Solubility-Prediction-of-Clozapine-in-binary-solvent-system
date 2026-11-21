# Solubility Prediction of Clozapine in Binary Solvent Systems

Machine-learning models for predicting the aqueous/organic solubility (log *S*) of the drug **clozapine** in binary solvent mixtures, with a focus on how much physical structure and physics-based pretraining actually buy over a plain data-driven model.

Two ingredients drive the design:

- **COSMO-RS / XTB pretraining.** A model is first trained on 6000 computationally generated solubility points (`combined_file_XTB.csv`), then fine-tuned on the 684 experimental points. The pretraining step can optionally carry a PCA basis fitted once on the unique solvents seen during pretraining.
- **A physics-informed output head.** Instead of predicting log *S* directly, the network predicts the two coefficients of a van 't Hoff–type relation and the temperature is applied analytically:

  $$\log S = A + \frac{B}{T}, \qquad A \in [2.5,\,6.5], \quad B \in [-2500,\,-500]$$

  *A* and *B* come from a small MLP applied to the mole-fraction-weighted Mordred descriptors of the two solvents, and each is bounded through a sigmoid so the head cannot leave physically plausible ranges.

The ablations walk through the crossings of three switches: **PCA on/off**, **COSMO-RS pretraining on/off**, **physics head on/off**, plus a set of generalization splits (unseen solvent pair, temperature interpolation/extrapolation, composition interpolation/extrapolation).

## Repository layout

```
.
├── Clozapine_dataset.csv      # 684 experimental solubility points
├── combined_file_XTB.csv      # 6000 COSMO-RS/XTB points used for pretraining
├── notebooks/
│   ├── Baseline (No physics_NO PCA- NO pretrain).ipynb
│   ├── Data-Driven.ipynb
│   ├── Hybrid model.ipynb
│   └── Model Evaluation on splits.ipynb
└── results/
    ├── *.png                  # parity plots, one per configuration / split
    └── baseline_outputs/      # plots for the three descriptor-mixing baselines
```

The notebooks are self-contained: each one re-declares the same utility cell (descriptor computation, splitting, models, training loop), so no local package or import path is needed. They are kept as the record of the runs, not refactored.

## Data

**`Clozapine_dataset.csv`** — 684 rows, experimental.

| Column | Meaning |
|---|---|
| `LogS` | Target: decimal log of solubility (mol fraction basis) |
| `Temperature (K)` | 278.15–333.15 K |
| `Drug` | Always `Clozapine` |
| `Solvent_1`, `Solvent_2` | Solvent names of the binary pair |
| `SMILES_1`, `SMILES_2` | SMILES of the two solvents |
| `comp_1`, `comp_2` | Mole fractions, summing to 100 |

**`combined_file_XTB.csv`** — 6000 rows, COSMO-RS (XTB level) pure-solvent solubilities used only for pretraining. Columns: `Solvent`, `SMILES_1`, `comp_1`, `Temperature`, `MWt (g/mol)`, `x_solute_non_iter`, `x_solute_iter`, `LogS_non_iter`, `LogS`.

## Method

**Descriptors.** 681 Mordred descriptors per solvent. Columns that are non-finite or constant in the training set are dropped (leaving 199 in the no-PCA path, 312 before PCA in the PCA path). A `TrainOnlyStandardizer` is fitted only on training rows, so no validation or test statistics leak into scaling.

**PCA (optional).** One shared PCA (95 % explained variance, 61 components) is fitted on the *unique training solvents only* — each chemical contributes one row rather than one row per data point. In the pretraining variants the PCA basis and standardizer are fitted on the pretraining set and reused unchanged during fine-tuning.

**Splits.** The primary evaluation holds out the **water + 2-propanol** pair entirely (103 rows) as the test set and uses `GroupShuffleSplit` on symmetric solvent-pair keys for validation, so a pair appearing in training cannot reappear in validation as B+A. Alternative splits used in the evaluation notebook: random row split, other unseen pairs (water + NMP, acetone + toluene), temperature interpolation/extrapolation, and composition interpolation/extrapolation.

**Model families.**
- Descriptor mixing baselines: weighted concatenation `[fp1·x1, fp2·x2, T]`, weighted sum `[x1·fp1 + x2·fp2, T]`, and an order-invariant shared-encoder variant that averages both concatenation orders.
- Data-driven: MLP on the same mixed representation, log *S* predicted directly.
- Hybrid: MLP on the mixed representation, van 't Hoff head as above.

**Training.** Adam, learning rate 1e-3, weight decay 1e-5, batch size 32, MSE loss, up to 600 epochs with early stopping (patience 50) on validation loss, seed 42 throughout. The hidden stack is 64 → 32 → 16 for the hybrid backbone, dropout 0.1, Tanh activations.

## Notebooks

| Notebook | What it covers |
|---|---|
| `Baseline (No physics_NO PCA- NO pretrain).ipynb` | Three descriptor-mixing baselines with Mordred descriptors only: weighted concatenation, weighted sum, order-invariant shared encoder |
| `Data-Driven.ipynb` | Ablation without the physics head: PCA × pretraining (no PCA / no pretrain, no PCA + pretrain, PCA, PCA in pretraining space, PCA + pretrain) |
| `Hybrid model.ipynb` | Same ablation grid with the van 't Hoff head, plus a variant where PCA is re-fitted on the experimental solvents |
| `Model Evaluation on splits.ipynb` | The final configuration (PCA + pretraining + physics head) evaluated on 8 different splits |

## Results

All numbers are from the stored notebook outputs (single seed, 42). Test-set R², RMSE and MAE are in log *S* units.

**Ablation, fixed water + 2-propanol holdout (103 test rows):**

| Configuration | R² | RMSE | MAE |
|---|---|---|---|
| No PCA, no pretrain, physics | 0.982 | 0.128 | 0.071 |
| No PCA, pretrain, physics | **0.986** | 0.114 | 0.094 |
| PCA, no pretrain, physics | 0.982 | 0.127 | **0.064** |
| PCA, pretrain, physics | 0.978 | 0.141 | 0.062 |
| PCA, pretrain, physics (PCA re-fitted on experimental solvents) | 0.982 | 0.128 | 0.102 |
| No PCA, no pretrain, no physics | 0.969 | 0.169 | 0.116 |
| No PCA, pretrain, no physics | 0.985 | 0.116 | 0.069 |
| PCA, no pretrain, no physics | 0.958 | 0.196 | 0.126 |
| PCA, no pretrain, no physics (PCA from pretraining space) | 0.957 | 0.198 | 0.120 |
| PCA, pretrain, no physics | 0.975 | 0.150 | 0.094 |
| Baseline, weighted concatenation | 0.988 | 0.104 | 0.061 |
| Baseline, weighted sum | 0.969 | 0.169 | 0.116 |
| Baseline, order-invariant shared encoder | 0.931 | 0.251 | 0.163 |

On this single holdout the configurations are close; the differences between 0.958 and 0.988 are at the level where the split itself, not the architecture, dominates.

**Generalization of the final configuration (PCA + pretraining + physics head):**

| Split | Test rows | R² | RMSE | MAE |
|---|---|---|---|---|
| Random row split | 99 | 0.987 | 0.112 | 0.089 |
| Unseen pair: water + 2-propanol | 99 | 0.892 | 0.221 | 0.180 |
| Unseen pair: water + NMP | 99 | 0.500 | 0.627 | 0.544 |
| Unseen pair: acetone + toluene | 63 | −0.598 | 0.177 | 0.172 |
| Temperature interpolation | 77 | **0.995** | 0.058 | 0.033 |
| Temperature extrapolation | 154 | 0.963 | 0.175 | 0.139 |
| Composition interpolation | 70 | 0.969 | 0.100 | 0.078 |
| Composition extrapolation | 132 | 0.799 | 0.588 | 0.343 |

The pattern is the expected one: interpolation is easy, extrapolation in composition and transfer to solvent pairs the model has never seen are not. The acetone + toluene split has negative R² despite a low RMSE, which happens when the held-out block has little variance of its own — RMSE should be read there, not R².

## Running the notebooks

```bash
pip install -r requirements.txt
jupyter lab            # start from the repository root
```

The notebooks resolve `Clozapine_dataset.csv` and `combined_file_XTB.csv` relative to the working directory, so **start Jupyter from the repository root**, not from `notebooks/`. Each notebook can then be run top to bottom. A CUDA device is used when available, otherwise CPU.

Pretraining variants write checkpoints (`pretrained_cosmo_model.pt`, `pretrained_pca_unique_solvents_cosmo_model.pt`, and a `.pkl` preprocessor) into their output folder; those binaries are git-ignored and regenerated on the first run.

## Notes and limitations

- One drug (clozapine) and one solvent-pair test block plus four alternative splits — this is a study of model structure, not a broad solubility benchmark.
- Single seed per configuration. No error bars are reported, so small gaps between rows in the ablation table should not be over-read.
- Descriptor dropping and standardization depend on the training rows of each split, so configurations are not strictly feature-identical across the grid.
- The physical constraint here is deliberately weak: only the *form* of the temperature dependence is imposed. *A* and *B* are still learned from descriptors.

## Citation

If you use this code or data, please cite the associated master's thesis (University of Padova, Department of Industrial Engineering):

> Solubility Prediction of Clozapine in Binary Solvent Systems with COSMO-RS and Machine Learning, 2026.

## License

Released under the MIT License — see `LICENSE`.
