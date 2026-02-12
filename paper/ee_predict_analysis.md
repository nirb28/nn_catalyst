# Project Analysis: ee-predict — Predicting Enantiomeric Excess via Coordinate Descent-Guided Catalyst Discovery

---

## 1. Project Overview

**ee-predict** is a computational chemistry / cheminformatics project focused on **predicting and optimizing enantiomeric excess (ee)** for asymmetric catalytic reactions — specifically **ruthenium-catalyzed asymmetric transfer hydrogenation (ATH)**. The project is based on the methodology described in [Zahrt et al., *Chem. Eur. J.* 2009, DOI: 10.1002/chem.200802192](https://doi.org/10.1002/chem.200802192), which uses **averaged steric occupancy (ASO)** descriptors computed on 3D molecular grids to represent catalyst structure.

Unlike the other two projects (nn_catalyst and redox_modgen), ee-predict does **not** use neural networks or deep learning. Instead, it employs a suite of **classical statistical and machine learning methods** — linear regression, partial least squares (PLS), random forests, regularized regression (Lasso, Ridge, ElasticNet), factor analysis, and k-nearest neighbors — combined with a novel **coordinate descent (CD) optimization loop** to iteratively discover catalysts with higher enantiomeric excess.

### 1.1 Scientific Context

**Enantiomeric excess (ee)** is a measure of how much one enantiomer (mirror-image molecule) is produced over the other in an asymmetric reaction. It is a critical metric in pharmaceutical chemistry, where the wrong enantiomer can be inactive or harmful. The target variable in this project is **ΔΔG (ddG)**, the difference in free energy between diastereomeric transition states, which directly relates to % ee. Higher ΔΔG means higher selectivity.

The catalyst system studied consists of **Ru-based transfer hydrogenation catalysts** with varying ligand structures. Each catalyst is identified by a label of the form `X_Y_Z` (e.g., `1_1_1`, `22_3_1`), encoding the ligand scaffold, substituent, and conformer.

### 1.2 Key Objectives

1. **Predict ΔΔG** from molecular descriptors (ASO grid features or dimensionality-reduced representations)
2. **Identify the most important spatial regions** around the catalyst that control enantioselectivity (via factor analysis and feature selection)
3. **Discover new high-ee catalysts** using a coordinate descent optimization loop guided by a surrogate regression model

---

## 2. Datasets

The project uses two primary representations of the same underlying catalyst data:

### 2.1 Reduced-Dimension Dataset (`reduced_dim_space_ddG.csv`)

| Property | Value |
|---|---|
| **File** | `data/reduced_dim_space_ddG.csv` |
| **Total rows** | 1,849 catalysts |
| **Non-zero ΔΔG rows** | 318 (17.2% — highly sparse) |
| **Columns** | `Catalyst`, `x`, `y`, `z`, `ddG (% ee)` |
| **Features** | 3 (pre-computed PCA/PLS latent variables: x, y, z) |
| **Target** | `ddG (% ee)` — ΔΔG in kcal/mol, related to enantiomeric excess |
| **Target range** | 0.0 – 1.640 (mean 0.077, std 0.243) |

This is a **pre-reduced** dataset where the original high-dimensional ASO features have already been projected into a 3D latent space. The `x`, `y`, `z` coordinates represent positions in this reduced descriptor space, not physical Cartesian coordinates.

**Critical data characteristic**: 83% of entries have `ddG = 0`, meaning most catalyst/conformer combinations produce racemic (non-selective) mixtures. Only 318 out of 1,849 entries show measurable enantioselectivity. This extreme class imbalance (for the regression task) is a significant challenge.

### 2.2 Full-Dimension Dataset (`large_cat_desc_col_names.csv` / `merged_large_catalyst.csv`)

| Property | Value |
|---|---|
| **File** | `data/large_cat_desc_col_names.csv` |
| **Total rows** | 1,903 catalysts (pre-merge) |
| **Non-zero ΔΔG rows** | 1,903 (all non-zero after filtering in `052024_fa_copy.ipynb`) |
| **Columns** | `Identifier` + 3,973 numeric features + `ddG` |
| **Features** | 3,973 ASO grid descriptors |
| **Merged version** | `merged_large_catalyst.csv` (1,850 rows × 3,976 cols after inner join with `Data.csv`) |

This is the **raw high-dimensional** dataset. Each of the 3,973 features corresponds to a **gridpoint** in 3D space around the catalyst, with the feature value representing the **averaged steric occupancy (ASO)** at that point — a measure of how often that spatial region is occupied by the catalyst's atoms across conformational sampling. The grid has **10,318 total gridpoints** before feature reduction.

### 2.3 Original Source Data (`data/original-datasets/`)

| File | Description |
|---|---|
| `Data.csv` | 1,850 catalysts with columns: `Catalyst`, `x`, `y`, `z`, `ddG (% ee)` |
| `cat_desc.csv` | 1,903 catalysts × 3,974 ASO descriptor columns (no headers) |

The `merge_large_datasets.ipynb` notebook merges these two files on the catalyst identifier to create the combined `merged_large_catalyst.csv`.

---

## 3. Feature Engineering & Dimensionality Reduction Pipeline

The project implements a sophisticated multi-stage feature reduction pipeline, going from 10,318 raw ASO gridpoints down to as few as 3–4 latent variables. This is documented primarily in `052024_fa_copy.ipynb` and `052124_feature_selection.ipynb`.

### 3.1 Stage 1: Variance Thresholding

**Raw ASO grid**: 10,318 gridpoints per catalyst (from `ASO.csv`)

```
Variance threshold = 0.03 (tuned)
10,318 → 330 gridpoints retained
```

Most gridpoints have near-zero variance across catalysts (i.e., all catalysts look the same in those regions). Only 330 gridpoints (3.2%) show meaningful variation — these are the regions where ligand structure actually differs between catalysts.

Alternative thresholds explored:
- `var=0.005`: 986 retained
- `var=0.01`: 763 retained
- `var=0.03`: 330 retained (selected)

### 3.2 Stage 2: Correlation Filtering

```
Correlation threshold = 0.90
330 → 159 gridpoints retained
```

Neighboring gridpoints in 3D space are highly correlated (if one is occupied, its neighbors likely are too). Removing one of each highly-correlated pair eliminates redundancy. Alternative thresholds:
- `corr=0.90`: 159 retained (selected)
- `corr=0.95`: 237 retained

### 3.3 Stage 3: Factor Analysis (FA)

**Statistical suitability tests** (on the 159 retained features):
- **Bartlett's test**: χ² = 120,385.7, p = 0.0 → highly significant, confirming correlation structure exists
- **KMO (Kaiser-Meyer-Olkin)**: 0.889 → "meritorious" suitability for factor analysis

**Factor extraction**:
```
159 features → 16 factors (eigenvalue > 1 criterion)
Rotation: Varimax (orthogonal)
```

Each factor represents a **spatially coherent region** around the catalyst where steric effects co-vary. The 16 factors can be mapped back to 3D space and visualized as colored clusters of gridpoints around the catalyst scaffold (conserved O, N atoms and chiral methyl group).

### 3.4 Stage 4: Recursive Feature Elimination (RFE)

After factor analysis, RFE with cross-validation further reduces the factors:
```
16 factors → 4 top factors (factors 0, 1, 7, 9)
```

These 4 factors correspond to 63 gridpoints in 3D space, representing the **most enantioselectivity-determining spatial regions** around the catalyst. The OLS coefficients for these factors are:
```
Factor 0: -0.0673  (minor negative influence)
Factor 1: +0.1745  (positive influence)
Factor 7: -0.1415  (negative influence)
Factor 9: -0.3515  (strongest influence, negative)
```

**Interpretation**: Factor 9 has the largest magnitude coefficient, meaning the steric occupancy in that spatial region has the strongest effect on enantioselectivity. The signs indicate whether steric bulk in that region helps or hinders selectivity.

### 3.5 Alternative Pipeline: High-Correlation Feature Selection

In `large/eda.ipynb`, an alternative feature selection approach is used:
1. Compute Pearson correlation of each feature with the target `ddG`
2. Retain features with |correlation| > 0.5
3. Result: **62 features** (stored in `models/high_corr_cols.txt`)

PCA on these 62 features shows that **4 principal components** explain **90% of variance** (first PC alone explains 75%).

---

## 4. Modeling Approaches

### 4.1 Linear Regression (OLS)

**On reduced 3D space** (`linear-regression.ipynb`):
| Metric | Value |
|---|---|
| Train/test split | 70/30, random_state=101 |
| MSE | 0.1069 |
| MAE | 0.2756 |
| Intercept | 0.4323 |
| Coefficients | x: -0.00123, y: -0.00341, z: -0.00200 |

Very weak coefficients — the 3D reduced space retains limited predictive information for a linear model.

**On full feature space (3,974 features)**:
| Metric | Value |
|---|---|
| Train R² | 0.886 |
| Test R² | **-4.07 × 10²⁰** (catastrophic overfitting) |

With n=255 and p=3,974 (p >> n), OLS perfectly memorizes the training data but produces astronomically bad test predictions. This is a textbook demonstration of why regularization or dimensionality reduction is essential when p >> n.

### 4.2 Partial Least Squares (PLS) Regression

PLS is the **primary modeling method** throughout the project, used both as a standalone predictor and as the surrogate model in the coordinate descent optimization loop.

**On reduced 3D space**:
| Metric | Value |
|---|---|
| n_components | 2 |
| MSE | 0.1063 |
| MAE | 0.2750 |

Marginally better than OLS on the reduced space.

**On 62 high-correlation features** (`large/eda.ipynb`):
| Metric | Value |
|---|---|
| n_components | 4 (from PCA 90% variance criterion) |
| R² | 0.547 |
| RMSE | 0.220 |

Moderate performance. The saved model (`models/pls_large.joblib`) uses this configuration and is loaded by the coordinate descent notebooks.

**PLS with all features (implicit LVs)** (`052124_feature_selection.ipynb`):
| Metric | Value |
|---|---|
| n_components | 18 |
| 4-fold CV R² | **0.9998** |

This suspiciously high score (near-perfect) suggests **data leakage** or severe overfitting in the PLS with 18 components on the full feature set. This is further confirmed when PLS-RF achieves 0.9896 on the same setup.

### 4.3 Random Forest

Random Forest consistently achieves the **best predictive performance** across all feature representations:

| Feature Space | 4-fold CV R² |
|---|---|
| Raw 159 features | 0.8644 |
| FA 16 latent variables | 0.7924 (ASO grid) / 0.8704 (large dataset) |
| FA-RFE 4 factors | 0.9193 (large dataset) / 0.7980 (ASO grid) |
| PLS 16 latent variables | 0.8646 |

**Key finding**: RF on FA-RFE features (4 factors from the large dataset) achieves **0.9193 R²** — the best reliable cross-validated score in the project.

### 4.4 Regularized Linear Models

On FA-derived 16 latent variables (`052024_fa_copy.ipynb`, ASO grid data):

| Model | 4-fold CV R² |
|---|---|
| OLS | 0.7441 |
| LassoCV | 0.7966 |
| RidgeCV | 0.7568 |
| ElasticNetCV | 0.7888 |
| Random Forest | 0.7924 |

After RFE (4 factors):
| Model | 4-fold CV R² |
|---|---|
| OLS | 0.8131 |
| LassoCV | 0.8133 |
| Random Forest | 0.7980 |

**Notable**: After RFE, OLS and Lasso perform **better** than RF, and nearly as well as RF on the full factor set. This suggests the 4-factor representation captures the essential linear structure of the problem.

On FA-derived 18 latent variables (`052124_feature_selection.ipynb`, large dataset):

| Model | 4-fold CV R² |
|---|---|
| OLS | 0.6224 |
| LassoCV | 0.6270 |
| RidgeCV | 0.6224 |
| ElasticNetCV | 0.6242 |
| Random Forest | **0.8704** |

After RFE (3 factors):
| Model | 4-fold CV R² |
|---|---|
| OLS | 0.6341 |
| Random Forest | **0.9193** |
| LassoCV | 0.6341 |

Here RF dramatically outperforms linear models, suggesting the relationship in the large-dataset representation is more non-linear.

### 4.5 Lasso Regression (Full Feature Space)

| Metric | Value |
|---|---|
| Features | 3,974 (all) |
| alpha | 0.0001 |
| R² | 0.579 |
| RMSE | 0.303 |

Lasso naturally performs feature selection via L1 regularization, avoiding the catastrophic overfitting of OLS.

### 4.6 K-Nearest Neighbors (Classification)

In `knn.ipynb`, KNN is used for **catalyst classification** (not regression):
- **Task**: Given 3D coordinates (x, y, z), predict the catalyst identity
- **k**: 3
- **Split**: 80/20, random_state=42
- This is used as a **lookup mechanism** in the coordinate descent loop, not as a primary predictive model

### 4.7 Forward Stepwise Selection (Cp Criterion)

In `large/eda.ipynb`, the ISLP package is used for forward stepwise variable selection:
- **Using MSE criterion**: Selected 5 features (`3954`, `3955`, `3964`, `3968`, `3970`)
- **Using negative Cp criterion**: Selected 7 features (`2595`, `3489`, `3955`, `3960`, `3964`, `3968`, `3970`)

These selected features are all in the 2500–3970 range, corresponding to gridpoints in specific spatial regions around the catalyst. Notably, features `3968` and `3970` appear in both selections and in the high-correlation set.

---

## 5. Coordinate Descent Optimization Loop

The most novel aspect of the project is the **coordinate descent (CD) optimization algorithm** for discovering catalysts with higher enantiomeric excess. This is implemented across several notebooks with increasing sophistication.

### 5.1 Core Algorithm (`cd_v2.ipynb`)

The algorithm operates in a loop:

```
1. START: Select initial set of catalysts (random sample of size 3)
2. FIT: Train a PLS regression model on current catalyst set
3. OPTIMIZE: Apply coordinate descent to each catalyst's descriptor vector
   - For each feature dimension:
     - Try step +δ: if predicted ΔΔG improves, keep it
     - Else try step -δ: if predicted ΔΔG improves, keep it
     - Else revert to original value
4. LOOKUP: Use KNN to find the nearest real catalyst to each optimized point
5. ADD: Add the found neighbors to the catalyst set
6. REFIT KNN: Remove already-found catalysts from the KNN pool
7. CHECK: If max(ΔΔG) ≥ top-5 threshold, exit; else goto 2
```

### 5.2 Tunable Parameters (`cd_v2.ipynb`)

| Parameter | Value | Description |
|---|---|---|
| `MODEL_REFRESH_STRATEGY` | `RETRAIN` | Retrain PLS after every loop iteration |
| `OPTIMIZE_STRATEGY` | `NEWLY_ADDED` | Only optimize the most recently added catalysts |
| `MODEL_FIT_USE_COLS` | `models/high_corr_cols.txt` | Use 62 high-correlation features |
| `DEFAULT_PCA_DIMENSION` | 4 | PLS n_components |
| `START_SAMPLE_SIZE` | 3 | Initial random catalyst count |
| `CD_STEP_SIZE` | 0.001 | Coordinate descent step size |
| `CD_STEP_SIZE_MULTIPLIER` | 1.5 | Increase step size when stuck |
| `CD_ITERATIONS_PER_STEP` | 2 | Inner CD iterations per feature |
| `CD_ITERATIONS` | 10 | Outer CD iterations per catalyst |
| `distance_threshold` | 2.75 | For initialization set diversity |
| `max_cycles` | 50 | Maximum outer loops |

### 5.3 Model Refresh Strategies

Two strategies were implemented:
1. **RETRAIN**: Rebuild the PLS model from scratch using all currently discovered catalysts. Adapts the response surface as more data is added.
2. **PRETRAIN**: Use a pre-trained PLS model (`pls_large.joblib`) throughout. Faster but cannot adapt to new information.

### 5.4 Optimization Strategies

Two strategies for which catalysts to optimize:
1. **NEWLY_ADDED**: Only optimize the most recently added catalysts (faster, focuses on frontier)
2. **ALL**: Re-optimize all catalysts in the current set (more thorough but slower)

### 5.5 Adaptive Step Size

When the maximum ΔΔG doesn't improve between iterations, the step size is multiplied by 1.5× to escape local plateaus. This allows the optimizer to take larger jumps in descriptor space to explore more distant catalyst configurations.

### 5.6 Iterative Version (`cd_iterative.ipynb`)

An earlier, simpler version processes catalysts in fixed chunks of 3, iterating through the entire dataset sequentially:
- Starts with first 3 catalysts
- Optimizes using CD, finds KNN neighbors
- Adds neighbors and their ΔΔG values to the PLS training set
- Refreshes the PLS model with growing data
- Repeats for the next chunk

This version does **not** have the adaptive step size or convergence check. The output shows the model being refreshed from 3 rows up to 321 rows.

### 5.7 Multi-Cycle Evaluation (`cd_v2.ipynb`)

The notebook includes a multi-cycle evaluation loop that runs the full CD algorithm up to 50 times with different random initializations, tracking how many catalysts need to be "synthesized" (discovered) before reaching the top-5 ΔΔG values. This provides a statistical measure of the algorithm's efficiency.

---

## 6. Factor Analysis Interpretation

### 6.1 Physical Meaning of Factors

The factor analysis in `052024_fa_copy.ipynb` provides the most scientifically valuable result: **mapping statistical factors back to 3D space** around the catalyst.

Each of the 159 retained gridpoints is assigned to the factor on which it has the highest loading. The top 4 factors (selected by RFE) correspond to 63 gridpoints that can be visualized in 3D:

- **Factor 0** (red): Gridpoints in a specific spatial region around the catalyst scaffold
- **Factor 1** (purple): Region near the conserved N atom
- **Factor 7** (green): Region near the chiral methyl group
- **Factor 9** (blue): Region with the strongest influence on selectivity

The 3D plot overlays these colored gridpoint clusters with the conserved O and N atoms, scaffold carbons, and chiral methyl group, providing a **visual "steric map"** of where catalyst modification matters most for enantioselectivity.

### 6.2 OLS Coefficients on Top Factors

```
Factor 0: -0.067 (minor negative)
Factor 1: +0.174 (positive — steric bulk HELPS selectivity)
Factor 7: -0.141 (negative — steric bulk HURTS selectivity)
Factor 9: -0.352 (strongest — steric bulk strongly HURTS selectivity)
```

**Chemical interpretation**: The most important factor (9) has a large negative coefficient, meaning that catalysts with more steric bulk in that spatial region tend to have **lower** enantioselectivity. Factor 1 is the only positive factor — steric bulk there **improves** selectivity, possibly by constraining the substrate approach angle.

---

## 7. Mordred Descriptors (Attempted Extension)

The `Mordred_Descriptors.ipynb` notebook attempted to use **Mordred molecular descriptors** (calculated from RDKit molecular objects) as an alternative representation to ASO descriptors. However, this notebook was run in Google Colab and failed with a `NameError` (missing `os` import), indicating this was an incomplete/exploratory effort that was not pursued further. The project ultimately stayed with the ASO-based descriptors.

---

## 8. Development Environment & Tools

| Aspect | Details |
|---|---|
| **Languages** | Python 3.10–3.11 |
| **Environments** | Local (Windows/Mac), Google Colab |
| **Key libraries** | scikit-learn, pandas, numpy, matplotlib, seaborn, factor_analyzer, cclib, ccheminfolib, molli, ISLP |
| **Colab integration** | Google Drive mounting for persistent storage |
| **Version control** | Git (GitHub: `nirb28/ee-predict`) |
| **IDE** | PyCharm (based on `.idea/` directory) |
| **Saved models** | `pls.joblib`, `models/pls_large.joblib` |
| **Collaboration** | Original code by AFZ (Aug 2023); edits by DAP (May 2024); assistance from Claude 3 Opus |

---

## 9. Results Summary

### 9.1 Best Predictive Models

| Rank | Model | Feature Space | CV R² | Notes |
|---|---|---|---|---|
| 1 | PLS (18 comp) | 171 full features | 0.9998 | Likely overfit/data leakage |
| 2 | RF + FA-RFE | 3 factors (large dataset) | 0.9193 | Best reliable score |
| 3 | RF + PLS LVs | 16 PLS components | 0.8646–0.8896 | Consistent |
| 4 | RF + FA LVs | 16–18 factors | 0.7924–0.8704 | Depends on dataset |
| 5 | OLS/Lasso + FA-RFE | 4 factors (ASO grid) | 0.8131–0.8133 | Best linear on ASO |
| 6 | Lasso (full) | 3,974 features | 0.579 | Regularization helps vs OLS |
| 7 | PLS (4 comp) | 62 high-corr features | 0.547 | Moderate |
| 8 | OLS | 3D reduced space | ~0.25 (implied) | Very weak |

### 9.2 Coordinate Descent Optimization

The CD algorithm demonstrates proof-of-concept for **model-guided catalyst discovery**:
- Starting from 3 random catalysts, the algorithm iteratively discovers catalysts with increasing ΔΔG
- The target is to reach the top-5 ΔΔG values in the dataset (max = 1.640)
- The `RETRAIN` strategy adapts the surrogate model as new data is discovered
- Adaptive step size (1.5× multiplier) helps escape plateaus

However, the main loop in `cd_v2.ipynb` was interrupted (`KeyboardInterrupt`), suggesting convergence was slow. This is likely due to:
1. The high dimensionality of the search space (62 features)
2. The small step size (0.001)
3. The need to re-fit KNN and PLS models at each iteration

### 9.3 Key Scientific Insights

1. **Sparse enantioselectivity**: Only 17% of catalyst entries show non-zero ΔΔG, making this an extremely challenging regression problem
2. **p >> n problem**: With 3,973 features and ~300 usable samples, standard regression fails catastrophically without regularization or dimensionality reduction
3. **4 key spatial factors** control most of the enantioselectivity, as identified by factor analysis + RFE
4. **Steric occupancy in Factor 9's region** is the single most important predictor (coefficient = -0.352)
5. **Random Forest** consistently outperforms linear models, suggesting non-linear structure in the steric-selectivity relationship
6. **ASO descriptors** provide a physically interpretable molecular representation that can be mapped back to 3D space

---

## 10. Comparative Analysis with nn_catalyst and redox_modgen

| Aspect | ee-predict | nn_catalyst | redox_modgen |
|---|---|---|---|
| **Domain** | Asymmetric catalysis (ee) | Catalyst property prediction | Redox potential prediction |
| **Target** | ΔΔG (1 target) | 29–30 properties | 4 properties |
| **Descriptors** | ASO grid (10K+) | Mordred (1,613) | Mordred + MACE (847) |
| **Models** | PLS, RF, OLS, Lasso, Ridge, ElasticNet | FNN with skip connections | FNN, RF, XGBoost |
| **Best method** | RF on FA-RFE factors | FNN | FNN |
| **Best R²** | ~0.92 (RF+FA-RFE) | ~0.99 (val) | ~0.96 (test, ion_pot) |
| **Novel contribution** | CD optimization loop | Skip connections in FNN | Dynamic FNN architecture |
| **Dataset size** | 319 (non-zero) | ~300 catalysts | 26,156 molecules |
| **Deep learning** | No | Yes (PyTorch) | Yes (PyTorch Lightning) |
| **Cross-validation** | 4-fold CV | Single split | 20-partition CV |
| **Interpretability** | High (factor → 3D space) | Low (black box FNN) | Low (black box FNN) |

### 10.1 Evolution Across Projects

The three projects represent an evolution in the user's approach to molecular property prediction:

1. **ee-predict** (earliest): Classical statistical methods with strong emphasis on **interpretability** — factor analysis maps statistical patterns back to physical 3D space. Limited by small dataset size (319 usable samples).

2. **nn_catalyst** (middle): Transition to **deep learning** with feedforward neural networks. Larger feature space handled by the FNN's ability to learn representations. Less interpretable but better raw performance.

3. **redox_modgen** (latest): Full **production-grade ML pipeline** with proper cross-validation, multiple model comparison (FNN vs RF vs XGBoost), package structure, and systematic hyperparameter tuning. The most rigorous and scalable approach.

---

## 11. Conclusion

**ee-predict** demonstrates a sophisticated approach to a challenging computational chemistry problem: predicting and optimizing enantioselectivity from high-dimensional steric descriptors. The project's key strengths are:

1. **Methodological rigor**: Proper statistical tests (Bartlett, KMO) before factor analysis, variance and correlation filtering, cross-validated model comparison
2. **Physical interpretability**: Factor analysis maps statistical patterns to 3D spatial regions, providing chemical insight into what controls selectivity
3. **Novel optimization**: The coordinate descent loop with KNN lookup provides a framework for iterative catalyst discovery
4. **Comprehensive model comparison**: Side-by-side evaluation of 6+ regression methods across multiple feature representations

The main limitations are:
1. **Small dataset**: Only 319 usable samples with extreme sparsity (83% zero ΔΔG)
2. **Slow CD convergence**: The optimization loop was interrupted, suggesting scalability issues
3. **Potential overfitting**: Some PLS configurations show suspiciously high scores (R² > 0.99)
4. **Incomplete Mordred extension**: The attempt to use alternative molecular descriptors was not completed

The project provides a strong foundation for catalyst design using steric descriptors, with the factor analysis results offering actionable chemical insights about which spatial regions to modify for improved enantioselectivity.
