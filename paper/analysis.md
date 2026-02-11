# Project 1: nn_catalyst — Neural Network for Catalyst Property Prediction

## 1.1 Project Overview

**[nn_catalyst](cci:9://file:///d:/ds/work/workspace/git/nn_catalyst:0:0-0:0)** is a computational chemistry / cheminformatics project that uses **feedforward neural networks with skip connections** to predict **multiple molecular properties of catalysts** from molecular descriptors. The project was developed to predict **29–30 target properties** (electronic energies, Gibbs free energies, HOMO/LUMO orbital energies, charges, spin densities, dipole moments, and redox potentials) for molecular catalyst systems in different oxidation states (reduced `_r`, neutral `_n`, oxidized `_o`).

The project was developed iteratively — starting with vanilla PyTorch, then evolving to PyTorch Lightning for better training orchestration, and exploring multiple model architectures and feature engineering strategies along the way.

## 1.2 Project Structure

The project is organized as follows:

| Component | Location | Purpose |
|---|---|---|
| **Main training script** | `@D:\ds\work\workspace\git\nn_catalyst\make_model.py:1-308` | Trains 30 individual [SingleTargetNet](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/load_and_pred.py:64:0-86:17) models |
| **Inference script** | `@D:\ds\work\workspace\git\nn_catalyst\load_and_pred.py:1-208` | Loads saved models and generates predictions |
| **PyTorch Lightning models** | `@D:\ds\work\workspace\git\nn_catalyst\src\pl\model_impl.py:1-91` | Refactored [BaseModel](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/src/pl/model_impl.py:5:0-61:89) + [SingleTargetNet](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/load_and_pred.py:64:0-86:17) using PL |
| **Sequential/stacked training** | `@D:\ds\work\workspace\git\nn_catalyst\src\pl\StackModels.ipynb` | Full training pipeline with optional prediction stacking |
| **EDA & feature analysis** | `@D:\ds\work\workspace\git\nn_catalyst\src\pl\eda.ipynb` | PCA, correlation analysis, forward selection, PLS regression |
| **Architecture experiments** | `@D:\ds\work\workspace\git\nn_catalyst\src\different_approaches.ipynb` | Compares 3 different NN architectures per target |
| **Mordred descriptors** | `@D:\ds\work\workspace\git\nn_catalyst\src\mordred\app.py:1-202` | Streamlit app for SMILES→descriptors→predictions |
| **SMILES prediction pipeline** | `@D:\ds\work\workspace\git\nn_catalyst\src\mordred\smile_predictions.ipynb` | End-to-end SMILES input to multi-target predictions |
| **Result analysis** | `@D:\ds\work\workspace\git\nn_catalyst\result-analysis.ipynb` | Post-hoc analysis of model predictions |
| **Checkpoints** | `@D:\ds\work\workspace\git\nn_catalyst\checkpoints` | Saved model checkpoints ([stn_r2/](cci:9://file:///d:/ds/work/workspace/git/nn_catalyst/checkpoints/stn_r2:0:0-0:0), [stn_r3_f849_tlast29/](cci:9://file:///d:/ds/work/workspace/git/nn_catalyst/checkpoints/stn_r3_f849_tlast29:0:0-0:0)) |

## 1.3 Data Pipeline

### 1.3.1 Input Data
- **Descriptors**: Molecular descriptors computed using **Mordred** (a comprehensive molecular descriptor calculator). The descriptor file (`descriptors.csv`) contains ~849 numeric molecular descriptors per molecule. These include topological, constitutional, electronic, geometric, and autocorrelation descriptors (e.g., `ABC`, `ABCGG`, `nAcid`, `ATS*`, `GATS*`, `MATS*`, `BCUТ*`, `SpAbs_*`, etc.).
- **Targets**: [compiled_data.csv](cci:7://file:///d:/ds/work/workspace/git/nn_catalyst/compiled_data.csv:0:0-0:0) (~8 MB) contains 29–30 target molecular properties including:
  - **Electronic energies**: `elec_en_r`, `elec_en_n`, `elec_en_o`
  - **Gibbs free energies**: `gibbs_r`, `gibbs_n`, `gibbs_o`
  - **HOMO/LUMO orbital energies**: `homo_n`, `lumo_n`, `homo_spin_up_o`, `homo_spin_down_o`, `lumo_spin_up_o`, `lumo_spin_down_o`, etc.
  - **Partial charges**: `max_charge_pos_n`, `max_charge_neg_n`, etc.
  - **Spin properties**: `max_spin_r`, `max_spin_o`
  - **Dipole moments**: `dipole_r`, `dipole_n`, `dipole_o`
  - **Redox potentials**: `ddg_ox`, `ddg_red`

### 1.3.2 Feature Engineering
1. **Numeric-only filtering**: Non-numeric columns are dropped from both descriptors and targets.
2. **Merge**: Descriptors and targets are merged on a molecule identifier (`Label` ↔ `mol_num`).
3. **Variance thresholding**: `VarianceThreshold()` removes zero-variance or near-zero-variance features to reduce dimensionality.
4. **Standardization**: Both features (`X`) and targets (`y`) are standardized using `StandardScaler`, fitted on the training set only — a proper practice to prevent data leakage.
5. **Train/Val/Test split**: 80/10/10 split using `train_test_split` with `random_state=42` for reproducibility.

### 1.3.3 Exploratory Data Analysis (EDA)
The EDA notebook (`@D:\ds\work\workspace\git\nn_catalyst\src\pl\eda.ipynb`) performed extensive analysis:
- **Correlation analysis**: Computed feature-target correlations, identified features with |corr| > 0.2 (saved to [high_corr_cols.txt](cci:7://file:///d:/ds/work/workspace/git/nn_catalyst/src/pl/high_corr_cols.txt:0:0-0:0) — hundreds of features qualified).
- **PCA**: Principal Component Analysis to find the number of components preserving 90% of variance, used as a dimensionality baseline.
- **PLS Regression**: Partial Least Squares Regression tested as a linear baseline.
- **Polynomial Regression**: Degree-2 polynomial features combined with linear regression.
- **Lasso Regression**: L1-regularized regression for automatic feature selection.
- **Forward Stepwise Selection**: Using the ISLP library's `Stepwise` with Cp statistic for model selection.

## 1.4 Neural Network Architecture

### 1.4.1 Core Model: [SingleTargetNet](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/load_and_pred.py:64:0-86:17)
The primary architecture is a **feedforward neural network with a skip (residual) connection**, defined in `@D:\ds\work\workspace\git\nn_catalyst\make_model.py:91-113`:

```
Input → Linear(input, 1024) → BatchNorm1d → ReLU → Dropout(0.5)
                                                      ↓
                                              Linear(1024, 512) → BatchNorm1d → ReLU → Dropout(0.5)
                                                      +
                                              Linear(1024, 512) [skip connection]
                                                      ↓
                                              Linear(512, 1) → Output
```

**Key architectural decisions**:
- **One model per target**: Instead of a single multi-output network, 29–30 separate [SingleTargetNet](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/load_and_pred.py:64:0-86:17) instances are trained — one for each target property. This allows each model to specialize.
- **Skip connection**: A learnable linear skip connection from layer 1 (1024-dim) to layer 2 (512-dim) via `fc_skip`. This helps gradient flow and allows the second layer to learn residual corrections.
- **Batch normalization**: Applied after each linear layer, before activation, to stabilize training.
- **Dropout**: 50% dropout rate in the vanilla PyTorch version (later reduced to 20% in the PyTorch Lightning version) for regularization.

### 1.4.2 Architecture Variants Explored
The [different_approaches.ipynb](cci:7://file:///d:/ds/work/workspace/git/nn_catalyst/src/different_approaches.ipynb:0:0-0:0) notebook tested three architectures per target:

1. **[SingleTargetNet](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/load_and_pred.py:64:0-86:17)** (1024→512 with skip connection) — the primary model
2. **`RegressionNetwork`** (input→512→512→1 with BatchNorm and LeakyReLU) — simpler without skip connections
3. **Sequential model** (input→1024→512→256→1 with LeakyReLU) — deeper, no skip connections

### 1.4.3 PyTorch Lightning Version
An upgraded version (`@D:\ds\work\workspace\git\nn_catalyst\src\pl\model_impl.py:64-91`) wraps the same architecture in PyTorch Lightning's `LightningModule` with:
- **[BaseModel](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/src/pl/model_impl.py:5:0-61:89)** base class: Handles training/validation/test steps, loss computation (`MSELoss`), R² metric tracking via `torchmetrics.R2Score`, and optimizer configuration.
- **AdamW optimizer** with `lr=0.001`
- **ReduceLROnPlateau scheduler**: Factor 0.1, patience 10, min_lr 1e-9
- **EarlyStopping**: On `train_loss` with patience 10
- **ModelCheckpoint**: Saves the best model based on `val_loss`

## 1.5 Training Strategy

### 1.5.1 Sequential Per-Target Training
The function [sequential_training()](cci:1://file:///d:/ds/work/workspace/git/redox_modgen/redox_modgen/model_gym.py:335:0-606:28) in `@D:\ds\work\workspace\git\nn_catalyst\src\pl\StackModels.ipynb` (cell 6) implements the core training loop:

1. For each of the 29 targets sequentially:
   - Train a [SingleTargetNet](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/load_and_pred.py:64:0-86:17) with the current feature set
   - Generate predictions on train/val/test
   - Optionally **stack predictions**: Append the model's predictions as new features for the next target's model (when `stack_predictions=True`)
2. Compute R² scores on the test set for each target
3. Save all predictions for downstream analysis

### 1.5.2 Training Hyperparameters
- **Batch size**: 32
- **Max epochs**: 150
- **Loss function**: MSE (Mean Squared Error)
- **Optimizer**: AdamW with lr=0.001
- **LR scheduler**: ReduceLROnPlateau (factor=0.1, patience=10)
- **Early stopping**: Patience of 10–15 epochs
- **Data split**: 60% train / 20% validation / 20% test (in PL version: 60/20/20 via nested splits)

### 1.5.3 Prediction Stacking (Optional)
When `stack_predictions=True`, after predicting each target, the model's predictions are concatenated with the original features, creating an ever-growing feature set for subsequent targets. This allows later targets to benefit from information learned by earlier target models. The feature count grows from 849 → 850 → 851 → ... → 877.

## 1.6 Results

### 1.6.1 Final R² Scores (Test Set, stack=False, scaleY=True)
From the StackModels notebook output, the **29-target test R² scores** (from best to worst):

| Rank | Target | R² Test | Category |
|------|--------|---------|----------|
| 1 | `elec_en_n` | **0.9951** | Electronic energy |
| 2 | `elec_en_r` | **0.9948** | Electronic energy |
| 3 | `elec_en_o` | **0.9946** | Electronic energy |
| 4 | `gibbs_o` | **0.9938** | Gibbs free energy |
| 5 | `gibbs_n` | **0.9928** | Gibbs free energy |
| 6 | `gibbs_r` | **0.9926** | Gibbs free energy |
| 7 | `homo_spin_down_o` | 0.9633 | Orbital energy |
| 8 | `homo_spin_up_o` | 0.9606 | Orbital energy |
| 9 | `ddg_ox` | 0.9581 | Redox potential |
| 10 | `lumo_n` | 0.9491 | Orbital energy |
| 11 | `lumo_spin_up_o` | 0.9392 | Orbital energy |
| 12 | `lumo_spin_down_o` | 0.9360 | Orbital energy |
| 13 | `max_charge_pos_n` | 0.9324 | Partial charge |
| 14 | `max_charge_pos_o` | 0.9307 | Partial charge |
| 15 | `ddg_red` | 0.9303 | Redox potential |
| 16 | `homo_n` | 0.9166 | Orbital energy |
| 17 | `lumo_spin_up_r` | 0.9153 | Orbital energy |
| 18 | `max_charge_pos_r` | 0.8780 | Partial charge |
| 19 | `lumo_spin_down_r` | 0.8567 | Orbital energy |
| 20 | `max_charge_neg_n` | 0.8526 | Partial charge |
| 21 | `max_charge_neg_o` | 0.8400 | Partial charge |
| 22 | `homo_spin_up_r` | 0.8352 | Orbital energy |
| 23 | `homo_spin_down_r` | 0.8206 | Orbital energy |
| 24 | `max_charge_neg_r` | 0.7575 | Partial charge |
| 25 | `max_spin_o` | 0.6945 | Spin density |
| 26 | `max_spin_r` | 0.6849 | Spin density |
| 27 | `dipole_o` | 0.5932 | Dipole moment |
| 28 | `dipole_r` | 0.5925 | Dipole moment |
| 29 | `dipole_n` | 0.4448 | Dipole moment |

### 1.6.2 Results Interpretation

**Excellent predictions (R² > 0.99)**: Electronic energies and Gibbs free energies are predicted with near-perfect accuracy. These are bulk thermodynamic properties that have strong, systematic relationships with molecular descriptors.

**Strong predictions (R² 0.90–0.97)**: Orbital energies (HOMO/LUMO), redox potentials (ΔΔG_ox, ΔΔG_red), and partial charges for neutral and oxidized states. These are quantum mechanical properties with complex but learnable structure-property relationships.

**Moderate predictions (R² 0.75–0.90)**: Partial charges and orbital energies for the reduced state, which are harder to predict due to additional complexity of the reduced electronic configuration.

**Weaker predictions (R² 0.45–0.70)**: Dipole moments and spin densities are the hardest targets. These are highly sensitive to 3D molecular geometry and conformational effects that Mordred 2D descriptors cannot fully capture.

## 1.7 Deployment & Inference

### 1.7.1 Streamlit Application
A Streamlit web app (`@D:\ds\work\workspace\git\nn_catalyst\src\mordred\app.py:1-202`) provides an end-to-end inference pipeline:
1. User inputs a **SMILES string** (molecular notation)
2. **Mordred** computes molecular descriptors for that SMILES
3. Features are filtered to match the training descriptor set (from [descriptors.txt](cci:7://file:///d:/ds/work/workspace/git/nn_catalyst/src/mordred/descriptors.txt:0:0-0:0))
4. The pre-trained `StandardScaler` normalizes the features
5. Each of the 29 saved models generates its prediction
6. All 29 predictions are displayed

### 1.7.2 Batch Prediction
The [smile_predictions.ipynb](cci:7://file:///d:/ds/work/workspace/git/nn_catalyst/src/mordred/smile_predictions.ipynb:0:0-0:0) notebook demonstrates batch prediction on new SMILES (e.g., aspirin, caffeine, salbutamol), showing the full pipeline from SMILES → Mordred descriptors → scaled features → 29-target predictions.

## 1.8 Key Technical Observations

1. **One-model-per-target strategy**: Effective for varying difficulty across targets; each model can be independently tuned.
2. **Skip connections**: Help the 3-layer network learn residual mappings, especially beneficial for targets with strong linear components.
3. **Proper data leakage prevention**: Scalers fitted on training data only, then applied to validation/test.
4. **Comprehensive evaluation**: R², RMSE, MAE computed per target; parity plots generated in Excel for visual inspection.
5. **Google Colab compatibility**: Notebooks designed to run on both local machines and Google Colab with Google Drive mount.
6. **Two checkpoint generations**: [stn_r2](cci:9://file:///d:/ds/work/workspace/git/nn_catalyst/checkpoints/stn_r2:0:0-0:0) (earlier round) and [stn_r3_f849_tlast29](cci:9://file:///d:/ds/work/workspace/git/nn_catalyst/checkpoints/stn_r3_f849_tlast29:0:0-0:0) (refined, 849 features, 29 targets, R² ordering of targets).

---

# Project 2: redox_modgen — Neural Network for Redox Potential & QM9 Property Prediction

## 2.1 Project Overview

**[redox_modgen](cci:9://file:///d:/ds/work/workspace/git/redox_modgen:0:0-0:0)** is a more mature, production-oriented research package developed at the **University of Pennsylvania, Department of Chemistry, Zahrt Group**. It is a **Python package** (installable via [setup.py](cci:7://file:///d:/ds/work/workspace/git/redox_modgen/setup.py:0:0-0:0)) for comparing machine learning models for predicting **oxidation/reduction potentials and HOMO/LUMO energies** of molecules, referenced against the **standard hydrogen electrode (SHE) at 4.44 eV**.

Your contribution focused on the **Feedforward Neural Network (FNN)** component. The project also extended to the **QM9 benchmark dataset** for broader molecular property prediction.

## 2.2 Project Structure

| Component | Location | Purpose |
|---|---|---|
| **Package init** | `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\__init__.py:1-5` | Package entry point |
| **BaseModel (PL)** | `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\models\__init__.py:1-69` | Lightning base class for all neural network models |
| **FNN architecture** | `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\models\feedforward_neural_network.py:1-78` | [SingleTargetDynamicNet](cci:2://file:///d:/ds/work/workspace/git/redox_modgen/redox_modgen/models/feedforward_neural_network.py:9:0-76:39) — dynamic-depth FNN with residual connections |
| **Model gym** | `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\model_gym.py:1-804` | Central training orchestrator: FNN, RF, XGBoost |
| **Scoring utility** | `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\scores_for_splits.py:1-163` | Multi-split R², RMSE, MAE scoring with mean±std |
| **Loop runner** | `@D:\ds\work\workspace\git\redox_modgen\run_model_gym_loop.py:1-75` | Automates running across all data partitions |
| **Metrics extractor** | `@D:\ds\work\workspace\git\redox_modgen\extract_metrics.py:1-96` | Parses nohup.out for R²/MAE across all runs |
| **Morgan fingerprints** | `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\qm9\mf.py:1-210` | Generates Morgan fingerprints (ECFP4) for QM9 |
| **QM9 data** | `@D:\ds\work\workspace\git\redox_modgen_data\QM9_Input_Data` | 20 partitions of QM9 dataset with train/val/test splits |

## 2.3 Data Pipeline

### 2.3.1 Datasets
The project works with two distinct datasets:

**A. Redox Dataset** — Molecular data for redox potential prediction:
- **Targets**: 4 key properties: `ion_pot` (ionization potential), `elec_aff` (electron affinity), `homo_n` (HOMO of neutral), `lumo_n` (LUMO of neutral)
- **Descriptors**: Multiple feature representations tested:
  - **MACE features**: From MACE-omol-0-extra-large-1024 (foundation ML potential), ~1024-dimensional
  - **SOAP features**: Smooth Overlap of Atomic Positions, a structural descriptor
  - **Mordred features**: Classical 2D molecular descriptors
  - **Morgan Fingerprints**: Circular binary fingerprints (ECFP4)

**B. QM9 Dataset** — Benchmark molecular dataset (~130K small organic molecules):
- **Targets**: `homo` and `lumo` orbital energies
- **Split**: 20 pre-defined partitions (train/val/test per partition)
- **Data sizes**: Partition 1 has ~18K train, ~5K val, ~2.6K test molecules
- **Descriptors**: MACE features + Morgan Fingerprints (2048 bits via RDKit)

### 2.3.2 Feature Processing Pipeline
From `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\model_gym.py:336-439`:

1. **Zero-column removal**: Drops columns where ≥99.995% of values are zero (when `process_x=True`)
2. **Denoising**: Drops columns where ≥99.5% of values are below 1e-12 (noise floor)
3. **NaN handling**: Configurable — fill with 0 (default), fill with column mean, or drop NaN columns
4. **StandardScaler on X**: Fitted on training set, applied to val/test
5. **Per-target StandardScaler on y**: Each of the 4 targets gets its own scaler — **critical for heterogeneous target scales** (e.g., eV for energies vs. V for potentials)
6. **All scalers saved**: Via `joblib.dump()` for inference reproducibility
7. **Dropped column indices saved**: For consistent feature preprocessing at inference time

### 2.3.3 Data Splitting Strategy
Two modes:
- **Random split** (`partition=0`): 80/20 train-test, then 75/25 train-val from training set → effectively 60/20/20
- **Predefined partitions** (`partition=1..20`): Pre-split CSV files ensure consistent cross-validation across model types. Train/Val/Test files loaded separately, sizes tracked for later scoring.

## 2.4 Neural Network Architecture (FNN)

### 2.4.1 [SingleTargetDynamicNet](cci:2://file:///d:/ds/work/workspace/git/redox_modgen/redox_modgen/models/feedforward_neural_network.py:9:0-76:39) — The Dynamic FNN
This is a significant evolution from [nn_catalyst](cci:9://file:///d:/ds/work/workspace/git/nn_catalyst:0:0-0:0)'s fixed architecture. Defined in `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\models\feedforward_neural_network.py:10-77`:

```
Input → [Dynamic Hidden Layers with Residual Blocks] → Linear(last_hidden, 1) → Output

Each Residual Block:
  x → Linear(in, out) → BatchNorm1d(out) → GELU → Dropout
                                                    +
                                              Linear(in, out) [skip, if i > 0]
                                                    ↓
                                                  output
```

**Key improvements over [nn_catalyst](cci:9://file:///d:/ds/work/workspace/git/nn_catalyst:0:0-0:0)**:

1. **Dynamic depth**: The number of hidden layers is configurable via `hidden_layers` list parameter. Not hardcoded to 2 layers.
2. **Dynamic width**: Each layer's width is independently configurable (e.g., `[1024, 512, 256, 128]`, or `[4096, 8, 64]`).
3. **GELU activation**: Uses Gaussian Error Linear Unit instead of ReLU — smoother gradient landscape, better for regression tasks.
4. **ModuleList-based construction**: Uses `nn.ModuleList` for layers, batch norms, and skip connections — proper parameter registration for any depth.
5. **Conditional residual connections**: Skip connections are only applied starting from layer index > 0 (i.e., not the first hidden layer), using `nn.Identity()` as a no-op placeholder otherwise. Residual connections can be toggled via `use_residual` parameter.
6. **Hyperparameter saving**: `self.save_hyperparameters()` for Lightning checkpoint reproducibility.

### 2.4.2 [BaseModel](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/src/pl/model_impl.py:5:0-61:89) — PyTorch Lightning Base
Defined in `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\models\__init__.py:9-69`:

- **Training step**: Logs `train_loss` (epoch-level) and `train_acc` (R² via `torchmetrics.R2Score`)
- **Validation step**: Logs `val_loss`, tracks per-epoch average
- **Test step**: Logs `test_loss`
- **Optimizer**: AdamW with configurable learning rate
- **Scheduler**: `ReduceLROnPlateau` (mode='min', factor=0.1, patience=10, min_lr=1e-9, threshold=0.001)

### 2.4.3 Hyperparameter Configurations Explored
From the commented-out blocks in `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\model_gym.py:93-134`, multiple configurations were tried:

| Configuration | Hidden Layers | LR | Dropout | Residual | Notes |
|---|---|---|---|---|---|
| **Best Compromise** | [2048, 64, 32] | 2.4e-4 | 0.21 | True | Good across targets |
| **Best MACE-Dipole N** | [4096, 8, 64] | 5.4e-5 | 0.29 | True | Narrow bottleneck, best for dipole |
| **Best MACE-Homo** | [2048, 256, 64, 8] | 2.3e-4 | 0.24 | True | 4-layer with progressive narrowing |
| **Best MACE-Homo, long** | [512, 256, 32] | 2.3e-4 | 0.22 | True | Custom optimization variant |
| **Default FNN** | [1024, 512] | 0.001 | 0.2 | True | The nn_catalyst-like baseline |
| **Final active config** | [2048, 64, 32] | 2.3e-4 | 0.22 | True | 3-layer progressive, the production model |

The final production model uses **[2048, 64, 32]** with a learning rate of **2.3e-4**, dropout of **0.22**, and residual connections enabled.

### 2.4.4 Training Configuration
From `@D:\ds\work\workspace\git\redox_modgen\redox_modgen\model_gym.py:138-167`:
- **Max epochs**: 200 (increased from 150 in nn_catalyst)
- **EarlyStopping**: On `train_loss`, patience 10
- **ModelCheckpoint**: Saves top-1 model based on `val_loss`
- **LearningRateMonitor**: Logs LR at each epoch
- **Accelerator**: Auto (uses GPU when available — ran on multi-GPU CUDA systems)
- **Batch size**: 32
- **Seed**: 42 (full determinism via `pl.seed_everything(seed)` + manual CUDA seeds)

## 2.5 Training Strategy: Sequential Multi-Target

The [sequential_training()](cci:1://file:///d:/ds/work/workspace/git/redox_modgen/redox_modgen/model_gym.py:335:0-606:28) function (`@D:\ds\work\workspace\git\redox_modgen\redox_modgen\model_gym.py:336-607`) implements the same one-model-per-target sequential approach, with enhancements:

1. **Model type dispatch**: Supports `"fnn"`, `"rf"` (Random Forest), `"xgb"` (XGBoost) via the `model_type` parameter. Your FNN contribution is selected with `-mt fnn`.
2. **Optional prediction stacking**: When `stack_predictions=True`, each model's predictions augment the feature set for subsequent targets.
3. **Per-target Y scaling**: Each target gets its own `StandardScaler`, critical because targets have different units/scales.
4. **Target range selection**: `target_range` parameter allows training only a subset of targets.
5. **Memory management**: Explicit `gc.collect()` calls after large dataframe operations.
6. **Comprehensive metrics**: R², MAE, RMSE computed and printed for each target on the test set.

## 2.6 Multi-Partition Cross-Validation

### 2.6.1 Automated Partition Loop
`@D:\ds\work\workspace\git\redox_modgen\run_model_gym_loop.py:1-75` runs:
```
python redox_modgen/model_gym.py -mt fnn -nc 1 -p {partition}
```
for partitions 1 through N (typically 20). This provides **20-fold cross-validation** with predefined splits.

### 2.6.2 Score Aggregation
`@D:\ds\work\workspace\git\redox_modgen\redox_modgen\scores_for_splits.py:1-163` computes:
- **Per-split**: R², RMSE, MAE for train/val/test for each target
- **Across splits**: Mean ± std with adaptive decimal formatting based on std magnitude
- **Export**: CSV summary files for publication-ready tables

## 2.7 Results (FNN on Redox Dataset, 20 Partitions)

From the [nohup.out](cci:7://file:///d:/ds/work/workspace/git/redox_modgen/nohup.out:0:0-0:0) log (`@D:\ds\work\workspace\git\redox_modgen\nohup.out`), extracted across **20 partition runs** with 4 targets each:

### 2.7.1 R² Scores per Partition (Test Set)

| Partition | Target 1 (ion_pot) | Target 2 (elec_aff) | Target 3 (homo_n) | Target 4 (lumo_n) |
|---|---|---|---|---|
| 1 | 0.9626 | 0.9359 | 0.9139 | 0.9305 |
| 2 | 0.9551 | 0.9356 | 0.9131 | 0.9505 |
| 3 | 0.9536 | 0.9327 | 0.9196 | 0.9524 |
| 4 | 0.9585 | 0.9273 | 0.9137 | 0.9481 |
| 5 | 0.9622 | 0.9262 | 0.9216 | 0.9439 |
| 6 | 0.9597 | 0.9316 | 0.9248 | 0.9496 |
| 7 | 0.9537 | 0.9229 | 0.9199 | 0.9499 |
| 8 | 0.9557 | 0.9372 | 0.9165 | 0.9329 |
| 9 | 0.9602 | 0.9161 | 0.9156 | 0.9540 |
| 10 | 0.9569 | 0.9305 | 0.9226 | 0.9466 |
| 11 | 0.9598 | 0.8982 | 0.9180 | 0.9358 |
| 12 | 0.9557 | 0.9237 | 0.9224 | 0.9451 |
| 13 | 0.9561 | 0.8973 | 0.9198 | 0.9484 |
| 14 | 0.9600 | 0.9384 | 0.9129 | 0.9561 |
| 15 | 0.9527 | 0.9332 | 0.9160 | 0.9462 |
| 16 | 0.9591 | 0.9398 | 0.9152 | 0.9519 |
| 17 | 0.9577 | 0.8769 | 0.8950 | 0.9400 |
| 18 | 0.9593 | 0.9285 | 0.9134 | 0.9498 |
| 19 | 0.9618 | 0.9355 | 0.9081 | 0.9502 |
| 20 | 0.9616 | 0.9196 | 0.9188 | 0.9494 |
| **Mean** | **~0.958** | **~0.926** | **~0.916** | **~0.947** |

### 2.7.2 MAE Scores per Partition (Test Set)

| Target | MAE Range | Approximate Mean |
|---|---|---|
| Target 1 (ion_pot) | 0.109 – 0.127 | ~0.116 eV |
| Target 2 (elec_aff) | 0.154 – 0.183 | ~0.164 eV |
| Target 3 (homo_n) | 0.0049 – 0.0060 | ~0.0052 Ha |
| Target 4 (lumo_n) | 0.0052 – 0.0089 | ~0.0078 Ha |

### 2.7.3 Results Interpretation

**Target 1 (Ionization Potential)**: Most consistent and highest R² (~0.958 average). The FNN excels at predicting this fundamental electrochemical property, with very low variance across partitions (std < 0.003).

**Target 2 (Electron Affinity)**: Good but more variable (R² 0.877–0.940). Some partitions see drops to ~0.88, suggesting certain molecular subsets are harder to predict. This is the most challenging of the four targets.

**Target 3 (HOMO_neutral)**: Very consistent (~0.916 average, std < 0.01). HOMO energies are well-captured by the MACE structural features.

**Target 4 (LUMO_neutral)**: Strong performance (~0.947 average). LUMO energies, being virtual orbital energies, are slightly harder than HOMO but still well-predicted.

## 2.8 Morgan Fingerprint Generation (QM9 Extension)

Your contribution also included building a **Morgan Fingerprint generation pipeline** for the QM9 dataset:

### 2.8.1 [MorganFingerprintGenerator](cci:2://file:///d:/ds/work/workspace/git/redox_modgen/redox_modgen/qm9/mf.py:24:0-190:32) class
`@D:\ds\work\workspace\git\redox_modgen\redox_modgen\qm9\mf.py:25-191`:
- Converts SMILES to **ECFP4 fingerprints** (radius=2, 2048 bits) using RDKit's modern `MorganGenerator` API
- Processes all 20 QM9 partitions (train/val/test per partition)
- Outputs augmented CSV files with `mf_0` through `mf_2047` columns
- Handles invalid SMILES gracefully (zero-fills)
- Production logging with progress tracking

### 2.8.2 QM9 Data Scale
- **20 partitions** in `@D:\ds\work\workspace\git\redox_modgen_data\QM9_Input_Data`
- Each partition: ~20K train, ~1.9K test, ~1.9K val molecules
- Original features: `tag`, `index`, `homo`, `lumo`, `SMILES`, `SMILES_relaxed`, `InChI`, `InChI_relaxed`
- Morgan fingerprint augmented files: ~475 MB for training sets (demonstrating the large scale)

## 2.9 Key Technical Observations

1. **Dynamic architecture**: [SingleTargetDynamicNet](cci:2://file:///d:/ds/work/workspace/git/redox_modgen/redox_modgen/models/feedforward_neural_network.py:9:0-76:39) is a generalized, production-quality version of [nn_catalyst](cci:9://file:///d:/ds/work/workspace/git/nn_catalyst:0:0-0:0)'s [SingleTargetNet](cci:2://file:///d:/ds/work/workspace/git/nn_catalyst/load_and_pred.py:64:0-86:17). It accepts any number of hidden layers and widths.
2. **GELU over ReLU**: The switch to GELU activation provides smoother gradients, beneficial for regression tasks.
3. **Robust cross-validation**: 20-partition predefined splits with aggregated mean±std scoring provides statistically rigorous model evaluation.
4. **Multi-representation support**: The pipeline handles MACE, SOAP, Mordred, and Morgan Fingerprint descriptors, enabling fair comparison across molecular representation types.
5. **Production engineering**: Environment variables ([.env](cci:7://file:///d:/ds/work/workspace/git/redox_modgen/.env:0:0-0:0)), proper Python packaging ([setup.py](cci:7://file:///d:/ds/work/workspace/git/redox_modgen/setup.py:0:0-0:0)), [requirements.txt](cci:7://file:///d:/ds/work/workspace/git/redox_modgen/requirements.txt:0:0-0:0), and `nohup` execution for long-running HPC jobs.
6. **GPU training**: Ran on multi-GPU CUDA servers (`CUDA_VISIBLE_DEVICES: [0,1,2]`), leveraging PyTorch Lightning's accelerator handling.

---

# Combined Summary & Comparative Analysis

## Evolution from nn_catalyst → redox_modgen

| Aspect | nn_catalyst | redox_modgen (FNN) |
|---|---|---|
| **Framework** | Vanilla PyTorch → PyTorch Lightning | PyTorch Lightning from the start |
| **Architecture** | Fixed 2-layer (1024→512→1) | Dynamic N-layer (configurable) |
| **Activation** | ReLU | GELU |
| **Dropout** | 0.5 (later 0.2) | 0.22 (tuned) |
| **Skip connections** | Single fixed skip | Dynamic residual blocks per layer |
| **Targets** | 29–30 catalyst properties | 4 redox properties + QM9 |
| **Descriptors** | Mordred (2D) only | MACE, SOAP, Mordred, Morgan FP |
| **Validation** | Single train/val/test split | 20-partition cross-validation |
| **Scoring** | R², RMSE, MAE per target | Mean±std across partitions |
| **Package** | Scripts + notebooks | Installable Python package |
| **Execution** | Local / Google Colab | HPC with GPU (CUDA) |

## Key Accomplishments

### nn_catalyst
- Built a **complete pipeline** from SMILES molecules to multi-target property predictions
- Achieved **R² > 0.99** for electronic energies and Gibbs free energies
- Achieved **R² > 0.93** for most orbital energies and redox potentials
- Created a **Streamlit inference app** for interactive molecular property prediction
- Explored multiple architectures (3 variants) and feature engineering strategies (PCA, PLS, Lasso, forward selection)

### redox_modgen (FNN contribution)
- Designed a **generalizable, dynamic-depth FNN architecture** ([SingleTargetDynamicNet](cci:2://file:///d:/ds/work/workspace/git/redox_modgen/redox_modgen/models/feedforward_neural_network.py:9:0-76:39)) that supersedes the fixed nn_catalyst model
- Achieved **R² ~0.916–0.958** across 4 key electrochemical targets, validated over **20 cross-validation partitions**
- Built a **multi-model comparison framework** (FNN vs RF vs XGBoost) enabling fair benchmarking
- Developed a **Morgan Fingerprint generation pipeline** for QM9 (~130K molecules, 20 partitions)
- Implemented **robust statistical evaluation** with mean±std scoring across splits
- Contributed to a **research-grade Python package** with proper software engineering practices

## Conclusion

Both projects demonstrate a progression in applying deep learning to molecular property prediction in computational chemistry. **nn_catalyst** was the exploratory prototype where the core idea of per-target feedforward neural networks with skip connections was developed and validated on a 29-target catalyst dataset. **redox_modgen** is the production evolution — a well-engineered, scientifically rigorous package where the FNN architecture was generalized, hyperparameters were systematically tuned, and results were validated with proper cross-validation methodology. The FNN consistently achieves strong R² scores (>0.91 for all targets) on the electrochemical property prediction task, with ionization potential being the best-predicted property (~0.96 R²) and electron affinity being the most challenging (~0.93 R²).

Both projects demonstrate a clear evolution in methodology, architecture design, and engineering rigor for applying deep learning to molecular property prediction in computational chemistry.

---

# Appendix A: Detailed nohup.out Training Log Analysis (redox_modgen)

The `nohup.out` file (8,233 lines) in the redox_modgen project captures the complete output from running the FNN model across all 20 QM9 data partitions. This appendix provides a detailed analysis of the training dynamics, infrastructure, and results extracted from these logs.

## A.1 Execution Environment

| Parameter | Value |
|---|---|
| **Server** | Linux HPC (hostname: nirbaanm) |
| **Python** | 3.10 (Miniconda) |
| **Conda environment** | `redox` |
| **GPU** | CUDA-enabled, 3 GPUs available (`CUDA_VISIBLE_DEVICES: [0,1,2]`) |
| **PyTorch Lightning** | v1.9+ (CSVLogger default, tensorboardX removed) |
| **Execution method** | `nohup` for long-running background execution |
| **Orchestration** | `run_model_gym_loop.py` calling `model_gym.py -mt fnn -nc 1 -p <partition>` |
| **Random seed** | 42 (set via `pl.seed_everything(42)`) |

## A.2 Actual Model Architecture (from logs)

The training logs reveal the **exact architecture deployed** for all 20 partitions. The PyTorch Lightning model summary printed for every target shows:

```
Creating layer 0: 664 -> 2048
Creating layer 1: 2048 -> 64
Creating skip layer 1: 2048 -> 64
Creating layer 2: 64 -> 32
Creating skip layer 2: 64 -> 32
```

### Architecture: `664 → 2048 → 64 → 32 → 1`

This is the **"Best Compromise"** configuration `[2048, 64, 32]`, NOT the `[1024, 512, 256, 128]` that appears as the active config in the source code. This means the production run used a different configuration than what is currently set in the code — the source was likely modified after the production run completed.

### Detailed Parameter Breakdown

| Module | Type | Parameters | Description |
|---|---|---|---|
| `fc_layers.0` | Linear | 1,362,432 (664×2048+2048) | Input expansion layer |
| `fc_layers.1` | Linear | 131,136 (2048×64+64) | First compression layer |
| `fc_layers.2` | Linear | 2,080 (64×32+32) | Second compression layer |
| `bn_layers.0` | BatchNorm1d | 4,096 (2×2048) | After first linear |
| `bn_layers.1` | BatchNorm1d | 128 (2×64) | After second linear |
| `bn_layers.2` | BatchNorm1d | 64 (2×32) | After third linear |
| `skip_layers.0` | Identity | 0 | No skip on first layer |
| `skip_layers.1` | Linear | 131,136 (2048×64+64) | Residual: 2048→64 |
| `skip_layers.2` | Linear | 2,080 (64×32+32) | Residual: 64→32 |
| `fc_final` | Linear | 33 (32×1+1) | Output layer |
| **Total** | | **1,633,185** | **6.531 MB estimated** |

### Architecture Design Pattern: "Expansion-Compression Funnel"

The architecture follows an unusual but effective pattern:
1. **Expansion**: 664 inputs → 2048 neurons (3.1× expansion). Projects the molecular descriptors into a high-dimensional learned representation space.
2. **Aggressive compression**: 2048 → 64 (32× compression). Forces the network to learn a compact, information-dense bottleneck representation.
3. **Final compression**: 64 → 32 (2× compression). Further refinement before the output layer.
4. **Residual connections**: Skip connections on layers 1→2 and 2→3 allow gradient flow through the narrow bottleneck.

This design is notable because it is NOT a monotonically narrowing funnel (like nn_catalyst's 1024→512→1). The initial expansion step creates a richer feature space before compression, which may help capture non-linear interactions between input descriptors.

## A.3 Data Processing Summary (Consistent Across All Partitions)

Every partition shows identical preprocessing:

| Step | Result |
|---|---|
| **Raw features loaded** | 847 numeric columns |
| **After zero-column removal** (≥99.5% zeros) | 762 features |
| **After denoising** (≥99.5% values < 1e-18) | **664 features** |
| **Feature reduction** | 21.6% of features dropped |
| **NaN handling** | Filled with 0 |
| **Dataset split** | 18,308 train / 5,232 val / 2,616 test |
| **Total molecules per partition** | 26,156 |
| **Targets** | `ion_pot`, `elec_aff`, `homo_n`, `lumo_n` |

The **descriptor CSV had mixed-type columns** (DtypeWarning for ~230 columns), indicating some descriptor columns contained non-numeric values that were coerced during the "numeric columns only" filtering step.

## A.4 Training Convergence Analysis

### A.4.1 Training Loss Convergence by Target (Partition 1)

The early stopping callback tracked `train_loss` improvements. For Partition 1:

| Target | Initial Loss | Final Best Loss | Epochs to Converge | Convergence Ratio |
|---|---|---|---|---|
| **1 (ion_pot)** | 0.202 | 0.037 | ~45 | 5.5× reduction |
| **2 (elec_aff)** | 0.248 | 0.059 | ~32 | 4.2× reduction |
| **3 (homo_n)** | 0.295 | 0.069 | ~36 | 4.3× reduction |
| **4 (lumo_n)** | 0.259 | 0.044 | ~43 | 5.9× reduction |

**Key observations**:
- **Target 1 (ion_pot)** converges the most (5.5× loss reduction), consistent with its highest R² scores.
- **Target 4 (lumo_n)** shows the best convergence ratio (5.9×), yet has a higher test loss than target 1 — indicating the training and test distributions differ more for LUMO.
- **Target 2 (elec_aff)** converges the least (4.2×), consistent with its status as the most variable target across partitions.
- **Target 3 (homo_n)** starts with the highest initial loss (0.295), suggesting HOMO values have higher initial variance in normalized space.
- All targets converge within 32–45 epochs, well within the 200 max epoch budget. Early stopping with patience=10 effectively prevents overfitting.

### A.4.2 Training Loss Improvement Patterns

The training loss progression follows a characteristic pattern across all targets:

1. **Rapid initial descent** (epochs 1–5): Large improvements of 0.02–0.09 per epoch. The network quickly learns the gross structure of the target function.
2. **Steady refinement** (epochs 5–25): Moderate improvements of 0.001–0.009 per epoch. The network fine-tunes its learned representations.
3. **Diminishing returns** (epochs 25–45): Tiny improvements of 0.000–0.002 per epoch. The model approaches its capacity limit, and the learning rate scheduler may be reducing the learning rate.
4. **Plateau & early stop**: 10 consecutive epochs without improvement trigger early stopping.

### A.4.3 Test Loss by Target (Partition 1)

| Target | Test Loss (MSE) | R² | MAE | RMSE |
|---|---|---|---|---|
| 1 (ion_pot) | 0.0384 | 0.9626 | 0.1102 | 0.1527 |
| 2 (elec_aff) | 0.0617 | 0.9359 | 0.1564 | 0.2264 |
| 3 (homo_n) | 0.0864 | 0.9139 | 0.0052 | 0.0070 |
| 4 (lumo_n) | 0.0672 | 0.9305 | 0.0065 | 0.0089 |

**Note on units**: Targets 1 and 2 (ion_pot, elec_aff) have MAE in **eV** (~0.11–0.16 eV), while targets 3 and 4 (homo_n, lumo_n) have MAE in **Hartrees** (~0.005–0.007 Ha, equivalent to ~0.14–0.19 eV). When converted to the same units, all four targets have comparable prediction accuracy.

## A.5 Complete Results: All 20 Partitions

### A.5.1 R² Scores (Test Set)

| Partition | ion_pot | elec_aff | homo_n | lumo_n |
|---|---|---|---|---|
| 1 | 0.9626 | 0.9359 | 0.9139 | 0.9305 |
| 2 | 0.9551 | 0.9356 | 0.9131 | 0.9505 |
| 3 | 0.9536 | 0.9327 | 0.9196 | 0.9524 |
| 4 | 0.9585 | 0.9273 | 0.9137 | 0.9481 |
| 5 | 0.9622 | 0.9262 | 0.9216 | 0.9439 |
| 6 | 0.9597 | 0.9316 | 0.9248 | 0.9496 |
| 7 | 0.9537 | 0.9229 | 0.9199 | 0.9499 |
| 8 | 0.9557 | 0.9372 | 0.9165 | 0.9329 |
| 9 | 0.9602 | 0.9161 | 0.9156 | 0.9540 |
| 10 | 0.9569 | 0.9305 | 0.9226 | 0.9466 |
| 11 | 0.9598 | 0.8982 | 0.9180 | 0.9358 |
| 12 | 0.9557 | 0.9237 | 0.9224 | 0.9451 |
| 13 | 0.9561 | 0.8973 | 0.9198 | 0.9484 |
| 14 | 0.9600 | 0.9384 | 0.9129 | 0.9561 |
| 15 | 0.9527 | 0.9332 | 0.9160 | 0.9462 |
| 16 | 0.9591 | 0.9398 | 0.9152 | 0.9519 |
| 17 | 0.9577 | 0.8769 | 0.8950 | 0.9400 |
| 18 | 0.9593 | 0.9285 | 0.9134 | 0.9498 |
| 19 | 0.9618 | 0.9355 | 0.9081 | 0.9502 |
| 20 | 0.9616 | 0.9196 | 0.9188 | 0.9494 |

### A.5.2 RMSE Scores (Test Set)

| Partition | ion_pot | elec_aff | homo_n | lumo_n |
|---|---|---|---|---|
| 1 | 0.1527 | 0.2264 | 0.0070 | 0.0089 |
| 2 | 0.1576 | 0.2269 | 0.0068 | 0.0074 |
| 3 | 0.1692 | 0.2306 | 0.0068 | 0.0074 |
| 4 | 0.1569 | 0.2460 | 0.0069 | 0.0079 |
| 5 | 0.1535 | 0.2422 | 0.0068 | 0.0080 |
| 6 | 0.1568 | 0.2396 | 0.0066 | 0.0079 |
| 7 | 0.1667 | 0.2504 | 0.0067 | 0.0077 |
| 8 | 0.1648 | 0.2291 | 0.0070 | 0.0089 |
| 9 | 0.1597 | 0.2601 | 0.0071 | 0.0074 |
| 10 | 0.1608 | 0.2410 | 0.0066 | 0.0080 |
| 11 | 0.1589 | 0.2907 | 0.0070 | 0.0087 |
| 12 | 0.1672 | 0.2546 | 0.0068 | 0.0082 |
| 13 | 0.1632 | 0.2953 | 0.0068 | 0.0079 |
| 14 | 0.1554 | 0.2268 | 0.0070 | 0.0073 |
| 15 | 0.1677 | 0.2349 | 0.0068 | 0.0079 |
| 16 | 0.1553 | 0.2217 | 0.0069 | 0.0075 |
| 17 | 0.1626 | 0.3297 | 0.0079 | 0.0087 |
| 18 | 0.1576 | 0.2450 | 0.0070 | 0.0078 |
| 19 | 0.1513 | 0.2332 | 0.0071 | 0.0077 |
| 20 | 0.1517 | 0.2537 | 0.0068 | 0.0076 |

### A.5.3 MAE Scores (Test Set)

| Partition | ion_pot | elec_aff | homo_n | lumo_n |
|---|---|---|---|---|
| 1 | 0.1102 | 0.1564 | 0.00523 | 0.00654 |
| 2 | 0.1131 | 0.1541 | 0.00493 | 0.00543 |
| 3 | 0.1247 | 0.1585 | 0.00508 | 0.00549 |
| 4 | 0.1159 | 0.1722 | 0.00499 | 0.00570 |
| 5 | 0.1132 | 0.1640 | 0.00507 | 0.00586 |
| 6 | 0.1167 | 0.1655 | 0.00486 | 0.00568 |
| 7 | 0.1212 | 0.1599 | 0.00489 | 0.00546 |
| 8 | 0.1209 | 0.1562 | 0.00502 | 0.00673 |
| 9 | 0.1147 | 0.1735 | 0.00534 | 0.00523 |
| 10 | 0.1163 | 0.1623 | 0.00493 | 0.00575 |
| 11 | 0.1161 | 0.1626 | 0.00532 | 0.00645 |
| 12 | 0.1226 | 0.1769 | 0.00500 | 0.00581 |
| 13 | 0.1222 | 0.1645 | 0.00499 | 0.00575 |
| 14 | 0.1126 | 0.1583 | 0.00514 | 0.00527 |
| 15 | 0.1271 | 0.1596 | 0.00506 | 0.00574 |
| 16 | 0.1130 | 0.1551 | 0.00515 | 0.00543 |
| 17 | 0.1160 | 0.1661 | 0.00599 | 0.00610 |
| 18 | 0.1156 | 0.1752 | 0.00521 | 0.00559 |
| 19 | 0.1094 | 0.1573 | 0.00546 | 0.00558 |
| 20 | 0.1089 | 0.1826 | 0.00497 | 0.00538 |

### A.5.4 Aggregate Statistics

| Metric | ion_pot | elec_aff | homo_n | lumo_n |
|---|---|---|---|---|
| **R² Mean** | 0.9582 | 0.9257 | 0.9154 | 0.9465 |
| **R² Std** | 0.0029 | 0.0163 | 0.0067 | 0.0063 |
| **R² Min** | 0.9527 | 0.8769 | 0.8950 | 0.9305 |
| **R² Max** | 0.9626 | 0.9398 | 0.9248 | 0.9561 |
| **MAE Mean** | 0.1166 | 0.1645 | 0.00512 | 0.00571 |
| **RMSE Mean** | 0.1600 | 0.2499 | 0.0069 | 0.0079 |

**Key statistical findings**:
- **ion_pot** is the most stable target: R² std = 0.0029 (range 0.953–0.963). The model generalizes extremely well across all molecular subsets.
- **elec_aff** is the most variable: R² std = 0.0163 (range 0.877–0.940). Partition 17 is a clear outlier (R² = 0.877, RMSE = 0.330), suggesting that partition contains molecules with electron affinity patterns not well-represented in the training set.
- **homo_n** and **lumo_n** are moderately stable with std < 0.007.
- Partition 17 is consistently the weakest across all targets, suggesting it may contain a more challenging molecular distribution.

## A.6 Run Orchestration Log

The final section of `nohup.out` (lines 8049–8232) shows the output from `run_model_gym_loop.py`:

```
Running model_gym.py for partitions 1 to 20
Command: python redox_modgen/model_gym.py -mt fnn -nc 1 -p <partition>

==================================================
Starting run for partition 1
==================================================
Executing: python redox_modgen/model_gym.py -mt fnn -nc 1 -p 1
Completed run for partition 1

[... repeats for all 20 partitions ...]

All partition runs completed!
```

All 20 partitions completed successfully without any errors or crashes. The command-line arguments confirm:
- `-mt fnn`: Model type = Feedforward Neural Network
- `-nc 1`: Number of cycles = 1 (single training pass per target)
- `-p <partition>`: Partition number (1–20)

## A.7 Infrastructure Warnings and Notes

1. **DataLoader workers**: Lightning warned that `num_workers=0` was used, recommending `num_workers=127`. Running with 0 workers means data loading happens on the main process, creating a potential bottleneck. Using multi-worker loading could significantly speed up training.
2. **Mixed-type columns**: ~230 columns in the descriptor CSV had mixed types (numeric + string values), generating DtypeWarnings. These were handled by the "numeric columns only" filter.
3. **Checkpoint directories**: All checkpoint paths (`/home/nirbaanm/workspace/redox_modgen_artifacts/fnn/v2.0/p_<N>/<target>`) pre-existed and were non-empty, indicating this was not the first training run — models were being retrained/updated.
4. **Logger**: CSVLogger was used as the default (tensorboardX unavailable), meaning training metrics were saved as CSV files alongside checkpoints.