# Hierarchical Differential MedViT-3D

**Hierarchical Differential MedViT-3D: Specialized Hybrid Transformer for Multiclass Diagnosis of Neurodegenerative Diseases**

*Accepted at BrainWorks 2026, MICCAI 2026 Satellite Event*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Abstract

Differential diagnosis of neurodegenerative diseases using 3D T1w MRI remains a challenge, particularly when distinguishing between clinically similar dementia subtypes. While deep learning has achieved high performance on binary screening (e.g., controls vs. Alzheimer's Disease), standard architectures often struggle with the class imbalance and subtle anatomical variations, leading to a critical trade-off between global specificity and focal sensitivity.

In this work, we propose a hybrid Dual-Stream framework using a Hierarchical Differential MedViT-3D. First, a fine-scale stream adapts the MedViTV2 architecture to 3D and adds recent Differential Attention and Rotary Positional Embeddings (RoPE) mechanisms to capture focal anomalies that standard attention misses. Second, to counter the "vanishing specificity" of such high-sensitivity models, we fuse this focal stream with global volumetric priors and introduce a hierarchical loss strategy grouping pathologies.

We evaluate our method on a challenging 7-class diagnostic task using over 4,000 scans aggregated from 16 datasets. Our experiments suggest that this dual-stream hierarchical specialization improves the empirical trade-off between rare-variant sensitivity and CN specificity, achieving the highest Top-2 balanced accuracy among the evaluated methods on the held-out OOD set. By explicitly modeling the balance between global context and local saliency, our approach provides a clinically motivated framework for computer-aided diagnosis in complex settings.

---

## Data Availability

This study uses over 4,000 T1-weighted structural MRI scans aggregated from 16 open or controlled-access research cohorts. Access procedures vary per cohort; most require a data use agreement (DUA).

| Dataset           | Reference               | Access                                                                                                     |
| ----------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| **ADNI**          | Mueller et al., 2005    | [adni.loni.usc.edu](https://adni.loni.usc.edu/)                                                            |
| **NACC**          | Beekly et al., 2007     | [naccdata.org](https://naccdata.org/)                                                                      |
| **OASIS**         | Marcus et al., 2007     | [oasis-brains.org](https://www.oasis-brains.org/)                                                          |
| **AIBL**          | Ellis et al., 2009      | [aibl.csiro.au](https://aibl.csiro.au/)                                                                    |
| **FTLDNI / NIFD** | FTLDNI, 2009            | [ida.loni.usc.edu](https://ida.loni.usc.edu/)                                                              |
| **MIRIAD**        | Malone et al., 2013     | [miriad.drc.ion.ucl.ac.uk](https://www.miriad.drc.ion.ucl.ac.uk/)                                          |
| **4RTNI**         | NCT01804452             | [clinicaltrials.gov](https://clinicaltrials.gov/study/NCT01804452)                                         |
| **ABIDE II**      | Di Martino et al., 2017 | [fcon_1000.projects.nitrc.org/indi/abide](http://fcon_1000.projects.nitrc.org/indi/abide/)                 |
| **Cam-CAN**       | Taylor et al., 2017     | [cam-can.org](https://www.cam-can.org/)                                                                    |
| **NDAR**          | Payakachat et al., 2016 | [nda.nih.gov](https://nda.nih.gov/)                                                                        |
| **SWU**           | Wei et al., 2018        | [fcon_1000.projects.nitrc.org](http://fcon_1000.projects.nitrc.org/indi/retro/southwestuni_qiu_index.html) |
| **ALLFTD**        | Boeve et al., 2020      | [allftd.org](https://www.allftd.org/)                                                                      |
| **ICBM**          | Mazziotta et al., 2001  | [loni.usc.edu/research/atlases](https://www.loni.usc.edu/research/atlases)                                 |
| **PDBP**          | Ofori et al., 2016      | [pdbp.ninds.nih.gov](https://pdbp.ninds.nih.gov/)                                                          |
| **IXI**           | Kotikalapudi, 2024      | [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/)                            |
| **DLBS**          | Park et al., 2025       | [openneuro.org](https://openneuro.org/)                                                                    |

We are unable to share the aggregated dataset directly due to the individual DUA restrictions. Researchers should request access from each repository independently.

---

## Key Components

| Component                         | Description                                                                                                          |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **MedViT-3D**                     | 3D adaptation of MedViTV2 (Manzari et al., 2026) with KAN-integrated transformers and dilated neighborhood attention  |
| **Differential Attention**        | Reduces attention noise by computing the difference of two softmax attention maps (Ye et al., 2024)                  |
| **3D Rotary Position Embeddings** | Triton-accelerated 3D RoPE kernel with PyTorch fallback for relative position encoding                               |
| **FasterKAN**                     | Kolmogorov-Arnold Network layers replacing standard MLPs in transformer blocks                                       |
| **Hierarchical Risk-Aware Loss**  | Groups pathologies by clinical atrophy pattern (Intact: CN; Diffuse & Subcortical: AD, DLB, PSP; Focal Cortical: bvFTD, PNFA, SD) and applies focal weighting |
| **SwinDPL Baseline**              | Swin Transformer 3D with DPL (comparison implementation not distributed; interface stub provided)                   |
| **ResNet-3D Baseline**            | 3D ResNet for volumetric classification                                                                              |
| **Global Volumetric Stream** | SVM operating on 133 regional brain volumes to provide complementary whole-brain morphometric predictions |
| **Adaptive Late Fusion** | MLP combining the 7-class SVM and MedViT-3D probability vectors |

---

## Repository Structure

```
├── configs/                          # YAML configuration overrides
│   ├── medvit-baseline.yaml          # MedViT baseline (no RoPE, no DiffAttn)
│   ├── medvit-rope.yaml              # MedViT + RoPE
│   ├── medvit-rope-diff_attn.yaml    # MedViT + RoPE + Differential Attention
│   ├── medvit-rope-diff_attn-hierarchical_loss.yaml  # Full proposed method
│   ├── resnet-baseline.yaml          # ResNet-3D baseline
│   └── swindpl-baseline.yaml         # SwinDPL baseline
├── config-defaults.yaml              # Base configuration (all hyperparameters)
├── dataset/
│   ├── dataset.py                    # NormalDataset, SVMDataset, MRIMixUp
│   ├── preprocessing.py              # MRI preprocessing (registration, normalization)
│   └── create_vit_svm_mlp_dataset.py # Paired ViT+SVM dataset creation
├── models/
│   ├── medvit_diff_3d.py             # MedViT-3D with DiffAttn + RoPE + KAN
│   ├── resnet_3d.py                  # 3D ResNet
│   ├── swin_transformer_3d_dpl.py    # SwinDPL (interface stub)
│   └── modules/
│       ├── fasterkan.py              # FasterKAN implementation
│       ├── rms_norm.py               # RMSNorm
│       └── rotary_kernel.py          # Triton 3D RoPE kernel + PyTorch fallback
├── train/
│   ├── train_transformer.py          # DDP training loop (step-based, gradient accumulation)
│   ├── train_svm.py                  # SVM baseline with Optuna HPO
│   └── train_vit_svm_mlp.py          # ViT+SVM+MLP ensemble training
├── eval/
│   ├── eval_transformer.py           # Bootstrap evaluation + W&B logging
│   ├── eval_svm.py                   # SVM evaluation
│   └── eval_vit_svm_mlp.py           # ViT+SVM+MLP evaluation
├── regularization/
│   ├── hierarchical_loss.py          # HierarchicalRiskAwareLoss
│   └── label_smoothing.py            # LabelSmoothingLoss
├── utils/
│   ├── balanced_sampler.py           # Distributed weighted sampler
│   ├── bootstrap_metric.py           # ECE, MCE, Brier score, bootstrap CIs
│   ├── calibrate_predictions.py      # Post-hoc calibration (Temperature, Platt, Isotonic)
│   ├── calibration.py                # Calibration method implementations
│   ├── distributed_training.py       # DDP initialization utilities
│   ├── ema.py                        # Exponential Moving Average over N model states
│   ├── helper.py                     # Scheduler, param groups, CV split utilities
│   ├── seed.py                       # Reproducibility seeding
│   └── transforms.py                 # MONAI transform wrappers for multi-channel inputs
├── scripts/
│   ├── transformer.sh                # End-to-end train + eval orchestration
│   └── svm.sh                        # SVM train + eval pipeline
├── requirements.txt
└── LICENSE
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA ≥ 11.8 (for Triton kernels and GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/EloiNavet/Hierarchical-Differential-MedViT-3D.git
cd Hierarchical-Differential-MedViT-3D

# Create and activate environment
conda create -n hdmedvit python=3.10 -y
conda activate hdmedvit

# Install dependencies
pip install -r requirements.txt
```

> **Note on NATTEN**: The `natten` package (Neighborhood Attention) requires a CUDA-compatible build. Follow [NATTEN installation instructions](https://github.com/SHI-Labs/NATTEN) if the pip install fails.

---

## Data Preparation

### Expected CSV Format

Training data must be organized as a K-fold cross-validation with CSV files:

```
your_data_dir/
├── fold_0.csv
├── fold_1.csv
├── ...
└── fold_K-1.csv
```
In our experiments, $K=10$.

Each CSV must contain:

| Column      | Required | Description                                   |
| ----------- | -------- | --------------------------------------------- |
| `Subject`   | Yes      | Unique subject identifier                     |
| `Diagnosis` | Yes      | Class label (must match `DISEASES` in config) |
| `T1_path`   | Yes      | Path to the T1-weighted MRI (NIfTI)           |
| `Mask_path` | No       | Path to brain mask or segmentation            |
| `Dataset`   | No       | Source dataset name                           |
| `Age`       | No       | Subject age                                   |
| `Sex`       | No       | Subject sex                                   |

### Preprocessing

Preprocessing converts NIfTI MRI volumes to float16 `.pt` tensors (shape `1×D×H×W`):

```bash
python dataset/preprocessing.py \
  --input-csv ./path/to/subjects.csv \
  --output-dir ./path/to/preprocessed/ \
  --img-size 144 168 144
```

Verify tensor shapes:
```bash
python -c "import torch; t = torch.load('preprocessed/SUBJECT.pt'); print(t.shape, t.dtype)"
# Expected: torch.Size([1, 144, 168, 144]) torch.float16
```

---

## Training

### Transformer Models (Main Method)

```bash
./scripts/transformer.sh \
    --training-csv-dir ./path/to/10fold_CV/ \
    --save-dir ./path/to/saved_models/ \
    --intermediate-dir ./path/to/preprocessed/ \
    --runname my-experiment \
    --eval-csv ./path/to/test_set.csv \
    --cuda-devices 0,1 \
    --config configs/medvit-rope-diff_attn-hierarchical_loss.yaml \
    --wandb-mode online \
    --project-name project
```

**Key options:**

| Flag             | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `--config`       | Config override file (see `configs/` for all variants)  |
| `--cuda-devices` | Comma-separated GPU IDs                                 |
| `--fold N`       | Train a single fold (0-9) instead of all 10             |
| `--checkpoint`   | Resume from a checkpoint `.pt` file                     |
| `--seed N`       | Set global seed for reproducibility (`none` to disable) |
| `--wandb-mode`   | `online`, `offline`, or `disabled`                      |

### Available Configurations

| Config                                         | Architecture | RoPE | DiffAttn | Hierarchical Loss |
| ---------------------------------------------- | ------------ | ---- | -------- | ----------------- |
| `medvit-baseline.yaml`                         | MedViT       |      |          |                   |
| `medvit-rope.yaml`                             | MedViT       | ✓    |          |                   |
| `medvit-rope-diff_attn.yaml`                   | MedViT       | ✓    | ✓        |                   |
| `medvit-rope-diff_attn-hierarchical_loss.yaml` | MedViT       | ✓    | ✓        | ✓                 |
| `resnet-baseline.yaml`                         | ResNet-3D    |      |          |                   |
| `swindpl-baseline.yaml`                        | SwinDPL      |      |          |                   |

### SVM Baseline

```bash
./scripts/svm.sh \
    --training-csv-dir ./path/to/10fold_CV/ \
    --save-dir ./path/to/saved_models/svm/ \
    --intermediate-dir ./path/to/preprocessed/ \
    --runname svm-baseline \
    --eval-csv ./path/to/test_set.csv \
    --metric loss \
    --n-trials 200 \
    --fold all \
    --wandb-mode disabled \
    --project-name project
```

### Training Details

- **Step-based training**: The loop counts optimizer steps (not epochs). Set `STEPS` in config.
- **Gradient accumulation**: Automatic. `accumulation_steps = EFFECTIVE_BATCH_SIZE / (BATCH_SIZE × num_gpus)`.
- **EMA**: Averages the last `EMA_N_MODELS` model states with configurable decay.
- **Validation**: Every `VALIDATION_FREQUENCY` steps. Keeps `KEEP_BEST_N` best checkpoints.
- **Mixed precision**: FP16 by default (`FP16: True` in config).

---

## Evaluation

Evaluation is automatically run after training via the shell scripts. To evaluate standalone:

```bash
python eval/eval_transformer.py \
    --eval-csv ./path/to/test_set.csv \
    --training-csv-dir ./path/to/10fold_CV/ \
    --intermediate-dir ./path/to/preprocessed/ \
    --checkpoints ./path/to/saved_models/model_best_0.pt \
                  ./path/to/saved_models/model_best_1.pt \
    --cuda-device 0
```

### Post-hoc Calibration

```bash
python -m utils.calibrate_predictions \
    --model-dir ./path/to/saved_models/my-experiment/ \
    --output-dir ./path/to/calibrated/ \
    --val-fold all \
    --method temperature
```

Supported calibration methods: `temperature` (Temperature Scaling), `platt` (Platt Scaling), `isotonic` (Isotonic Regression).

### Metrics

#### Reported in the paper
- Top-1 and top-2 balanced accuracy
- Per-class Top-1 / Top-2 sensitivity
- ECE
- 95% bootstrap confidence intervals

### Additional metrics implemented in this repository
- MCC
- Macro-F1
- AUROC / AUPRC
- MCE
- Brier score
- Post-hoc calibration

---

## Results

### Dataset

Subjects are split into In-Distribution (ID, for K-fold CV training, validation and ID testing) and Out-of-Distribution (OOD, held-out test set).

| Subset  | AD                      | CN                        | DLB                    | bvFTD                 | PNFA                 | SD                   | PSP                  | Total |
| ------- | ----------------------- | ------------------------- | ---------------------- | --------------------- | -------------------- | -------------------- | -------------------- | ----- |
| **ID**  | 428 (217M/211F, 55–96y) | 1926 (827M/1099F, 44–94y) | 126 (104M/22F, 50–90y) | 154 (97M/57F, 49–83y) | 41 (19M/22F, 55–82y) | 39 (23M/16F, 50–79y) | 89 (42M/47F, 52–80y) | 2803  |
| **OOD** | 488 (184M/304F, 46–96y) | 528 (220M/308F, 30–100y)  | 47 (39M/8F, 55–89y)    | 90 (59M/31F, 45–76y)  | 40 (17M/23F, 54–81y) | 44 (24M/20F, 50–85y) | 67 (29M/38F, 55–86y) | 1304  |

### Ablation Study (OOD test set, 1,304 subjects)

All results report Top-1 and Top-2 per-class sensitivities (%), BACC (%), and ECE (%). 95% bootstrap CIs in brackets.

| Model Configuration    | Params       | ECE ↓  | Top | BACC ↑         | AD ↑           | CN ↑           | DLB ↑          | bvFTD ↑        | PNFA ↑         | SD ↑            | PSP ↑          |
| ---------------------- | ------------ | ------ | --- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | --------------- | -------------- |
| MedViT-3D (Vanilla)    | 34.70M       | 49     | @1  | 59 [56–63]     | 71 [66–75]     | **88** [85–90] | 6 [0–14]       | **62** [52–72] | 46 [30–63]     | 66 [51–80]      | 78 [67–87]     |
|                        |              |        | @2  | 78 [74–81]     | 90 [88–93]     | **96** [95–98] | 26 [14–39]     | **81** [72–89] | 68 [52–83]     | 93 [85–100]     | 90 [82–96]     |
| + RoPE                 | 34.70M       | 48     | @1  | 57 [53–61]     | 72 [68–76]     | 86 [83–89]     | 9 [2–18]       | **62** [53–73] | 27 [13–42]     | 66 [51–80]      | 76 [66–86]     |
|                        |              |        | @2  | 77 [73–80]     | **91** [88–93] | **96** [94–97] | 26 [14–39]     | 79 [70–87]     | 65 [49–80]     | 93 [85–100]     | 90 [82–96]     |
| + RoPE + DiffAttn      | 31.57M       | **39** | @1  | 57 [53–60]     | 75 [71–79]     | 49 [45–54]     | 6 [0–14]       | **62** [52–72] | 62 [46–78]     | 64 [49–78]      | 78 [67–87]     |
|                        |              |        | @2  | 75 [72–78]     | 90 [88–93]     | 72 [68–75]     | 21 [10–33]     | 74 [65–83]     | **87** [74–97] | **95** [88–100] | 88 [80–95]     |
| + RoPE + DiffAttn + HL | 31.57M       | 41     | @1  | 60 [56–63]     | **76** [72–80] | 49 [45–53]     | 6 [0–14]       | 58 [47–68]     | **73** [58–87] | 82 [70–93]      | 75 [64–85]     |
|                        |              |        | @2  | 75 [72–78]     | **91** [88–93] | 72 [69–76]     | 15 [6–26]      | 74 [65–83]     | 84 [71–95]     | **95** [88–100] | **92** [86–98] |
| **Dual-Stream (Ours)** | 31.57M + seg | 47     | @1  | **62** [58–66] | 59 [55–63]     | 63 [59–67]     | **21** [10–33] | **62** [52–72] | 60 [43–76]     | **86** [75–96]  | **84** [74–92] |
|                        |              |        | @2  | **82** [79–86] | 78 [75–82]     | 84 [80–87]     | **83** [71–93] | 76 [66–84]     | 73 [58–87]     | **95** [88–100] | 88 [80–95]     |

### Comparison with Baselines (OOD test set, 1,304 subjects)

| Method              | Params       | ECE ↓  | Top | BACC ↑         | AD ↑           | CN ↑             | DLB ↑          | bvFTD ↑        | PNFA ↑         | SD ↑            | PSP ↑          |
| ------------------- | ------------ | ------ | --- | -------------- | -------------- | ---------------- | -------------- | -------------- | -------------- | --------------- | -------------- |
| Lifespan Tree       | seg          | N/A    | @1  | 55 [52–59]     | 44 [39–48]     | 74 [70–78]       | **32** [19–46] | 43 [33–54]     | 33 [18–48]     | **91** [81–98]  | 71 [61–82]     |
|                     |              |        | @2  | 71 [68–75]     | 60 [56–65]     | 80 [77–83]       | 72 [59–85]     | 59 [48–69]     | 50 [34–66]     | **95** [88–100] | 82 [72–91]     |
| SVM (Volumes Only)  | seg          | **47** | @1  | 46 [42–49]     | 43 [38–47]     | **97** [96–98]   | 0 [0–0]        | 51 [41–61]     | 17 [6–30]      | 57 [41–72]      | 58 [46–70]     |
|                     |              |        | @2  | 65 [61–69]     | 77 [73–81]     | **100** [99–100] | 13 [4–23]      | 70 [60–79]     | 43 [27–58]     | 75 [61–87]      | 79 [69–88]     |
| 3D ResNet-50        | 46.17M       | 49     | @1  | 56 [52–60]     | 68 [63–72]     | 93 [91–95]       | 2 [0–7]        | 60 [50–70]     | 24 [11–39]     | 66 [51–80]      | 79 [69–88]     |
|                     |              |        | @2  | 76 [73–80]     | 89 [86–92]     | 98 [97–99]       | 28 [15–41]     | 78 [69–86]     | 59 [43–75]     | 91 [81–98]      | **91** [84–97] |
| 3D Swin DPL         | 41.13M       | 48     | @1  | 56 [52–60]     | **71** [66–75] | 81 [77–84]       | 4 [0–11]       | **64** [54–74] | 30 [15–46]     | 70 [56–84]      | 70 [59–81]     |
|                     |              |        | @2  | 74 [71–78]     | **91** [88–93] | 94 [92–96]       | 24 [12–37]     | **84** [77–92] | 49 [32–66]     | **95** [88–100] | 84 [74–92]     |
| MedViT-3D (Vanilla) | 34.70M       | 49     | @1  | 59 [56–63]     | **71** [66–75] | 88 [85–90]       | 6 [0–14]       | 62 [52–72]     | 46 [30–63]     | 66 [51–80]      | 78 [67–87]     |
|                     |              |        | @2  | 78 [74–81]     | 90 [88–93]     | 96 [95–98]       | 26 [14–39]     | 81 [72–89]     | 68 [52–83]     | 93 [85–100]     | 90 [82–96]     |
| **Ours**            | 31.57M + seg | **47** | @1  | **62** [58–66] | 59 [55–63]     | 63 [59–67]       | 21 [10–33]     | 62 [52–72]     | **60** [43–76] | 86 [75–96]      | **84** [74–92] |
|                     |              |        | @2  | **82** [79–86] | 78 [75–82]     | 84 [80–87]       | **83** [71–93] | 76 [66–84]     | **73** [58–87] | **95** [88–100] | 88 [80–95]     |

### Reproducing Results

The commands below cover the configurations supported by the distributed code. The SwinDPL comparison is not reproducible from this repository: the implementation used for that comparison is not distributed here, and `models/swin_transformer_3d_dpl.py` is an interface stub that raises `NotImplementedError`.

```bash
# MedViT-3D Vanilla baseline
./scripts/transformer.sh \
    --training-csv-dir ./data/10fold_CV/ --save-dir ./checkpoints/ \
    --intermediate-dir ./data/preprocessed/ --runname medvit-vanilla \
    --eval-csv ./data/test_set.csv --config configs/medvit-baseline.yaml

# + RoPE
./scripts/transformer.sh \
    --training-csv-dir ./data/10fold_CV/ --save-dir ./checkpoints/ \
    --intermediate-dir ./data/preprocessed/ --runname medvit-rope \
    --eval-csv ./data/test_set.csv --config configs/medvit-rope.yaml

# + RoPE + DiffAttn
./scripts/transformer.sh \
    --training-csv-dir ./data/10fold_CV/ --save-dir ./checkpoints/ \
    --intermediate-dir ./data/preprocessed/ --runname medvit-rope-diffattn \
    --eval-csv ./data/test_set.csv --config configs/medvit-rope-diff_attn.yaml

# + RoPE + DiffAttn + Hierarchical Loss
./scripts/transformer.sh \
    --training-csv-dir ./data/10fold_CV/ --save-dir ./checkpoints/ \
    --intermediate-dir ./data/preprocessed/ --runname medvit-rope-diffattn-hl \
    --eval-csv ./data/test_set.csv --config configs/medvit-rope-diff_attn-hierarchical_loss.yaml

# Dual-Stream (Full proposed method: ViT + SVM + MLP ensemble)
# Step 1: Train the transformer backbone (+ RoPE + DiffAttn + HL)
./scripts/transformer.sh \
    --training-csv-dir ./data/10fold_CV/ --save-dir ./checkpoints/ \
    --intermediate-dir ./data/preprocessed/ --runname dual-stream \
    --eval-csv ./data/test_set.csv --config configs/medvit-rope-diff_attn-hierarchical_loss.yaml

# Step 2: Train the SVM on volumetric features
./scripts/svm.sh \
    --training-csv-dir ./data/10fold_CV/ --save-dir ./checkpoints/svm/ \
    --intermediate-dir ./data/preprocessed/ --runname dual-stream-svm \
    --eval-csv ./data/test_set.csv --metric bacc --n-trials 200 --fold all \
    --wandb-mode disabled --project-name project

# Step 3: Generate paired ViT/SVM probability datasets for MLP training
python dataset/create_vit_svm_mlp_dataset.py \
    --training-csv-dir ./data/10fold_CV/ \
    --vit-intermediate-dir ./data/preprocessed/transformer/7classes/ \
    --svm-intermediate-dir ./data/preprocessed/svm/7classes/ \
    --vit-checkpoints ./checkpoints/dual-stream/model_*_best*.pt \
    --svm-dir ./checkpoints/svm/dual-stream-svm/ \
    --output-dir ./data/paired_datasets/7classes/ \
    --cuda-device 0

# Step 4: Train the MLP fusion model on paired probability vectors
python train/train_vit_svm_mlp.py \
    --dataset-dir ./data/paired_datasets/7classes/ \
    --save-dir ./checkpoints/mlp/ \
    --runname dual-stream-mlp \
    --wandb-mode disabled

# Step 5: Evaluate the trained ViT + SVM + MLP ensemble end-to-end
python eval/eval_vit_svm_mlp.py \
    --mode end-to-end \
    --training-csv-dir ./data/10fold_CV/ \
    --vit-intermediate-dir ./data/preprocessed/transformer/7classes/ \
    --svm-intermediate-dir ./data/preprocessed/svm/7classes/ \
    --vit-dir ./checkpoints/dual-stream/ \
    --svm-dir ./checkpoints/svm/dual-stream-svm/ \
    --mlp-checkpoints ./checkpoints/mlp/dual-stream-mlp/mlp_*.pt \
    --eval-csv ./data/test_set.csv \
    --output-dir ./results/dual-stream-mlp/ \
    --cuda-device 0

# ResNet-3D baseline
./scripts/transformer.sh \
    --training-csv-dir ./data/10fold_CV/ --save-dir ./checkpoints/ \
    --intermediate-dir ./data/preprocessed/ --runname resnet-baseline \
    --eval-csv ./data/test_set.csv --config configs/resnet-baseline.yaml

# SVM baseline (volumes only)
./scripts/svm.sh \
    --training-csv-dir ./data/10fold_CV/ --save-dir ./checkpoints/svm-baseline/ \
    --intermediate-dir ./data/preprocessed/ --runname svm-baseline \
    --eval-csv ./data/test_set.csv --metric bacc --n-trials 200 --fold all \
    --wandb-mode disabled --project-name project
```

---

## Configuration System

All hyperparameters are defined in [config-defaults.yaml](config-defaults.yaml) with `desc` (description) and `value` fields. Override files in `configs/` only specify changed values and are merged at runtime.

**Example**: To create a new experiment variant, create a YAML file with only the parameters you want to change:

```yaml
# configs/my-experiment.yaml
ARCHITECTURE:
  value: "MedViT"

USE_ROPE:
  value: True

USE_DIFF_ATTN:
  value: True

LOSS:
  value: "hierarchical_risk_aware"
```

See [config-defaults.yaml](config-defaults.yaml) for all available parameters and their descriptions.

---

## Acknowledgments

This work builds upon the following open-source projects:

- **MedViTV2** - Manzari, O., Asgariandehkordi, H., Koleilat, T., Xiao, Y., Rivaz, H. (2026). *MedViTV2: Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention*. Applied Soft Computing, 186, 114045 [[Paper]](https://doi.org/10.1016/j.asoc.2025.114045) [[Code]](https://github.com/Omid-Nejati/MedViTV2)
- **Differential Transformer** - Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., Wei, F. (2024). *Differential Transformer*. [[Paper]](
https://doi.org/10.48550/arXiv.2410.05258) [[Code]](https://github.com/microsoft/unilm/tree/master/Diff-Transformer)
- **Rotary Position Embeddings (RoPE)** - Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. Neurocomputing, 568, 127063. [[Paper]](https://doi.org/10.1016/j.neucom.2023.127063) [[Code]](https://huggingface.co/docs/transformers/model_doc/roformer)
- **FasterKAN** - Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, Y., Tegmark, M. (2024). *KAN: Kolmogorov-Arnold Networks*. [[Paper]](https://doi.org/10.48550/arXiv.2404.19756) [[Code]](https://github.com/AthanasiosDelis/faster-kan)
- **NATTEN** - Hassani, A., Walton, S., Li, J., Li, S., Shi, H. (2023). *Neighborhood Attention Transformer*. [[Paper 1]](https://doi.org/10.48550/arXiv.2204.07143) [[Paper 2]](https://doi.org/10.48550/arXiv.2209.15001) [[Code]](https://github.com/SHI-Labs/NATTEN)

---

## Citation

```bibtex
@inproceedings{
  navet2026hierarchical,
  title={Hierarchical Differential MedViT-3D: Specialized Hybrid Transformer for Multiclass Diagnosis of Neurodegenerative Diseases},
  author={{\'E}loi Navet and R{\'e}mi Giraud and Boris Mansencal and Pierrick Coup{\'e}},
  booktitle={The Brain Abnormality Workshop at MICCAI 2026},
  year={2026},
  url={https://openreview.net/forum?id=VxngQF1rlt}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
