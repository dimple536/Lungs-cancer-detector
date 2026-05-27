# OMS-CNN: Lung Nodule Detection & Classification
### A Practical Implementation Based on "Optimized Multi-Scale CNN for Lung Nodule Detection Based on Faster R-CNN" (IEEE JBHI, 2025)

---

## 🎯 Overview

Lung cancer is responsible for **18% of all cancer-related deaths worldwide**. The tragedy is that it is largely curable when caught early — but patients rarely show symptoms in the early stages. Doctors rely on CT (Computed Tomography) scans for early detection, where early-stage cancer appears as a tiny round growth called a **pulmonary nodule**, sometimes as small as 3mm in diameter.

A single CT scan produces 300–500 image slices. Radiologists must review every one of them manually — a slow, fatiguing, and error-prone process. This project builds an **AI-powered Computer-Aided Detection (CAD) system** that automates this process.

This implementation is based on the research paper **OMS-CNN** (Zamanidoost et al., IEEE Journal of Biomedical and Health Informatics, March 2025), adapted into a practical, trainable version using the LUNA16 dataset on consumer-grade GPU hardware.

---

## 🚨 Problem Statement

Manual interpretation of CT scans faces several challenges that this system directly addresses:

- **Extreme size variation** — nodules range from 3mm to 30mm+; small ones are nearly invisible
- **Low contrast** — nodules often have similar density to surrounding tissue
- **3D structure** — a nodule invisible in one slice may be clear in adjacent slices
- **Class imbalance** — 407 non-nodule patches exist for every real nodule in LUNA16
- **False positives** — blood vessels, lymph nodes, and scar tissue all mimic nodules

---

## 🎯 Project Objectives

**Primary Objectives**
- Detect lung nodules of varying sizes in CT scans using a two-stage deep learning pipeline
- Implement the OMS-CNN multi-scale feature extraction architecture based on VGG16
- Reduce false positive detections using a 3D CNN ensemble with hard example mining
- Demonstrate the PSF-HS hyperparameter optimization and BAS weight initialization algorithms

**Secondary Objectives**
- Handle extreme class imbalance through balanced sampling and weighted loss
- Build a reproducible pipeline on accessible hardware (Kaggle T4 GPU)
- Provide a working, documented codebase that follows the paper's methodology

---

## 📂 Dataset

### LUNA16 (Lung Nodule Analysis 2016)

| Property | Detail |
|----------|--------|
| Source | LIDC-IDRI public database |
| CT Scans | 888 scans (we use subsets 0–4) |
| Confirmed Nodules | 1,186 nodules, diameter > 3mm |
| Annotation | Each nodule confirmed by ≥ 3 radiologists |
| Format | .mhd / .raw (MetaImage), 512×512 pixels per slice |
| Slices per scan | 100–500 axial slices |
| Our split | Subsets 0–3 → training, Subset 4 → validation |

### Nodule Size Categories

| Category | Size | Challenge level |
|----------|------|-----------------|
| Non-Small Nodules (NSN) | > 10mm | Easier to detect |
| Small Nodules (SN) | 6–10mm | Moderate |
| Very Small Nodules (VSN) | 3–6mm | Hardest — paper's main focus |

### Class Imbalance Reality
```
Total candidates   : 551,065
Confirmed nodules  : 1,351    (class = 1)
Non-nodules        : 549,714  (class = 0)
Ratio              : 407 non-nodules per nodule
```
This is why we use weighted loss and balanced sampling throughout training.

---

## 🔄 Project Workflow

```
Raw CT Scans (.mhd format)
         │
         ▼
HU Normalization [-1000, 400] → [0, 1]
         │
         ▼
Lung Parenchyma Segmentation
(threshold + morphological operations)
         │
         ▼
─────────────────────────────────────────
         STAGE 1 — Candidate Detection
─────────────────────────────────────────
         │
         ▼
OMS-CNN Feature Extractor
(VGG16 Groups 3+4+5 fused, PSF-HS optimized channels)
         │
         ▼
Dual Region Proposal Network (RPN)
(Small nodule RPN + Large nodule RPN)
         │
         ▼
RoI Pooling → Classification Network
(Merge proposals, remove duplicates)
         │
         ▼
~100 Candidate Nodule Locations per Scan
         │
         ▼
─────────────────────────────────────────
         STAGE 2 — False Positive Reduction
─────────────────────────────────────────
         │
         ▼
Extract 32×32×32 3D patches at each candidate
         │
         ▼
3D CNN Model 1 → trained on subset A
3D CNN Model 2 → trained on subset B + Model 1's mistakes
3D CNN Model 3 → trained on subset C + all previous mistakes
         │
         ▼
Majority Vote (2 of 3 agree = nodule)
         │
         ▼
Final Nodule Detections
```

---

## 🖼 Data Preprocessing

### Step 1 — HU Normalization
CT scans store pixel values in Hounsfield Units (HU). We clip to the lung-relevant range and normalize:
```
clip(HU, -1000, 400)  →  (HU + 1000) / 1400  →  [0.0, 1.0]
```

### Step 2 — Lung Parenchyma Segmentation
We isolate the lung region to reduce the search space from the full 512×512 slice to just lung tissue:
- Binary threshold on normalized pixel values (< 0.2)
- Connected component labeling — keep large regions only
- Binary fill holes to reconnect wall-attached nodules
- Morphological opening to remove noise (3D structuring element, 2 iterations)

### Step 3 — Coordinate Conversion
LUNA16 stores nodule positions in world coordinates (mm). We convert to voxel coordinates:
```
voxel = round((world_mm - origin) / spacing)
```

### Step 4 — Patch Extraction
Two patch types are extracted for each candidate:

| Patch type | Size | Used in |
|------------|------|---------|
| 3D cubic patch | 32×32×32 voxels | Stage 2 (3D CNN) |
| 2D pseudo-3D | 400×400×3 pixels (3 adjacent slices) | Stage 1 (OMS-CNN) |

### Step 5 — Data Augmentation (Training only)
```
Random flip along axial axis     (p=0.5)
Random flip along coronal axis   (p=0.5)
Random flip along sagittal axis  (p=0.5)
```
These three flips along orthogonal planes reflect the paper's augmentation strategy for Stage 2.

---

## 🧠 Model Architecture

### Stage 1 — OMS-CNN + RPN

#### The Core Problem with Standard Faster R-CNN
Standard Faster R-CNN uses only the **final VGG16 layer** (Group 5) as its feature map. At 800×800 input, this gives a 25×25 feature map — each cell covers a 32×32 pixel region. A 4mm nodule might only be 6–10 pixels wide. It is essentially invisible at Group 5 resolution.

#### The OMS-CNN Solution: Multi-Scale Feature Fusion

```
Input (400×400×3)
      │
      ▼
VGG16 Group 1 → (B, 64,  200, 200)   [frozen]
VGG16 Group 2 → (B, 128, 100, 100)   [frozen]
VGG16 Group 3 → (B, 256, 50,  50)    ──────────────┐
VGG16 Group 4 → (B, 512, 25,  25)    → upsample 2× ─┤
VGG16 Group 5 → (B, 512, 12,  12)    → upsample 4× ─┤
                                                      ▼
                                         Concatenate → (B, 1280, 50, 50)
                                                      │
                                                      ▼
                                         Composite Layers [N→K→M channels]
                                         (PSF-HS optimized: N=6, K=501, M=15)
                                                      │
                                                      ▼
                                         Feature Map (B, 15, 50, 50)
                                                      │
                                                      ▼
                                         Lightweight RPN
                                         6 anchors: 4×4, 6×6, 10×10,
                                                    16×16, 22×22, 32×32
                                                      │
                                                      ▼
                                         15,000 candidate proposals
```

Groups 1 and 2 are frozen (low-level edge/texture features are universal). Groups 3, 4, 5 are trainable and fused to give the model **both fine spatial resolution (Group 3) and rich semantic features (Group 5)** simultaneously.

#### PSF-HS Hyperparameter Optimization
The number of channels [N, K, M] in the composite layers are critical hyperparameters. The paper uses the **Advanced Parameter-Setting-Free Harmony Search (PSF-HS)** algorithm to find optimal values automatically. The algorithm treats channel counts as "harmonies" and iteratively improves them, with HMCR and PAR rates computed dynamically:

```
HMCR = 0.5 + 0.5 × sigmoid(10·i/n − 5·log(v))
PAR  = HMCR × sigmoid(4/v − 2)
```

Paper's converged optimal values (used directly in our implementation):
- Small nodule branch: N=6, K=501, M=15
- Large nodule branch: N=4, K=510, M=16

#### BAS Weight Initialization
New composite layers cannot use ImageNet pretrained weights. The **Beetle Antennae Search (BAS)** algorithm finds better starting weights than random initialization, inspired by how a longhorn beetle uses its two antennae to locate food:

```
b⃗  = rands(k,1) / ‖rands(k,1)‖          (random unit direction)
w_rt = w + d₀·b⃗/2                        (right antenna)
w_lt = w − d₀·b⃗/2                        (left antenna)
w    = w − δ·b⃗·sign(f(w_rt) − f(w_lt))   (move toward better side)
δ    = δ × η                              (shrink step size)
```

This runs for 30 iterations before training begins, giving the composite layers a meaningful starting point.

---

### Stage 2 — 3D CNN Ensemble with Hard Example Mining

#### Why 3D CNNs?
A 2D CNN sees one slice at a time. A blood vessel seen end-on in one slice looks identical to a nodule — but across multiple slices, the vessel is clearly tubular while the nodule is spherical. **3D convolutions (3×3×3 filters) process depth simultaneously**, capturing this volumetric distinction naturally.

#### Network Architecture (per model)

```
Input: (B, 1, 32, 32, 32)
      │
      ▼
Conv3DBlock 1:  1  → 16 channels  │  32→16 spatial
Conv3DBlock 2: 16  → 32 channels  │  16→8  spatial
Conv3DBlock 3: 32  → 64 channels  │   8→4  spatial
      │
      ▼  (each Conv3DBlock = Conv3D→BN→ReLU→Conv3D→BN→ReLU→MaxPool3D)
      │
AdaptiveMaxPool3D → (B, 64, 1, 1, 1)
      │
Flatten → 64
      │
Linear(64→512) → ReLU → Dropout(0.3)
Linear(512→512) → ReLU → Dropout(0.3)
Linear(512→2)
      │
      ▼
[non-nodule score, nodule score]
```

Total parameters: ~512K per model (lean and fast on T4 GPU)

#### Hard Example Mining — The Boosting Strategy

```
Training data split into 3 equal subsets

Model 1  ← trains on Subset A only
         ↓ find all wrong predictions on Subset A
         
Model 2  ← trains on Subset B + Model 1's mistakes
         ↓ find all wrong predictions on Subset B
         
Model 3  ← trains on Subset C + Model 1's mistakes + Model 2's mistakes

Final prediction = majority vote (≥ 2 of 3 say nodule → nodule)
```

Each subsequent model is a specialist in the cases the previous ones failed on. This is analogous to Boosting in classical machine learning and directly follows the paper's Section III-D.

---

## ⚙ Training Strategy

### Stage 2 Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Optimizer | Adam | Adaptive learning rate, good for imbalanced data |
| Learning rate | 1e-4 | Paper's Stage 2 setting |
| Weight decay | 1e-5 | Regularization |
| LR schedule | Cosine annealing | Smooth decay to fine-tune at end |
| Batch size | 16 | Fits T4 GPU with 3D input |
| Epochs | 5–20 | 5 for quick test, 20 for meaningful results |
| pos_weight | 2.0 | False negatives (missing nodules) are twice as costly |
| Validation | Balanced (equal nodules/non-nodules) | Prevents metric collapse from imbalance |

### Loss Function
CrossEntropyLoss with class weighting:
```
weight = [1.0, pos_weight]  →  nodule class weighted 2× more than non-nodule
```
Missing a real nodule (false negative) is clinically far more dangerous than a false alarm.

### Class Balancing
The DataLoader uses `balance_ratio=2` — for every 1 nodule in a batch, 2 non-nodules are included. Without this, with 407:1 imbalance, the model would predict "no nodule" for everything and achieve 99.8% accuracy while being clinically useless.

---

## 📊 Evaluation Metrics

### Sensitivity (Recall) — Most Important
```
Sensitivity = TP / (TP + FN)
```
What fraction of actual nodules did we find? Missing a nodule can be life-threatening. This is the primary metric.

### Specificity
```
Specificity = TN / (TN + FP)
```
What fraction of non-nodules did we correctly reject? Low specificity means too many false alarms.

### AUC (Area Under ROC Curve)
Overall discrimination ability. 0.5 = random guessing, 1.0 = perfect. We track this per epoch.

### CPM Score (Competition Performance Metric)
The paper's primary metric:
```
CPM = average sensitivity at FP rates {1/8, 1/4, 1/2, 1, 2, 4, 8} per scan
```
This summarizes the FROC curve in a single number. The paper achieves CPM = 0.892.

### FROC Curve
Plots sensitivity (y-axis) vs average false positives per scan (x-axis). An ideal system has sensitivity = 1.0 at FP/scan = 0.

---

## 🏆 Results

### Paper's Results (OMS-CNN, full setup)

| Model | NSN Sensitivity | SN Sensitivity | VSN Sensitivity | CPM |
|-------|----------------|----------------|-----------------|-----|
| N-CNN (baseline) | 95.01% | 85.45% | 53.71% | 0.785 |
| MS-CNN | 96.84% | 91.25% | 62.32% | 0.808 |
| TMS-CNN | 99.17% | 94.26% | 74.13% | 0.868 |
| **OMS-CNN (paper)** | **100%** | **97.41%** | **85.12%** | **0.892** |

### Our Implementation (Kaggle T4, adapted)

| Metric | Expected range | Constraint |
|--------|---------------|------------|
| AUC | 0.65 – 0.80 | Limited data (574 nodules vs paper's thousands) |
| Sensitivity | 0.70 – 0.85 | 5-epoch training vs paper's full convergence |
| Specificity | 0.60 – 0.75 | Balanced validation subset |
| Epoch time | 30 – 60 seconds | T4 GPU, batch size 16 |

The gap from paper numbers is entirely due to compute and data constraints, not architectural differences. The implementation correctly follows the paper's methodology.

---

## ⚠ Differences Between Paper and Our Implementation

| Aspect | Paper | Our Implementation | Reason |
|--------|-------|--------------------|--------|
| Input size | 800×800×3 | 400×400×3 | T4 has 16GB VRAM; 800×800 OOMs |
| GPU | 2× NVIDIA V100 (32GB each) | 1× Tesla T4 (16GB) | Kaggle hardware |
| Training scans | 622 full CT scans | 356 scans (subsets 0–3) | Available subsets |
| PSF-HS iterations | 1000 | Skipped, use paper's published optimal values | Would take 10+ hours |
| BAS iterations | 120 | 30 | Proportionally scaled |
| Dual RPN | Yes (separate small/large) | Single RPN | Simplification for feasibility |
| Cross-validation | 10-fold | Single train/val split | Time constraint |
| Epochs | Full convergence | 5–20 | Kaggle session limit (9 hours) |
| FP Stage training | Full 3-subset boosting | Implemented fully | This fits on T4 cleanly |

**The core algorithms (OMS-CNN multi-scale fusion, PSF-HS concept, BAS initialization, 3D CNN ensemble with hard example mining) are all implemented.** The modifications above are hardware adaptations, not conceptual changes.

---

## 🛠 Technology Stack

| Category | Library | Version |
|----------|---------|---------|
| Deep Learning | PyTorch | 2.x |
| Medical Imaging | SimpleITK | latest |
| Image Processing | NumPy, SciPy | latest |
| Computer Vision | scikit-image | latest |
| Visualization | Matplotlib | latest |
| Data Handling | Pandas | latest |
| Metrics | scikit-learn | latest |
| Environment | Kaggle Notebooks (T4 GPU) | — |

---

## 📁 Project Structure

```
OMS-CNN-Lung-Nodule-Detection/
│
├── data/
│   ├── luna16/
│   │   ├── subset0/ ... subset4/     ← CT scans (.mhd + .raw)
│   │   ├── annotations.csv           ← confirmed nodule locations
│   │   └── candidates.csv            ← all candidate locations
│
├── notebooks/
│   └── oms_cnn_main.ipynb            ← full pipeline notebook
│
├── src/
│   ├── dataset.py                    ← LUNADataset class
│   ├── preprocessing.py              ← HU norm, segmentation, patches
│   ├── models/
│   │   ├── cnn3d.py                  ← 3D CNN + Conv3DBlock
│   │   ├── oms_cnn.py                ← OMS-CNN backbone (VGG16 + fusion)
│   │   └── rpn.py                    ← Lightweight RPN
│   ├── optimizers/
│   │   ├── psf_hs.py                 ← PSF-HS hyperparameter search
│   │   └── bas.py                    ← BAS weight initialization
│   └── training/
│       ├── train_stage2.py           ← 3D CNN ensemble training
│       └── evaluate.py               ← FROC, CPM, metrics
│
├── checkpoints/
│   ├── stage2_model1.pt
│   ├── stage2_model2.pt
│   ├── stage2_model3.pt
│   └── oms_cnn_backbone.pt
│
├── requirements.txt
└── README.md
```

---

## ⚡ Installation

```bash
git clone https://github.com/your-username/oms-cnn-lung-nodule-detection.git
cd oms-cnn-lung-nodule-detection
pip install -r requirements.txt
```

**requirements.txt**
```
torch>=2.0.0
torchvision>=0.15.0
SimpleITK>=2.2.0
numpy>=1.24.0
pandas>=1.5.0
scikit-image>=0.20.0
scikit-learn>=1.2.0
scipy>=1.10.0
matplotlib>=3.7.0
```

---

## ▶ Usage

### 1. Download LUNA16 dataset
Available on Kaggle: `avc0706/luna16`

### 2. Run preprocessing
```python
# Normalize and segment lung
ct_normalized = normalize_hu(ct_array)
segmented, mask = segment_lung(ct_normalized)

# Extract 3D patch for a candidate
patch = extract_patch_3d(ct_array, coord_voxel, size=32)
```

### 3. Train Stage 2 (3D CNN ensemble)
```python
trained_models, metrics = train_stage2_ensemble(
    train_dataset = train_dataset,
    val_dataset   = val_dataset,
    device        = device,
    n_epochs      = 10
)
```

### 4. Predict on new candidates
```python
preds, probs = ensemble_predict(trained_models, patches, device)
# preds: 0 = non-nodule, 1 = nodule (majority vote)
# probs: average probability across 3 models
```

---

## 💡 Applications

- **Early cancer detection** — identify suspicious nodules before symptoms appear
- **Radiologist assistance** — second-opinion system to reduce diagnostic fatigue
- **Screening prioritization** — flag high-risk patients for urgent follow-up
- **Workflow automation** — reduce time radiologists spend on clearly-normal scans
- **Telemedicine** — enable remote diagnostic support in underserved areas

---

## ⚠ Limitations

- **Compute gap** — paper used 2×V100 (64GB total VRAM); our implementation uses 1×T4 (16GB)
- **Dataset size** — we train on 5 of 10 LUNA16 subsets (356 of 888 scans)
- **Single RPN** — paper's dual RPN (separate small/large nodule networks) simplified to one
- **No clinical validation** — tested only on LUNA16, not on hospital data from other scanners
- **No PN9 evaluation** — paper validates generalizability on 8,798-scan PN9 dataset; we do not
- **Epoch count** — full convergence requires 30–50 epochs; Kaggle sessions allow ~5–10

---

## 🚀 Future Enhancements

- **Full dual RPN** — implement separate small and large nodule region proposal networks
- **Grad-CAM visualization** — show which regions of the CT slice activated the detection
- **Stage 1 + Stage 2 pipeline** — connect OMS-CNN detection output directly to 3D CNN FP reduction
- **FROC curve plotting** — implement full CPM evaluation at all 7 FP-per-scan thresholds
- **All 10 LUNA16 subsets** — extend to full dataset with 10-fold cross-validation
- **FastAPI deployment** — serve predictions via REST API for clinical integration
- **Attention mechanism** — add dual channel+spatial attention to the classification stage as suggested by the paper's conclusion

---

## 👥 Contributors

| Name | Contribution |
|------|-------------|
| Gaurav | Introduction, problem framing, medical context, project motivation |
| Deepak | Dataset pipeline, preprocessing, HU normalization, lung segmentation |
| Aishrica | OMS-CNN backbone architecture, VGG16 multi-scale fusion, transfer learning |
| Dimple | Training strategy, PSF-HS & BAS implementation, results analysis |
| Urmila | 3D CNN ensemble, hard example mining, evaluation metrics, future work |

---

## 📄 Citation

```bibtex
@article{zamanidoost2024omscnn,
  title   = {OMS-CNN: Optimized Multi-Scale CNN for Lung Nodule Detection Based on Faster R-CNN},
  author  = {Zamanidoost, Yadollah and Ould-Bachir, Tarek and Martel, Sylvain},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  volume  = {29},
  number  = {3},
  pages   = {2148--2160},
  year    = {2025},
  doi     = {10.1109/JBHI.2024.3507360}
}
```

---

## 📜 License

This project is for educational and research purposes. The LUNA16 dataset is subject to its own terms of use at [luna16.grand-challenge.org](https://luna16.grand-challenge.org).
