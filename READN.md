# Federated Learning and Distributed Training with Vision Transformers & Structured Sparsity

This repository contains the official PyTorch implementation of our **Distributed Training & Federated Learning** project, focusing on fine-tuning Vision Transformers (ViTs) under statistical data heterogeneity (IID and Non-IID) on the **CIFAR-100** dataset. 

Additionally, the project integrates advanced model sparsification/pruning strategies (Fisher Sensitivity, Weight Magnitude, and Hybrid schemes) optimized through a custom **SparseSGDM** optimizer, alongside an analytical **Mask Overlap Analysis** using the Jaccard similarity index.

---

## 📌 Project Overview & Architecture

Modern Deep Learning models require massive datasets, raising critical concerns regarding data privacy and infrastructure scalability. **Federated Learning (FL)** addresses these by allowing multiple edge clients to collaboratively learn a global model without raw data leaving their local devices.

### Key Components:
1. **Backbone Architecture:** We utilize a pre-trained `vit_small_patch16_224` (Vision Transformer Small with 16x16 patches) sourced from the `timm` library, utilizing **DINO** (Self-distillation with NO labels) self-supervised weights for robust feature extraction.
2. **Federated Aggregation:** Implemented using the foundational **FedAvg (Federated Averaging)** framework.
3. **Structured Sparsification:** Mask-based pruning strategies are evaluated via a custom-built `SparseSGDM` optimizer to freeze parameter subsets during backward passes, mitigating communication overhead and local compute limits.
4. **Mask Overlap Analysis:** Computes pairwise Jaccard similarity matrices between decentralized client masks to assess architectural alignment and collaborative coherence across data silos.

---

## 🛠️ Methodological Framework

### 1. Data Partitioning Schemes
To replicate real-world environments, the 50,000 training images of CIFAR-100 are upscaled to $224 \times 224$ and distributed across **100 clients** using two distinct settings:
* **IID (Independent and Identically Distributed):** Data is uniformly and randomly assigned.
* **Non-IID:** Each client receives samples from only a limited number of classes ($NC \in \{1, 5, 10, 50\}$), simulating massive non-uniform data skewness.

### 2. Custom Optimization via `SparseSGDM`
Standard optimizers do not inherently handle static binary gradient masks during distributed steps. We developed `SparseSGDM` (extending `torch.optim.SGD`), which applies a pre-calculated sensitivity mask directly to gradients *before* parameter state updates:

$$\mathbf{g}_t \leftarrow \mathbf{g}_t \odot \mathbf{M}$$

Where $\mathbf{M}$ is a binary tensor evaluated across strategies including **Fisher Sensitivity** (empirical Fisher Information Matrix diagonal approximation), **Low/High-Magnitude**, and **Hybrid interpolation**.

---

## 📊 Experimental Setup & Hyperparameters

Configuration constraints and settings managed inside the pipeline (`Config` class):

| Parameter | Centralized Baseline | Federated Learning (FedAvg) | Sparse Training |
| :--- | :---: | :---: | :---: |
| **Model Backbone** | `vit_small_patch16_224` | `vit_small_patch16_224` | `vit_small_patch16_224` |
| **Dataset** | CIFAR-100 | CIFAR-100 | CIFAR-100 |
| **Batch Size** | 64 / 128 | 64 | 64 |
| **Base Learning Rate ($\eta$)** | 0.01 / 0.03 / 0.05 | 0.03 | 0.03 |
| **Momentum / Weight Decay** | 0.9 / 5e-4 | 0.9 / 5e-4 | 0.9 / 5e-4 |
| **Total Rounds / Epochs** | 50 Epochs | 50 Comm. Rounds | 5 Epochs (Pruned state) |
| **Local Client Compute** | - | 4 / 8 / 16 Local Epochs | - |
| **Client Selection Fraction ($\mathcal{C}$)** | - | 10% Selected per Round | - |
| **Sparsity Ratios Tested** | - | - | 0.1 / 0.3 / 0.5 / 0.7 / 0.9 |

---

## 📈 Key Results & Findings

### 🏆 1. Centralized vs. Federated Baseline
* **Centralized Baseline:** Reached **45.1% Final Test Accuracy** (with perfect 1.000 training accuracy by epoch 24), highlighting a clear overfitting profile due to the vast capacity of ViT relative to standard CIFAR-100 scales without extreme regularization.
* **Federated Learning Convergence:** FedAvg exhibited significant robustness against client drift. The **optimal federated setting achieved 64.52% Test Accuracy** under a Non-IID configuration of **$NC=5$ classes per client and 4 local epochs**.

### ✂️ 2. Sparse Training Trade-Offs
* The **Low-Magnitude pruning strategy** outperformed all other methods, retaining high structural integrity and reaching **46.47% test accuracy at 0.3 sparsity** (30% parameters frozen).
* High-magnitude pruning drastically degraded predictions, indicating that maintaining large scale weights is necessary for standard classification trajectories in vision tasks.

### 🧬 3. Mask Alignment (Jaccard Index Analytics)
* **Hybrid and Fisher-based masks** recorded high structural coherence across client models, showing **pairwise Jaccard similarity indices $> 0.7$ in deep Transformer layers**.
* Deeper self-attention projection blocks exhibited increased alignment, suggesting that late-stage global semantic features are universally extracted by clients regardless of non-IID label shifts, whereas early blocks remain highly specialized to local visual structures.

---

## 📂 Repository Structure

```text
├── FL_GroupID_14.py            # Complete unified pipeline (Dataloaders, Models, Custom Optimizer, Experiments)
├── sparse_results.csv          # Output performance tracking for sparse masking iterations
├── mask_overlap_results.csv    # Evaluated average Jaccard indices across clients
└── README.md                   # Repository documentation