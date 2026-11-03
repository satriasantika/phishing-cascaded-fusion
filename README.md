# Adaptive Confidence-Based Cascaded Fusion for Multi-Modal Webpage Phishing Detection

Code for the paper *"Adaptive Confidence-Based Cascaded Fusion for Multi-Modal Webpage Phishing Detection"*.

The framework evaluates the **URL**, **HTML text**, and **screenshot** of a webpage in order of computational cost. At every stage, the available evidence is weighted per sample by its confidence, and a page exits the cascade as soon as the fused probability is sufficiently certain. The costly screenshot branch is therefore only used for pages that the textual modalities cannot resolve.

## Pipeline

| Stage | Modality | Features | Classifier |
|---|---|---|---|
| 1 | URL | Character n-gram TF-IDF + 20 lexical features | Linear SVM |
| 2 | URL + HTML | Word n-gram TF-IDF + 32 structural features | Linear SVM |
| 3 | URL + HTML + Screenshot | Frozen MobileNetV3-Small (576-d embedding) | RBF SVM |

All SVM outputs are calibrated with Platt scaling. The proposed method is compared with six static fusion strategies (early fusion, majority voting, average probability, weighted average, stacking, and hybrid static fusion).

## Main results (test set)

| Method | F1 (%) | Mean CPU latency (ms) | Speedup |
|---|---|---|---|
| Average probability fusion | 92.57 | 381.90 | 1.00x |
| Hybrid static fusion | 93.05 | 388.82 | 0.98x |
| **Proposed (cascade + dynamic weighting)** | **92.09** | **29.34** | **13.02x** |

Latency is measured end to end on a CPU with batch size 1. Full results are produced by the notebook.

## Dataset

This work uses the public multi-modal phishing dataset of Lee et al. (2024):
https://github.com/JehLeeKR/Multimodal_LLM_Phishing_Detection

The notebook clones the dataset automatically. After removing samples with conflicting labels or without a screenshot, 3,408 samples remain (1,991 benign and 1,417 phishing), split into 70/15/15 train, validation, and test sets with stratification.

## How to run

1. Open `Phishing_Detection_Cascaded_Fusion.ipynb` in Google Colab.
2. Select a GPU runtime (the paper used a high-RAM runtime with an NVIDIA T4 GPU).
3. Run all cells in order.

The GPU is used only for embedding extraction during training, and all latency measurements run on the CPU. Results, trained models, and figures are written to `outputs/`.

## Reproducibility

- All random processes use `SEED = 42`.
- Every hyperparameter, fusion weight, and cascade threshold is selected on the validation set only.
- The test set is evaluated once at the end, with 1,000 bootstrap resamples for confidence intervals.
- Tested with TensorFlow 2.20.0 on Google Colab. The notebook prints the exact Python, scikit-learn, and TensorFlow versions in its first cell.

## Repository structure

```
.
├── Phishing_Detection_Cascaded_Fusion.ipynb   # full experiment pipeline
├── requirements.txt                            # dependencies for running outside Colab
└── README.md
```

## Citation

If you use this code, please cite the paper (citation will be added after publication).
