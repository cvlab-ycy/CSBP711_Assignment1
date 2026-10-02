# Spatial structure in clothing classification

**CSBP711 — Datasets and Algorithm Comparison**

Public repository URL: **https://github.com/cvlab-ycy/CSBP711_Assignment1**

This project compares four classifiers on Fashion-MNIST and tests why a CNN performs best on intact clothing images. The explanation is that local edges and neighbouring pixels carry useful shape information. The ablation applies one fixed pixel permutation to every train, validation and test image, preserves the pixel values and labels, and retrains every model. The CNN–MLP accuracy gap changes from **+3.03 percentage points** on intact images to **−0.63 points** after permutation.

## Dataset

- **Name:** Fashion-MNIST
- **Source:** https://github.com/zalandoresearch/fashion-mnist
- **Licence:** MIT, as stated in the source repository
- **Version record:** official IDX files downloaded and checksum-verified on **22 September 2026**
- **Size:** 70,000 labelled 28 × 28 grayscale images in 10 balanced clothing classes
- **Official split:** 60,000 development images and 10,000 test images
- **Corrected working split:** 53,979 training, 5,998 validation and 10,000 test images after the documented duplicate-group audit

The preparation stage downloads the four official files, verifies their published MD5 checksums, validates their IDX headers and labels, checks missing values and exact repeats, performs the declared near-duplicate audit, and saves a stratified split using seed 711. See [DATASET.md](DATASET.md) for the collection description, preprocessing, exclusions and leakage controls.

## Methods and fair comparison

The four algorithms are:

1. nearest-centroid baseline;
2. multinomial logistic regression;
3. a two-hidden-layer MLP;
4. a small two-block CNN.

All models use the same retained images, labels, split and seeds 42, 43 and 44. Pixels are scaled by 1/255, with no augmentation or pretrained weights. Trainable models use cross-entropy, Adam, batch size 256, weight decay 0.0001 and the same early-stopping rule. Two prespecified learning rates are tested for each trainable model on validation data only. The selected rate is fixed across the intact and permuted conditions.

## Main results

| Model | Intact test accuracy % | Permuted test accuracy % | Change (pp) |
|---|---:|---:|---:|
| Nearest centroid | 67.69 ± 0.00 | 67.69 ± 0.00 | +0.00 |
| Logistic regression | 84.36 ± 0.06 | 84.36 ± 0.06 | +0.00 |
| MLP | 88.25 ± 0.17 | 88.02 ± 0.32 | −0.23 |
| CNN | 91.28 ± 0.34 | 87.40 ± 0.41 | −3.88 |

Values are mean ± sample standard deviation across the three fixed seeds. Full tables, learning curves, confusion matrices and the ablation analysis are in `reports/`.

## Reproduce from a fresh clone

Python 3.11 is recommended. A CUDA GPU is strongly recommended for the complete experiment, although the pipeline also supports CPU execution.

```bash
conda env create -f environment.yml
conda activate advanced-ai

# Install the PyTorch build appropriate for your machine.
# CPU or macOS:
python -m pip install torch==2.7.1

# CUDA 12.6 alternative, matching the reported run:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

python -m pytest -q
bash scripts/run_all.sh auto
```

Run only one of the two PyTorch installation commands above.

`auto` uses CUDA when available and otherwise uses CPU. To require a GPU and fail instead of silently falling back, run `bash scripts/run_all.sh cuda`.

The end-to-end command performs these stages:

```text
download + verify → prepare + audit → pilot selection → 24 final fits
→ freeze checkpoint hashes → test evaluation → tables and figures → verification
```

All paths are resolved relative to the repository. The code contains no user-specific local path. Raw data, prepared arrays, model weights and predictions are generated under `data/` and `outputs/` and are intentionally excluded from Git. The reported experiment used protocol v2 in `config/protocol.json`.

To run individual stages:

```bash
python src/study.py prepare --device auto
python src/study.py train --device auto
python src/study.py evaluate --device auto
python src/study.py report --device auto
python scripts/verify_results.py
```

## Repository structure

```text
config/                 frozen experimental protocol
src/study.py            preparation, models, training and evaluation pipeline
src/report.py           result tables, plots and ablation analysis
src/training_review.py  analysis of saved histories and pilot fits
scripts/                one-command runner, verification and audit utilities
tests/                  scientific invariant tests
reports/                compact result tables and figures from the reported run
```

## Contributions

Replace every bracketed field with the three members' real names, IDs and truthful work before publication. Commit IDs must come from the final public repository.

| Member | Individual contribution | Commit evidence |
|---|---|---|
| **Yingfeng Wang 700049354** | Verified dataset provenance, licence and checksums; reviewed duplicate and split controls; tested the centroid and logistic baselines; checked the reported runtime, parameter-size and comparison values. | **wyfwyfwyf1234567** |
| **[MEMBER 2 NAME / ID]** | Verified preprocessing, stratified splitting and class balance; reviewed the audit controls; tested the MLP, shared training loop and early stopping; analysed learning curves and learning-rate pilots. | **[ADD MEMBER 2 COMMIT IDS OR PRS]** |
| **[MEMBER 3 NAME / ID]** | Verified duplicate and leakage handling; reviewed dataset preparation; tested the CNN and pixel-permutation condition; reproduced the paired ablation, uncertainty analysis and interpretation. | **[ADD MEMBER 3 COMMIT IDS OR PRS]** |

## AI and library disclosure

OpenAI Codex assisted with code drafting, the duplicate-audit correction, experiment execution, numerical checks, analysis, report drafting and repository preparation. The group is responsible for reviewing the work, understanding the training and evaluation process, and reporting each member's real contribution. Reported performance values come from the saved executable runs, not a paper or leaderboard.

The implementation uses PyTorch layers and optimisers, scikit-learn metrics and splitting, and NumPy, pandas and Matplotlib for analysis. No pretrained weights are used. Fashion-MNIST should be cited as:

> Han Xiao, Kashif Rasul and Roland Vollgraf (2017), “Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.” https://arxiv.org/abs/1708.07747

## Limitations

The conclusions apply to small centred grayscale product images, the tested architectures, three seeds and one fixed permutation. They should not be transferred directly to colour photographs, object detection, video or image generation. The corrected run reused the official test set after results from an earlier audit version had been inspected; this limitation is disclosed in the report and saved audit.
