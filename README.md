# Spatial structure in clothing classification

**CSBP711 — Datasets and Algorithm Comparison**

Public repository URL: **[REPLACE WITH THE FINAL GITHUB URL]**

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

All three members took part in choosing the dataset, agreeing on the hypothesis and ablation design, fixing the experimental protocol, and preparing the slides. Some shared work was committed from one member's account; the table names who did what, not just who pushed it.

| Member | Individual contribution | Main files committed |
|---|---|---|
| **Yingfeng Wang 700049354** | **Data provenance and reproducibility.** Verified the Fashion-MNIST source, MIT licence, download date and MD5 checksums, and wrote the dataset record. Designed the data-preparation checks in `study.py` (checksums, IDX headers, label ranges, class counts, missing values) together with Member 2, and recorded the data-audit outputs. Wrote the README and the one-command runner, and built the repository checker. Wrote the training-history review that summarises the six learning-rate pilots and the checkpoint chosen for each run. Checked the comparison table against the saved runs, including the timing and model-size definitions. | `README.md`, `DATASET.md`, `scripts/run_all.sh`, `scripts/check_repository.py`, `src/training_review.py`, `reports/data_audit/`, `reports/comparison.csv`, pilot and checkpoint review outputs |
| **[MEMBER 2 NAME / ID]** | **Models and training pipeline.** Implemented the main pipeline in `study.py`: download, preprocessing and the stratified split with seed 711; the four models (nearest centroid, logistic regression, MLP, CNN); the shared training loop with Adam, early stopping and best-checkpoint restore; the fixed pixel permutation; and test evaluation. Set up the frozen protocol and the environment files, ran the 24 final fits, produced the learning curves, and wrote the training walkthrough. | `src/study.py`, `config/protocol.json`, `requirements.txt`, `environment.yml`, `TRAINING_WALKTHROUGH.md`, `reports/all_runs.csv`, `reports/figures/learning_curves.png` |
| **[MEMBER 3 NAME / ID]** | **Leakage audit, testing and explanation.** Wrote the exhaustive near-duplicate audit and designed the duplicate-grouping rule used in `study.py` together with Member 2. Wrote the test suite (permutation invariance, centroid invariance, duplicate grouping, model saving). Built the report stage: result tables, confusion matrices, error examples, the paired CNN–MLP ablation and bootstrap intervals. Wrote the independent result verification and the results write-up. | `scripts/exhaustive_duplicate_audit.py`, `tests/test_experiment.py`, `src/report.py`, `scripts/verify_results.py`, `reports/results.md`, `reports/paired_ablation.csv`, `reports/ablation_analysis.json`, ablation and confusion figures |

## AI and library disclosure

The dataset choice, experimental design, model implementation, training runs and analysis were done by the group. We used AI assistants (OpenAI Codex) in a supporting role: tidying code style and structure, helping us organise the project plan, and adjusting the format, structure and wording of the README and reports. All code and text were reviewed by the group, and every reported number comes from our own saved runs, not from a paper or leaderboard.

The implementation uses PyTorch layers and optimisers, scikit-learn metrics and splitting, and NumPy, pandas and Matplotlib for analysis. No pretrained weights are used. Fashion-MNIST should be cited as:

> Han Xiao, Kashif Rasul and Roland Vollgraf (2017), “Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.” https://arxiv.org/abs/1708.07747

## Limitations

The conclusions apply to small centred grayscale product images, the tested architectures, three seeds and one fixed permutation. They should not be transferred directly to colour photographs, object detection, video or image generation. The corrected run reused the official test set after results from an earlier audit version had been inspected; this limitation is disclosed in the report and saved audit.
