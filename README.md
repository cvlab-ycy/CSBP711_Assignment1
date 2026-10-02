# Spatial structure in clothing classification

**CSBP711 — Datasets and Algorithm Comparison**

Public repository URL: **https://github.com/cvlab-ycy/CSBP711_Assignment1**

This project compares four classifiers on Fashion-MNIST and tests why a CNN performs best on intact clothing images. The explanation is that local edges and neighbouring pixels carry useful shape information. The ablation applies one fixed pixel permutation to every train, validation and test image, preserves the pixel values and labels, and retrains every model. The CNN–MLP accuracy gap changes from **+3.03 percentage points** on intact images to **−0.63 points** after permutation.

## Dataset

- **Name:** Fashion-MNIST
- **Source:** https://github.com/zalandoresearch/fashion-mnist
- **Licence:** MIT, as stated in the source repository
- **Size:** 70,000 labelled 28 × 28 grayscale images in 10 balanced clothing classes
- **Official split:** 60,000 development images and 10,000 test images
- **Corrected working split:** 53,979 training, 5,998 validation and 10,000 test images after the documented duplicate-group audit

The preparation stage downloads the four official files, verifies their published MD5 checksums, checks missing values and exact repeats, performs the declared near-duplicate audit, and saves a stratified split using seed 711. See [DATASET.md](DATASET.md) for the collection description, preprocessing, exclusions and leakage controls.

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

| Member | Individual contribution |
|---|---|
| **Yingfeng Wang 700049354** | **Data provenance and reproducibility.** Verified the Fashion-MNIST source, MIT licence, download date and MD5 checksums, and wrote the dataset record. Designed the data-preparation checks in `study.py` (checksums, label ranges, class counts, missing values) together with Chenye Yang, and recorded the data-audit outputs. Wrote the README and the one-command runner, and built the repository checker. Wrote the training-history review that summarises the six learning-rate pilots and the checkpoint chosen for each run. Checked the comparison table against the saved runs, including the timing and model-size definitions. 
| **Chenye Yang 700052938** | **Models and training pipeline.** Implemented the main pipeline in `study.py`: download, preprocessing and the stratified split with seed 711; the four models (nearest centroid, logistic regression, MLP, CNN); the shared training loop with Adam, early stopping and best-checkpoint restore; the fixed pixel permutation; and test evaluation. Set up the frozen protocol and the environment files, ran the 24 final fits, produced the learning curves, and wrote the training walkthrough. 
| **Tianyi Wang  700053286** | **Leakage audit, testing and explanation.** Wrote the exhaustive near-duplicate audit and designed the duplicate-grouping rule used in `study.py` together with Chenye Yang. Wrote the test suite (permutation invariance, centroid invariance, duplicate grouping, model saving). Built the report stage: result tables, confusion matrices, error examples, the paired CNN–MLP ablation and bootstrap intervals. Wrote the independent result verification and the results write-up. 

## AI and library disclosure

The dataset choice, experimental design, model implementation, training runs and analysis were done by the group. We used AI assistants (OpenAI Codex) in a supporting role: tidying code style and structure, checking and adjusting the model architectures and code, helping us organise the project plan, and adjusting the format, structure and wording of the README and reports. All code and text were reviewed by the group, and every reported number comes from our own saved runs, not from a paper or leaderboard.

We did not use pretrained models. The four models are standard architectures assembled from PyTorch's built-in layers. We based the MLP and CNN designs on the official PyTorch and TensorFlow Fashion-MNIST tutorials listed below, and the CNN follows the convolution–pooling design of LeCun et al. (1998). OpenAI Codex helped us check and adjust the architectures and code. 

The implementation is built on the following open-source libraries:

- **PyTorch**: model layers, Adam optimiser, training and evaluation
- **scikit-learn**: stratified splitting, accuracy and macro-F1, confusion matrices
- **NumPy** and **pandas**: data handling and result tables
- **Matplotlib**: figures
- **pytest**: test suite
- SciPy and Pillow are installed as supporting dependencies of the libraries above.

No pretrained weights are used.



### References

- Xiao H, Rasul K, Vollgraf R. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms[J]. arXiv preprint arXiv:1708.07747, 2017.

- TensorFlow tutorial. *Basic classification: Classify images of clothing.* https://www.tensorflow.org/tutorials/keras/classification
- PyTorch tutorial. *Quickstart* (Fashion-MNIST). https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- PyTorch tutorial. *Training a Classifier.* https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Kingma D P, Ba J. Adam: A method for stochastic optimization[J]. arXiv preprint arXiv:1412.6980, 2014.


## Limitations

The conclusions apply to small centred grayscale product images, the tested architectures, three seeds and one fixed permutation. They should not be transferred directly to colour photographs, object detection, video or image generation. The corrected run reused the official test set after results from an earlier audit version had been inspected; this limitation is disclosed in the report and saved audit.
