# Results

Every number here comes from our own runs. Nothing is copied from published leaderboards.

## Data

After cleaning, the splits are 53,979 training, 5,998 validation and 10,000 test images. Every model sees the same 784 pixels, scaled to [0, 1]. The exclusions and the audit's limits are recorded in `data_audit/audit.json`.

## How the models compare

Each model was trained with three seeds (42, 43 and 44), and the table reports the mean. The ± figure is the standard deviation across those three runs, not a confidence interval.

| Condition | Model | Accuracy % (mean ± SD) | Macro-F1 | Fit time (s) | Size (KiB) | Stored coefficients |
|---|---|---:|---:|---:|---:|---:|
| original | Nearest centroid | 67.69 ± 0.00 | 0.6727 | 0.01 | 30.62 | 7840 |
| original | Logistic regression | 84.36 ± 0.06 | 0.8427 | 3.20 | 30.66 | 7850 |
| original | MLP | 88.25 ± 0.17 | 0.8819 | 6.77 | 462.04 | 118282 |
| original | CNN | 91.28 ± 0.34 | 0.9124 | 19.06 | 413.54 | 105866 |
| permuted | Nearest centroid | 67.69 ± 0.00 | 0.6727 | 0.00 | 30.62 | 7840 |
| permuted | Logistic regression | 84.36 ± 0.06 | 0.8427 | 3.18 | 30.66 | 7850 |
| permuted | MLP | 88.02 ± 0.32 | 0.8792 | 5.83 | 462.04 | 118282 |
| permuted | CNN | 87.40 ± 0.41 | 0.8737 | 14.74 | 413.54 | 105866 |

A few notes on the columns:

- **Fit time** covers training plus the validation passes that early stopping needs. It leaves out data transfer, warm-up, saving checkpoints and the learning-rate search.
- **Learning rates:** each trainable model had two learning rates chosen in advance. The centroid model has nothing to tune.
- **Size** counts only the tensors needed for prediction, not the optimizer state. The centroid model just stores one mean image per class, so it has no gradient-trained parameters.

The learning-rate pilots took 53.22 seconds in total. Hardware and package versions are saved with each run.

## What the permutation shows

On the original images the CNN comes first. Once the pixels are shuffled, the MLP comes first.

The CNN beats the MLP by 3.03 percentage points on the original images. After permutation it trails by 0.63 points, so it loses 3.65 points of its lead.

This is the reversal we predicted. Shuffling the pixels with one fixed permutation keeps every pixel value and label but breaks up which pixels sit next to each other. That hurts the CNN, which depends on local patterns, and barely affects the MLP, which treats each pixel as a separate input. So a good part of the CNN's advantage on Fashion-MNIST seems to come from local structure in the images. As a control, the dense models start from weights matched to the shuffled coordinates.

That said, the experiment only changes spatial arrangement, so it can't show that locality is the *only* thing driving performance. Pooling, model capacity, optimisation and early stopping can all affect how sensitive a model is to the shuffle. The CNN and MLP are roughly the same size, and the learning curves show how well each one converged. We picked learning rates on the original validation data and deliberately kept them the same for the permuted runs.

As a sanity check, we compared each model's predictions on original and permuted images. The centroid and logistic models agree on 100% of images, and the MLP on 93.4%. The centroid model is exactly invariant to the shuffle in exact arithmetic. The dense models start from matched weights, so they should agree closely, but small numerical differences can build up during training.

Bootstrap intervals over the paired test predictions, along with the results for each seed, are in `ablation_analysis.json` and `paired_ablation.csv`. Those intervals only describe these particular trained models, and three seeds is a small sample.

## Limitations

- **Duplicate check:** it finds every image pair with mean absolute pixel difference ≤ 2 and RMSE ≤ 6, but it won't catch shifted, rotated or otherwise transformed duplicates.
- **Test set:** it is used exactly as published, repeats included.
- **Rerun after seeing test results:** these results come from a corrected rerun, done after we had already looked at test results from the first version. We kept the hypothesis, models, seeds and learning-rate grid unchanged, and froze the rerun's checkpoints before evaluating them. Still, the test set isn't truly unseen.
- **Scope:** Fashion-MNIST is small, centred, grayscale product photos. These findings shouldn't be assumed to carry over to colour images, object detection in the wild, video or image generation.

## Reproducing

Run `python src/study.py all --device cuda` in the pinned environment. Runs that are already complete are reused. Runs whose protocol or code hash no longer matches are rejected. A manifest of frozen checkpoints is written before any test evaluation.
