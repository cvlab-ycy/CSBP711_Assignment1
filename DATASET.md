# Dataset record: Fashion-MNIST

## Source, licence and date

- Source repository: https://github.com/zalandoresearch/fashion-mnist
- Paper: https://arxiv.org/abs/1708.07747
- Licence: MIT, according to the source repository
- Files used: the four official compressed IDX image and label files
- Download record: 22 September 2026
- Integrity: published MD5 checksums are embedded in `src/study.py`; each download is also recorded with a SHA-256 hash in the generated audit

Fashion-MNIST was created from Zalando article images as a drop-in replacement for MNIST. It contains 70,000 standardised 28 × 28 grayscale product images across ten balanced clothing categories. Labels were obtained from the product-category information described by the dataset creators.

## Preparation and split

The official 60,000-image development pool is cleaned before a stratified 90/10 split. The official 10,000-image test set remains unchanged. The corrected retained counts are:

| Split | Images |
|---|---:|
| Training | 53,979 |
| Validation | 5,998 |
| Test | 10,000 |

The split uses seed 711. All four algorithms use exactly the same saved IDs, labels and split. Pixel values are converted from unsigned bytes to floating-point values in [0, 1]. No augmentation, learned normalisation or pretrained feature extractor is used.

## Data-quality and leakage checks

The pipeline validates file checksums, IDX magic values, array dimensions, label ranges and class counts. It found no missing or non-finite pixels and no exact repeated images.

The declared near-duplicate rule requires both mean absolute pixel difference ≤ 2 and pixel RMSE ≤ 6 on the 0–255 scale. A safe block-mean lower bound prunes impossible pairs, after which full-resolution distances verify the candidates. The corrected audit found 31 near pairs. Six development images belonging to groups that touched the official test set were removed, along with 17 additional repeated development-group members. The retained train, validation and test groups are disjoint under this numeric rule.

This rule does not identify every semantic or geometrically transformed duplicate. The test images are consulted for contamination screening, while model and hyperparameter selection use validation data. Results from the original audit version had already been inspected before the corrected rerun, so the final test is not claimed as a newly blind external holdout.

The exact generated audit is saved as `reports/data_audit/audit.json`; excluded IDs and prepared arrays are regenerated under `data/prepared/` by `python src/study.py prepare`.

