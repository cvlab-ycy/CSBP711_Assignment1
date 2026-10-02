from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"centroid": "Nearest centroid", "logistic": "Logistic regression", "mlp": "MLP", "cnn": "CNN"}


def read(path):
    return json.loads(Path(path).read_text())


def paired_group_bootstrap(values, groups, seed=711, repeats=2000):
    """CI for the mean paired difference, resampling detected test image groups.

    Values first average the fixed training seeds. This captures test-sample
    uncertainty conditional on those runs, not all possible training randomness.
    """
    _, inverse = np.unique(groups, return_inverse=True)
    n = inverse.max() + 1
    sums = np.bincount(inverse, weights=values, minlength=n)
    counts = np.bincount(inverse, minlength=n)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        indices = rng.integers(0, n, n)
        samples.append(sums[indices].sum() / counts[indices].sum())
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def main():
    cfg = read(ROOT / "config/protocol.json")
    expected = len(cfg["models"]) * len(cfg["conditions"]) * len(cfg["seeds"])
    paths = sorted((ROOT / "outputs/runs").glob("*/test.json"))
    if len(paths) != expected or not (ROOT / "outputs/test_complete.json").exists():
        raise RuntimeError(f"Need all {expected} completed runs before reporting; found {len(paths)}")
    reports = ROOT / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = [read(p) for p in paths]
    frame = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows])
    frame.to_csv(reports / "all_runs.csv", index=False)
    summary = frame.groupby(["condition", "model"], sort=False).agg(
        accuracy_mean=("test_accuracy", "mean"), accuracy_sd=("test_accuracy", "std"),
        macro_f1_mean=("test_macro_f1", "mean"), macro_f1_sd=("test_macro_f1", "std"),
        training_seconds_mean=("fit_seconds_including_early_stopping_validation", "mean"),
        training_seconds_sd=("fit_seconds_including_early_stopping_validation", "std"),
        stored_coefficients=("stored_coefficients", "first"), trainable_parameters=("parameters", "first"),
        model_tensor_bytes=("tensor_bytes", "first"), checkpoint_bytes=("checkpoint_bytes", "first"),
        epochs_mean=("epochs_run", "mean"), best_epoch_mean=("best_epoch", "mean"),
        inference_seconds_mean=("test_inference_seconds", "mean")
    ).reset_index()
    summary.to_csv(reports / "comparison.csv", index=False)
    original = summary[summary.condition == "original"].sort_values("accuracy_mean", ascending=False)
    shuffled = summary[summary.condition == "permuted"].sort_values("accuracy_mean", ascending=False)
    pivot = frame.pivot(index="seed", columns=["condition", "model"], values="test_accuracy")
    paired = pd.DataFrame({"seed": pivot.index,
        "cnn_minus_mlp_original": pivot[("original", "cnn")].values - pivot[("original", "mlp")].values,
        "cnn_minus_mlp_permuted": pivot[("permuted", "cnn")].values - pivot[("permuted", "mlp")].values})
    paired["cnn_relative_advantage_lost"] = paired.cnn_minus_mlp_original - paired.cnn_minus_mlp_permuted
    paired.to_csv(reports / "paired_ablation.csv", index=False)
    with np.load(ROOT / "data/prepared/test.npz") as d:
        labels, images, groups = d["y_test"], d["x_test"], d["groups"]
    correct = {}
    predictions = {}
    for condition in cfg["conditions"]:
        for model in cfg["models"]:
            arr = np.stack([np.load(ROOT / "outputs/runs" / f"{model}_{condition}_{seed}" / "test_predictions.npy") for seed in cfg["seeds"]])
            predictions[condition, model] = arr
            correct[condition, model] = (arr == labels).mean(0)
    original_gap = correct["original", "cnn"] - correct["original", "mlp"]
    permuted_gap = correct["permuted", "cnn"] - correct["permuted", "mlp"]
    control_agreement = {}
    for model in ["centroid", "logistic", "mlp"]:
        control_agreement[model] = float((predictions["original", model] == predictions["permuted", model]).mean())
    inference = {
        "original_winner": original.iloc[0]["model"], "permuted_winner": shuffled.iloc[0]["model"],
        "cnn_mlp_rank_reversal": bool(original_gap.mean() > 0 and permuted_gap.mean() < 0),
        "original_cnn_minus_mlp": float(original_gap.mean()),
        "permuted_cnn_minus_mlp": float(permuted_gap.mean()),
        "cnn_relative_advantage_lost": float((original_gap - permuted_gap).mean()),
        "original_gap_group_bootstrap_95ci": paired_group_bootstrap(original_gap, groups),
        "permuted_gap_group_bootstrap_95ci": paired_group_bootstrap(permuted_gap, groups),
        "relative_advantage_lost_group_bootstrap_95ci": paired_group_bootstrap(original_gap - permuted_gap, groups),
        "bootstrap_interpretation": "Paired test-group bootstrap, averaging the three fixed training seeds first. Conditional on these fitted models; not a confidence interval over all possible training randomness.",
        "original_vs_permuted_prediction_agreement": control_agreement,
        "all_original_models_ranked": original.model.tolist(), "all_permuted_models_ranked": shuffled.model.tolist()
    }
    (reports / "ablation_analysis.json").write_text(json.dumps(inference, indent=2) + "\n")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    positions = np.arange(len(cfg["models"]))
    for j, condition in enumerate(cfg["conditions"]):
        subset = summary[summary.condition == condition].set_index("model").loc[cfg["models"]]
        bars = ax.bar(positions + (j - .5) * .36, subset.accuracy_mean * 100, width=.36, yerr=subset.accuracy_sd * 100, capsize=4, label="Original" if condition == "original" else "Fixed pixel permutation", color=["#245B78", "#DE9850"][j])
        ax.bar_label(bars, labels=[f"{v*100:.2f}" for v in subset.accuracy_mean], padding=4, fontsize=8)
    ax.set_xticks(positions, [NAMES[m] for m in cfg["models"]])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Same images and labels, changed spatial relationships")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0, -.17, "Bars: mean of three seeds. Error bars: sample standard deviation across seeds.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "accuracy_ablation.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / "accuracy_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for j, name in enumerate(["logistic", "mlp", "cnn"]):
        for condition, color in [("original", "#245B78"), ("permuted", "#DE9850")]:
            for k, seed in enumerate(cfg["seeds"]):
                history = read(ROOT / "outputs/runs" / f"{name}_{condition}_{seed}" / "history.json")
                axes[0, j].plot([a["epoch"] for a in history], [a["validation_accuracy"] * 100 for a in history], color=color, alpha=.6, label=condition if k == 0 else None)
                axes[1, j].plot([a["epoch"] for a in history], [a["validation_loss"] for a in history], color=color, alpha=.6)
        axes[0, j].set_title(NAMES[name])
        axes[1, j].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Validation accuracy (%)")
    axes[1, 0].set_ylabel("Validation cross-entropy")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "learning_curves.png", dpi=180)
    plt.close(fig)
    classes = read(ROOT / "data/prepared/audit.json")["class_names"]
    for condition in cfg["conditions"]:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.7))
        for ax, model in zip(axes, cfg["models"]):
            matching = [r for r in rows if r["model"] == model and r["condition"] == condition]
            cm = np.array([r["confusion_matrix"] for r in matching]).mean(0)
            rates = cm / cm.sum(1, keepdims=True)
            ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(NAMES[model])
            ax.set_xticks(range(10), classes, rotation=90, fontsize=7)
            ax.set_yticks(range(10), classes, fontsize=7)
            ax.set_xlabel("Predicted")
        axes[0].set_ylabel("True class")
        fig.suptitle(f"Class-normalised confusion matrices: {condition}")
        fig.tight_layout()
        fig.savefig(figures / f"confusion_{condition}.png", dpi=160)
        plt.close(fig)
   
    winner = original.iloc[0]["model"]
    pred = predictions["original", winner][0]
    wrong = np.flatnonzero(pred != labels)[:12]
    fig, axes = plt.subplots(3, 4, figsize=(8, 7))
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"True: {classes[labels[i]]}\nPred: {classes[pred[i]]}", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(wrong):]:
        ax.axis("off")
    fig.suptitle(f"First 12 errors by test index: {NAMES[winner]}, seed {cfg['seeds'][0]}")
    fig.tight_layout()
    fig.savefig(figures / "error_examples.png", dpi=180)
    plt.close(fig)
    audit = read(ROOT / "data/prepared/audit.json")
    selection = read(ROOT / "outputs/selection.json")
    lines = ["# Experiment results", "", "All values below come from runs in this project. No leaderboard numbers are used.", "", "## Data", "", f"Final split counts: {audit['final_counts']}. All models received 784 pixels scaled by 1/255. See data_audit/audit.json for exclusions and limitations.", "", "## Measured comparison", "", "Accuracy and macro-F1 are means across seeds 42, 43 and 44. Accuracy uncertainty below is sample SD across seeds, not a confidence interval.", "", "| Condition | Model | Accuracy % mean ± SD | Macro-F1 | Fit seconds | Model KiB | Stored coefficients |", "|---|---|---:|---:|---:|---:|---:|"]
    for condition in cfg["conditions"]:
        for model in cfg["models"]:
            r = summary[(summary.condition == condition) & (summary.model == model)].iloc[0]
            lines.append(f"| {condition} | {NAMES[model]} | {r.accuracy_mean*100:.2f} ± {r.accuracy_sd*100:.2f} | {r.macro_f1_mean:.4f} | {r.training_seconds_mean:.2f} | {r.model_tensor_bytes/1024:.2f} | {r.stored_coefficients} |")
    lines += ["", "Fit time includes validation checks needed for early stopping, excludes data transfer, startup warmup, checkpoint serialization and hyperparameter search. Each trainable method had two prespecified learning-rate candidates; the centroid has no tuning. Model KiB measures prediction-state tensors, excluding the optimizer. The centroid stores class means and has zero gradient-trained parameters.", "", f"Total pilot fit time: {selection['pilot_fit_seconds']:.2f} seconds. Hardware and package versions are recorded per run.", "", "## Explanation and ablation", "", f"Original winner: **{NAMES[inference['original_winner']]}**. Permuted winner: **{NAMES[inference['permuted_winner']]}**.", "", f"CNN minus MLP accuracy: {inference['original_cnn_minus_mlp']*100:.2f} percentage points originally and {inference['permuted_cnn_minus_mlp']*100:.2f} after permutation. Relative CNN advantage lost: {inference['cnn_relative_advantage_lost']*100:.2f} percentage points.", ""]
    if inference["cnn_mlp_rank_reversal"]:
        lines += ["The predicted CNN–MLP rank reversal occurred. This supports the explanation that useful local relationships in these clothing images contribute to the CNN's original advantage. The fixed permutation preserves pixel values and labels while disrupting locality; dense-model initialisation is coordinate-matched as a control."]
    elif inference["cnn_relative_advantage_lost"] > 0:
        lines += ["The CNN lost relative advantage, but the predicted CNN–MLP rank reversal did not occur. This gives partial support to the locality explanation, not confirmation of the full prediction. The reported rankings are retained without altering the test protocol."]
    else:
        lines += ["The predicted relative loss of CNN advantage was not observed. These results do not support the proposed explanation under this protocol. Report this finding rather than claiming the hypothesis succeeded."]
    lines += ["", "This experiment isolates a change to spatial arrangement, but does not prove locality is the only cause of a model's performance. Pooling, finite capacity, optimisation and early stopping can also affect sensitivity. CNN and MLP capacity are of similar order; consult learning curves for convergence. Learning rates selected on original validation data are deliberately held fixed across conditions.", "", f"Original/permuted prediction agreement controls: {control_agreement}. The centroid is invariant in exact arithmetic. Coordinate-matched dense models should be close; numerical differences may compound during training.", "", "Paired test-group bootstrap intervals and paired seed results are in ablation_analysis.json and paired_ablation.csv. Test-group intervals are conditional on these trained models; only three training seeds were used.", "", "## Limitations", "", audit["limitations"], "", "Fashion-MNIST consists of small centred grayscale product images. These findings should not be generalised directly to colour images, unconstrained detection, video, or image generation.", "", "## Reproduction", "", "Run `python src/study.py all --device cuda` in the pinned environment. Completed immutable runs are reused; stale protocol/code hashes are rejected. A frozen checkpoint manifest is written before test evaluation.", ""]
    (reports / "results.md").write_text("\n".join(lines))
    print((reports / "results.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
