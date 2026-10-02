# Dataset audit, validation-only model selection, paired ablation and sealed test.

from __future__ import annotations

import argparse
import collections
import copy
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import platform
import random
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
SOURCE = "https://github.com/zalandoresearch/fashion-mnist"
RAW_SOURCE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def config():
    return read_json(ROOT / "config/protocol.json")


def source_hash():
    return digest(__file__)


def raw_images(name):
    with gzip.open(ROOT / "data/raw" / name, "rb") as f:
        content = f.read()
    if "images" in name:
        magic, n, rows, cols = struct.unpack(">IIII", content[:16])
        if magic != 2051 or (rows, cols) != (28, 28):
            raise ValueError("Invalid image IDX header")
        return np.frombuffer(content[16:], dtype=np.uint8).reshape(n, rows, cols).copy()
    magic, n = struct.unpack(">II", content[:8])
    if magic != 2049:
        raise ValueError("Invalid label IDX header")
    return np.frombuffer(content[8:], dtype=np.uint8).reshape(n).copy()


def image_hashes(x):
    return [hashlib.sha256(a.tobytes()).hexdigest() for a in x]


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def duplicate_groups(images, rules):
# Exhaustive for fixed raw-pixel MAE/RMSE thresholds, not semantic similarity.
    
    hashes = image_hashes(images)
    uf = UnionFind(len(images))
    first, exact_pairs = {}, []
    for i, h in enumerate(hashes):
        if h in first:
            uf.union(first[h], i)
            exact_pairs.append((first[h], i))
        else:
            first[h] = i
    coarse = images.reshape(-1, 4, 7, 4, 7).mean(axis=(2, 4)).reshape(-1, 16)
    norms = np.square(coarse).sum(1)
    flat = images.reshape(-1, 784)
    near_pairs = []
    radius2 = 16 * rules["max_root_mean_squared_difference_0_255"] ** 2
    for first_index in range(0, len(images), 256):
        block = coarse[first_index:first_index + 256]
        distances = norms[first_index:first_index + len(block), None] + norms[None, first_index:] - 2 * (block @ coarse[first_index:].T)
        aa, bb = np.where(distances <= radius2 + 1e-5)
        for offset in range(len(block)):
            a = first_index + offset
            candidates = bb[aa == offset] + first_index
            candidates = [int(b) for b in candidates if b > a and hashes[a] != hashes[b]]
            for begin in range(0, len(candidates), 2048):
                ids = candidates[begin:begin + 2048]
                diff = flat[ids].astype(np.float64) - flat[a].astype(np.float64)
                mae = np.abs(diff).mean(1)
                rmse = np.sqrt(np.square(diff).mean(1))
                valid = (mae <= rules["max_mean_absolute_difference_0_255"]) & (rmse <= rules["max_root_mean_squared_difference_0_255"])
                for k in np.flatnonzero(valid):
                    uf.union(a, ids[k])
                    near_pairs.append((a, ids[k], float(mae[k]), float(rmse[k])))
    groups = np.array([uf.find(i) for i in range(len(images))])
    return hashes, groups, exact_pairs, near_pairs, []


def prepare():
    cfg = config()
    rawdir, prepared = ROOT / "data/raw", ROOT / "data/prepared"
    rawdir.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    if (prepared / "audit.json").exists():
        audit = read_json(prepared / "audit.json")
        if audit["protocol_sha256"] != digest(ROOT / "config/protocol.json"):
            raise RuntimeError("Protocol changed after preparation. Use a new output/data version.")
        print("Prepared data already exists and protocol matches.", flush=True)
        return
    downloads = []
    for name, expected in FILES.items():
        path = rawdir / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            temporary = path.with_suffix(".part")
            urllib.request.urlretrieve(RAW_SOURCE + name, temporary)
            temporary.replace(path)
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        downloads.append({"file": name, "url": RAW_SOURCE + name, "md5": actual, "sha256": digest(path), "verified_utc": stamp()})
    x = raw_images("train-images-idx3-ubyte.gz")
    y = raw_images("train-labels-idx1-ubyte.gz")
    tx = raw_images("t10k-images-idx3-ubyte.gz")
    ty = raw_images("t10k-labels-idx1-ubyte.gz")
    assert x.shape == (60000, 28, 28) and tx.shape == (10000, 28, 28)
    assert np.array_equal(np.unique(y), np.arange(10)) and np.array_equal(np.unique(ty), np.arange(10))
    allx = np.concatenate([x, tx])
    print("Auditing all exact and threshold-defined near duplicates before splitting.", flush=True)
    hashes, groups, exact, near, skipped = duplicate_groups(allx, cfg["near_duplicate_rule"])
    test_groups = set(groups[len(x):].tolist())
    excluded_test = [i for i in range(len(x)) if groups[i] in test_groups]
    train_groups = collections.defaultdict(list)
    for i in range(len(x)):
        if groups[i] not in test_groups:
            train_groups[int(groups[i])].append(i)
    retained, excluded_repeated, conflicting = [], [], []
    for ids in train_groups.values():
        # A single representative per detected group makes group-disjoint splitting explicit.
        if len(set(y[ids].tolist())) > 1:
            conflicting.extend(ids)
        else:
            retained.append(ids[0])
            excluded_repeated.extend(ids[1:])
    retained = np.array(sorted(retained))
    train_ids, val_ids = train_test_split(retained, test_size=cfg["validation_fraction"], stratify=y[retained], random_state=cfg["split_seed"])
    assert not set(groups[train_ids]) & set(groups[val_ids])
    assert not (set(groups[train_ids]) | set(groups[val_ids])) & test_groups
    permutation = np.random.default_rng(cfg["permutation_seed"]).permutation(784)
    np.savez_compressed(prepared / "train_val.npz", x_train=x[train_ids], y_train=y[train_ids], x_val=x[val_ids], y_val=y[val_ids], train_ids=train_ids, val_ids=val_ids, permutation=permutation)
    # The training code never opens this file. Audit uses test images for deduplication,
    # and labels only for class counts, not hyperparameter or hypothesis selection.
    np.savez_compressed(prepared / "test.npz", x_test=tx, y_test=ty, official_ids=np.arange(len(tx)), groups=groups[len(x):])
    write_json(prepared / "exclusions.json", {"training_indices_matching_test_groups": excluded_test, "repeated_training_group_members": excluded_repeated, "conflicting_training_group_members": conflicting})
    auditdir = ROOT / "reports/data_audit"
    auditdir.mkdir(parents=True, exist_ok=True)
    with (auditdir / "detected_near_pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["combined_index_a", "combined_index_b", "mean_absolute_pixel_difference", "root_mean_squared_pixel_difference"])
        writer.writerows(near)
    with (auditdir / "exact_duplicate_pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["combined_index_a", "combined_index_b"])
        writer.writerows(exact)
    audit = {
        "created_utc": stamp(), "protocol_sha256": digest(ROOT / "config/protocol.json"),
        "source": SOURCE, "licence": "MIT", "files": downloads,
        "original_counts": {"train": len(x), "test": len(tx)},
        "final_counts": {"train": len(train_ids), "validation": len(val_ids), "test": len(tx)},
        "class_names": CLASSES, "image_shape": [28, 28, 1], "features": 784,
        "class_counts": {"train": np.bincount(y[train_ids], minlength=10).tolist(), "validation": np.bincount(y[val_ids], minlength=10).tolist(), "test": np.bincount(ty, minlength=10).tolist()},
        "missing_or_nonfinite_pixels": 0, "invalid_labels": 0,
        "exact_repeated_images": len(exact), "detected_near_pairs": len(near),
        "training_images_removed_due_to_test_overlap": len(excluded_test),
        "training_images_removed_as_group_repeats": len(excluded_repeated),
        "training_images_removed_in_label_conflict_groups": len(conflicting),
        "near_duplicate_rule": cfg["near_duplicate_rule"],
        "skipped_large_near_hash_buckets": len(skipped), "images_in_skipped_buckets": sum(skipped),
        "limitations": "The scan covers every pair satisfying raw-pixel MAE <= 2 and RMSE <= 6. It does not detect all transformed or semantic duplicates. Official test images remain unchanged, including repeats. This is a corrective rerun after v1 test results were inspected. The hypothesis, models, seeds and learning-rate grid were retained. Corrected-run checkpoints were frozen before corrected-run evaluation, but the official test set is not a newly unseen holdout.",
        "audit_revision": cfg.get("audit_revision"),
        "split_seed": cfg["split_seed"], "permutation_seed": cfg["permutation_seed"],
        "train_val_sha256": digest(prepared / "train_val.npz"), "test_sha256": digest(prepared / "test.npz")
    }
    write_json(prepared / "audit.json", audit)
    write_json(auditdir / "audit.json", audit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.4))
    for c in range(10):
        sample = x[train_ids[np.flatnonzero(y[train_ids] == c)[0]]]
        axes[0, c].imshow(sample, cmap="gray", vmin=0, vmax=255)
        axes[1, c].imshow(sample.reshape(-1)[permutation].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        axes[0, c].set_title(CLASSES[c], fontsize=8)
        axes[0, c].axis("off")
        axes[1, c].axis("off")
    fig.suptitle("Original images and the shared fixed pixel permutation")
    fig.tight_layout()
    fig.savefig(auditdir / "original_and_permuted.png", dpi=180)
    plt.close(fig)
    print(json.dumps(audit["final_counts"]), flush=True)


class Centroid(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("centres", torch.zeros(10, 784))

    def forward(self, x):
        flat = x.flatten(1)
        return 2 * flat @ self.centres.T - self.centres.square().sum(1)


def make_model(name, permutation=None):
    if name == "centroid":
        return Centroid()
    if name == "logistic":
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    elif name == "mlp":
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
    elif name == "cnn":
        model = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))
    else:
        raise ValueError(name)
    if permutation is not None and name in ("logistic", "mlp"):
        # Coordinate-matched initialisation makes the dense models a strong
        # control: identical information and paired optimisation up to rounding.
        with torch.no_grad():
            model[1].weight.copy_(model[1].weight[:, permutation].clone())
    return model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inputs(x, condition, permutation, device):
    array = x.reshape(-1, 784)
    if condition == "permuted":
        array = array[:, permutation]
    # Identical fixed scaling for every method; no learned normalisation.
    return torch.tensor(array.copy(), dtype=torch.float32, device=device).reshape(-1, 1, 28, 28) / 255.0


@torch.no_grad()
def predict(model, x, y=None, batch_size=1024):
    model.eval()
    chunks, loss_sum = [], 0.0
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size])
        chunks.append(logits.argmax(1).cpu().numpy())
        if y is not None:
            loss_sum += nn.functional.cross_entropy(logits, y[i:i + batch_size], reduction="sum").item()
    labels = np.concatenate(chunks)
    return labels, (loss_sum / len(x) if y is not None else None)


def device_info(device):
    return {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(), "job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or os.environ.get("JOB_ID")}


def train_one(name, condition, seed, lr, outdir, data, device, cfg):
    outdir.mkdir(parents=True, exist_ok=True)
    done = outdir / "validation.json"
    if done.exists():
        result = read_json(done)
        if result["source_sha256"] != source_hash() or result["protocol_sha256"] != digest(ROOT / "config/protocol.json"):
            raise RuntimeError(f"Refusing to reuse stale run: {outdir}")
        return result
    seed_all(seed)
    permutation = data["permutation"]
    model = make_model(name, permutation if condition == "permuted" else None).to(device)
    x = inputs(data["x_train"], condition, permutation, device)
    vx = inputs(data["x_val"], condition, permutation, device)
    y = torch.tensor(data["y_train"].astype(np.int64), device=device)
    vy = torch.tensor(data["y_val"].astype(np.int64), device=device)
    # Warm up device/context before the measured fit. Evaluation is separate below.
    with torch.no_grad():
        model(x[:cfg["batch_size"]])
    sync(device)
    start = time.perf_counter()
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    if name == "centroid":
        with torch.no_grad():
            for c in range(10):
                model.centres[c].copy_(x[y == c].flatten(1).mean(0))
        sync(device)
        fit_seconds = time.perf_counter() - start
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs = 0
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            order = torch.randperm(len(x), generator=generator).to(device)
            loss_sum = 0.0
            for j in range(0, len(x), cfg["batch_size"]):
                idx = order[j:j + cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach().item() * len(idx)
            vp, vl = predict(model, vx, vy, cfg["evaluation_batch_size"])
            va = float(accuracy_score(data["y_val"], vp))
            history.append({"epoch": epoch, "train_loss": loss_sum / len(x), "validation_loss": vl, "validation_accuracy": va})
            if vl < best_loss - cfg["min_delta"]:
                best_loss, best_epoch, stale = vl, epoch, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch == 1 or epoch % 5 == 0 or stale >= cfg["patience"]:
                print(f"{name} {condition} seed={seed} lr={lr} epoch={epoch}: val_acc={va:.4f} val_loss={vl:.4f}", flush=True)
            if stale >= cfg["patience"]:
                break
        sync(device)
        fit_seconds = time.perf_counter() - start
        epochs = epoch
    model.load_state_dict(best_state)
    pred, val_loss = predict(model, vx, vy, cfg["evaluation_batch_size"])
    checkpoint = outdir / "model.pt"
    torch.save(best_state, checkpoint)
    np.save(outdir / "validation_predictions.npy", pred)
    write_json(outdir / "history.json", history)
    result = {"model": name, "condition": condition, "seed": seed, "learning_rate": lr, "validation_accuracy": float(accuracy_score(data["y_val"], pred)), "validation_macro_f1": float(f1_score(data["y_val"], pred, average="macro")), "validation_loss": val_loss, "fit_seconds_including_early_stopping_validation": fit_seconds, "epochs_run": epochs, "best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.parameters()), "stored_coefficients": sum(v.numel() for v in best_state.values()), "tensor_bytes": sum(v.numel() * v.element_size() for v in best_state.values()), "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_sha256": digest(checkpoint), "source_sha256": source_hash(), "protocol_sha256": digest(ROOT / "config/protocol.json"), "hardware": device_info(device), "completed_utc": stamp()}
    write_json(done, result)
    print(f"Completed {name}/{condition}/{seed}: validation accuracy {result['validation_accuracy']:.4f}", flush=True)
    return result


def load_training():
    audit = read_json(ROOT / "data/prepared/audit.json")
    if audit["protocol_sha256"] != digest(ROOT / "config/protocol.json"):
        raise RuntimeError("Prepared data/protocol mismatch")
    path = ROOT / "data/prepared/train_val.npz"
    if digest(path) != audit["train_val_sha256"]:
        raise RuntimeError("Prepared training data hash mismatch")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def train(device):
    cfg = config()
    data = load_training()
    base = ROOT / "outputs"
    lock = base / "selection.json"
    if lock.exists():
        selection = read_json(lock)
        if selection["source_sha256"] != source_hash() or selection["protocol_sha256"] != digest(ROOT / "config/protocol.json"):
            raise RuntimeError("Selection manifest is stale")
    else:
        selected, pilots = {"centroid": None}, []
        # Prespecified bounded search on ORIGINAL validation only. Two candidates
        # per trainable model. No test accuracy is loaded or computed here.
        for name in cfg["models"]:
            if name == "centroid":
                continue
            candidates = []
            for lr in cfg["learning_rates"][name]:
                row = train_one(name, "original", cfg["pilot_seed"], lr, base / "pilot" / name / str(lr), data, device, cfg)
                candidates.append(row)
                pilots.append(row)
            winner = min(candidates, key=lambda r: (-r["validation_accuracy"], r["validation_loss"], r["learning_rate"]))
            selected[name] = winner["learning_rate"]
        selection = {"created_utc": stamp(), "selected_learning_rates": selected, "pilot_fit_seconds": sum(r["fit_seconds_including_early_stopping_validation"] for r in pilots), "source_sha256": source_hash(), "protocol_sha256": digest(ROOT / "config/protocol.json"), "selection_rule": "Highest original validation accuracy at the minimum-validation-loss checkpoint; tie break loss then smaller learning rate. Same chosen learning rate in both conditions.", "test_used_for_selection": False}
        write_json(lock, selection)
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            for name in cfg["models"]:
                train_one(name, condition, seed, selection["selected_learning_rates"][name], base / "runs" / f"{name}_{condition}_{seed}", data, device, cfg)
    # Hash all selected checkpoints before the first final test evaluation.
    frozen = {"created_utc": stamp(), "source_sha256": source_hash(), "protocol_sha256": digest(ROOT / "config/protocol.json"), "selection_sha256": digest(lock), "checkpoints": {p.parent.name: digest(p) for p in sorted((base / "runs").glob("*/model.pt"))}, "hypothesis": cfg["hypothesis"], "test_evaluation_started": False}
    if len(frozen["checkpoints"]) != len(cfg["seeds"]) * len(cfg["conditions"]) * len(cfg["models"]):
        raise RuntimeError("Missing primary runs")
    freeze_path = base / "frozen_experiment.json"
    if freeze_path.exists():
        old = read_json(freeze_path)
        if old["checkpoints"] != frozen["checkpoints"]:
            raise RuntimeError("Checkpoint changed after experiment was frozen")
    else:
        write_json(freeze_path, frozen)


def evaluate(device):
    cfg = config()
    frozen = read_json(ROOT / "outputs/frozen_experiment.json")
    if frozen["source_sha256"] != source_hash() or frozen["protocol_sha256"] != digest(ROOT / "config/protocol.json"):
        raise RuntimeError("Code/protocol changed after model freeze")
    audit = read_json(ROOT / "data/prepared/audit.json")
    test_path = ROOT / "data/prepared/test.npz"
    if digest(test_path) != audit["test_sha256"]:
        raise RuntimeError("Test data hash mismatch")
    with np.load(test_path) as f:
        tx, ty = f["x_test"], f["y_test"]
    data = load_training()
    for run_name, expected in frozen["checkpoints"].items():
        outdir = ROOT / "outputs/runs" / run_name
        if digest(outdir / "model.pt") != expected:
            raise RuntimeError(f"Checkpoint changed: {run_name}")
        if (outdir / "test.json").exists():
            continue
        row = read_json(outdir / "validation.json")
        seed_all(row["seed"])
        model = make_model(row["model"]).to(device)
        model.load_state_dict(torch.load(outdir / "model.pt", map_location=device, weights_only=True))
        x = inputs(tx, row["condition"], data["permutation"], device)
        with torch.no_grad():
            model(x[:cfg["batch_size"]])
        sync(device)
        start = time.perf_counter()
        pred, _ = predict(model, x, batch_size=cfg["evaluation_batch_size"])
        sync(device)
        elapsed = time.perf_counter() - start
        result = {**row, "test_accuracy": float(accuracy_score(ty, pred)), "test_macro_f1": float(f1_score(ty, pred, average="macro")), "test_inference_seconds": elapsed, "confusion_matrix": confusion_matrix(ty, pred, labels=list(range(10))).tolist(), "evaluated_utc": stamp(), "test_sha256": audit["test_sha256"]}
        np.save(outdir / "test_predictions.npy", pred)
        write_json(outdir / "test.json", result)
        print(f"TEST {run_name}: accuracy={result['test_accuracy']:.4f} macroF1={result['test_macro_f1']:.4f}", flush=True)
    write_json(ROOT / "outputs/test_complete.json", {"completed_utc": stamp(), "frozen_manifest_sha256": digest(ROOT / "outputs/frozen_experiment.json"), "runs": len(frozen["checkpoints"])})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "train", "evaluate", "report", "all"])
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get("STUDY_CPU_THREADS", "4")))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; refusing silent CPU fallback")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(json.dumps(device_info(device)), flush=True)
    if args.command in ("prepare", "all"):
        prepare()
    if args.command in ("train", "all"):
        train(device)
    if args.command in ("evaluate", "all"):
        evaluate(device)
    if args.command in ("report", "all"):
        subprocess.run([sys.executable, str(ROOT / "src/report.py")], check=True)


if __name__ == "__main__":
    main()
