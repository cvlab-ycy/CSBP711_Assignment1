"""Check every pair satisfying the fixed raw-pixel near-duplicate thresholds.

A 4x4 grid of block means gives a lower bound on full-image RMSE (Jensen's
inequality). A blockwise distance scan therefore safely rejects impossible pairs,
then the exact 784-pixel MAE/RMSE verifies all candidates. There is no hash filter
or bucket-size cap. This is exhaustive for this numeric rule, not for arbitrary
semantic, rotated or translated duplicates.
"""
import csv
import datetime as dt
import gzip
import hashlib
import json
import struct
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_images(name):
    with gzip.open(ROOT / 'data/raw' / name, 'rb') as f:
        raw = f.read()
    magic, count, height, width = struct.unpack('>IIII', raw[:16])
    assert magic == 2051 and (height, width) == (28, 28)
    return np.frombuffer(raw[16:], np.uint8).reshape(count, 28, 28)


def main():
    started = time.perf_counter()
    destination = ROOT / 'reports/complete_duplicate_audit'
    destination.mkdir(parents=True, exist_ok=True)
    x = np.concatenate([load_images('train-images-idx3-ubyte.gz'), load_images('t10k-images-idx3-ubyte.gz')])
    flat = x.reshape(-1, 784)
    coarse = x.reshape(-1, 4, 7, 4, 7).mean(axis=(2, 4)).reshape(-1, 16)
    pairs, candidates = [], 0
    squared_norms = np.square(coarse).sum(1)
    for first in range(0, len(x), 256):
        block = coarse[first:first + 256]
        targets = coarse[first:]
        squared_distances = squared_norms[first:first+len(block), None] + squared_norms[None, first:] - 2 * (block @ targets.T)
        possible_a, possible_b = np.where(squared_distances <= 16 * 6.0**2 + 1e-5)
        for offset in range(len(block)):
            a = first + offset
            ids = possible_b[possible_a == offset] + first
            ids = ids[ids > a]
            candidates += len(ids)
            for start in range(0, len(ids), 2048):
                selected = ids[start:start + 2048]
                diff = flat[selected].astype(np.float32) - flat[a].astype(np.float32)
                mae = np.abs(diff).mean(1)
                rmse = np.sqrt(np.square(diff).mean(1))
                for k in np.flatnonzero((mae <= 2.0) & (rmse <= 6.0)):
                    pairs.append([int(a), int(selected[k]), float(mae[k]), float(rmse[k])])
        if first % 5120 == 0:
            print(f'Checked {min(first+256,len(x))}/{len(x)} images; {candidates} candidates; {len(pairs)} pairs', flush=True)
    with np.load(ROOT / 'data/prepared/train_val.npz') as f:
        train, validation = set(f['train_ids'].tolist()), set(f['val_ids'].tolist())
    def split(i):
        if i >= 60000: return 'test'
        if i in train: return 'train'
        if i in validation: return 'validation'
        return 'excluded'
    crossing = []
    for a, b, mae, rmse in pairs:
        sa, sb = split(a), split(b)
        if sa != sb and 'excluded' not in (sa, sb):
            crossing.append([a, b, sa, sb, mae, rmse])
    with (destination / 'all_near_pairs.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['combined_index_a', 'combined_index_b', 'mean_absolute_difference', 'root_mean_squared_difference'])
        writer.writerows(pairs)
    with (destination / 'cross_split_pairs.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['combined_index_a', 'combined_index_b', 'split_a', 'split_b', 'mean_absolute_difference', 'root_mean_squared_difference'])
        writer.writerows(crossing)
    result = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'images_checked': len(x), 'total_possible_pairs': len(x)*(len(x)-1)//2, 'candidates_verified_at_full_resolution': candidates, 'near_pairs_detected': len(pairs), 'cross_split_pairs_in_existing_run': len(crossing), 'skipped_buckets': 0, 'mae_threshold_0_255': 2.0, 'rmse_threshold_0_255': 6.0, 'search': 'Exhaustive Euclidean distance scan in 16 block means followed by full-resolution verification. Jensen lower bound guarantees coverage under the stated pixel rule.', 'limitations': 'This rules out only pairs satisfying the stated raw-pixel thresholds. It does not guarantee absence of semantically related or geometrically transformed images.', 'elapsed_seconds': time.perf_counter()-started, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (destination/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
