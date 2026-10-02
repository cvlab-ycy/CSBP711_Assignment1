import numpy as np
import torch
from src.study import Centroid, config, duplicate_groups, make_model, seed_all, train_one


def test_fixed_permutation_preserves_pixels_and_dense_initial_predictions():
    permutation = np.random.default_rng(711).permutation(784)
    x = torch.rand(8, 1, 28, 28)
    xp = x.flatten(1)[:, permutation].reshape(-1, 1, 28, 28)
    assert torch.equal(x.flatten(1).sort(1).values, xp.flatten(1).sort(1).values)
    for name in ["logistic", "mlp"]:
        seed_all(42)
        original = make_model(name)
        seed_all(42)
        permuted = make_model(name, permutation)
        torch.testing.assert_close(original(x), permuted(xp), atol=2e-6, rtol=2e-5)


def test_centroid_is_coordinate_permutation_invariant():
    p = np.random.default_rng(711).permutation(784)
    seed_all(42)
    a, b = Centroid(), Centroid()
    a.centres.copy_(torch.rand(10, 784))
    b.centres.copy_(a.centres[:, p])
    x = torch.rand(12, 1, 28, 28)
    xp = x.flatten(1)[:, p].reshape(-1, 1, 28, 28)
    torch.testing.assert_close(a(x), b(xp), atol=1e-4, rtol=1e-5)
    assert torch.equal(a(x).argmax(1), b(xp).argmax(1))


def test_duplicate_grouping_is_label_independent_and_links_near_pairs():
    x = np.zeros((4, 28, 28), dtype=np.uint8)
    x[0, 8:20, 8:20] = 180
    x[1] = x[0]
    x[2] = x[0]
    x[2, 12, 12] += 1
    x[3, :, :14] = 200
    rules = {"maximum_bucket_size": 200, "max_mean_absolute_difference_0_255": 2, "max_root_mean_squared_difference_0_255": 6}
    _, groups, exact, near, _ = duplicate_groups(x, rules)
    assert len(exact) == 1 and len(near) == 2
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_all_four_models_fit_and_save_without_test_data(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    data = {"x_train": rng.integers(0, 256, (40, 28, 28), dtype=np.uint8),
            "y_train": np.tile(np.arange(10), 4),
            "x_val": rng.integers(0, 256, (20, 28, 28), dtype=np.uint8),
            "y_val": np.tile(np.arange(10), 2),
            "permutation": rng.permutation(784)}
    cfg = config()
    cfg.update(max_epochs=2, patience=2, batch_size=20)
    for name in cfg["models"]:
        out = tmp_path / name
        result = train_one(name, "permuted", 42, 0.001, out, data, torch.device("cpu"), cfg)
        assert 0 <= result["validation_accuracy"] <= 1
        assert result["tensor_bytes"] > 0
        assert (out / "model.pt").exists()
        assert not (out / "test.json").exists()
        saved = torch.load(out / "model.pt", weights_only=True)
        make_model(name).load_state_dict(saved)


def test_exhaustive_duplicate_scan_matches_bruteforce_across_hashes():
    rng = np.random.default_rng(711)
    base = rng.integers(10, 220, (12, 28, 28), dtype=np.uint8)
    copies = base.astype(np.int16) + rng.integers(-2, 3, base.shape)
    x = np.concatenate([base, copies.astype(np.uint8)])
    rules = {"max_mean_absolute_difference_0_255": 2, "max_root_mean_squared_difference_0_255": 6}
    _, _, _, near, skipped = duplicate_groups(x, rules)
    brute = set()
    for a in range(len(x)):
        for b in range(a+1, len(x)):
            diff = x[a].astype(float)-x[b].astype(float)
            if np.abs(diff).mean() <= 2 and np.sqrt(np.square(diff).mean()) <= 6:
                brute.add((a,b))
    assert {(a,b) for a,b,_,_ in near} == brute
    assert len(brute) == 12
    assert skipped == []
