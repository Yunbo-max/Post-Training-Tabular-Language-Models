"""
Non-classifier-based evaluation metrics for synthetic data quality.

Option 2: These metrics are completely independent of any classifier,
so they cannot be "gamed" by a generator trained against a classifier.

Metrics:
1. Wasserstein Distance (per-column average)
2. Maximum Mean Discrepancy (MMD) with RBF kernel
3. Jensen-Shannon Divergence (JSD, per-column average)
4. Nearest Neighbor Distance Ratio (NNDR)
5. DCR statistics (mean, median, 5th percentile)
"""

import numpy as np
import torch
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Non-classifier statistical distance metrics")
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--path', type=str, default=None, help='Path to synthetic data CSV')
parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
parser.add_argument('--batch_size', type=int, default=100)

args = parser.parse_args()


def compute_wasserstein(real_data, syn_data, num_col_idx, cat_col_idx):
    """Per-column Wasserstein distance, averaged."""
    distances = []

    # Numerical columns: direct Wasserstein
    for col in num_col_idx:
        real_col = real_data[col].dropna().values.astype(float)
        syn_col = syn_data[col].dropna().values.astype(float)
        if len(real_col) > 0 and len(syn_col) > 0:
            d = wasserstein_distance(real_col, syn_col)
            # Normalize by range
            col_range = real_col.max() - real_col.min()
            if col_range > 0:
                d = d / col_range
            distances.append(d)

    # Categorical columns: L1 between frequency distributions
    for col in cat_col_idx:
        real_counts = real_data[col].astype(str).value_counts(normalize=True)
        syn_counts = syn_data[col].astype(str).value_counts(normalize=True)
        all_cats = set(real_counts.index) | set(syn_counts.index)
        d = 0.0
        for cat in all_cats:
            d += abs(real_counts.get(cat, 0.0) - syn_counts.get(cat, 0.0))
        distances.append(d / 2.0)  # Normalize to [0, 1]

    return np.mean(distances) if distances else 0.0


def compute_jsd(real_data, syn_data, num_col_idx, cat_col_idx, n_bins=50):
    """Per-column Jensen-Shannon Divergence, averaged."""
    jsd_scores = []

    # Numerical columns: bin into histograms
    for col in num_col_idx:
        real_col = real_data[col].dropna().values.astype(float)
        syn_col = syn_data[col].dropna().values.astype(float)
        if len(real_col) == 0 or len(syn_col) == 0:
            continue
        # Common bin edges from real data
        col_min = min(real_col.min(), syn_col.min())
        col_max = max(real_col.max(), syn_col.max())
        if col_min == col_max:
            jsd_scores.append(0.0)
            continue
        bins = np.linspace(col_min, col_max, n_bins + 1)
        real_hist, _ = np.histogram(real_col, bins=bins, density=False)
        syn_hist, _ = np.histogram(syn_col, bins=bins, density=False)
        # Add smoothing to avoid zeros
        real_hist = real_hist.astype(float) + 1e-10
        syn_hist = syn_hist.astype(float) + 1e-10
        real_hist = real_hist / real_hist.sum()
        syn_hist = syn_hist / syn_hist.sum()
        jsd_scores.append(float(jensenshannon(real_hist, syn_hist) ** 2))  # Squared = divergence

    # Categorical columns: frequency distributions
    for col in cat_col_idx:
        real_counts = real_data[col].astype(str).value_counts(normalize=True)
        syn_counts = syn_data[col].astype(str).value_counts(normalize=True)
        all_cats = sorted(set(real_counts.index) | set(syn_counts.index))
        real_probs = np.array([real_counts.get(c, 0.0) for c in all_cats]) + 1e-10
        syn_probs = np.array([syn_counts.get(c, 0.0) for c in all_cats]) + 1e-10
        real_probs = real_probs / real_probs.sum()
        syn_probs = syn_probs / syn_probs.sum()
        jsd_scores.append(float(jensenshannon(real_probs, syn_probs) ** 2))

    return np.mean(jsd_scores) if jsd_scores else 0.0


def compute_mmd(real_np, syn_np, sigma=None, max_samples=2000):
    """MMD with RBF kernel using unbiased estimator."""
    # Subsample if too large
    if len(real_np) > max_samples:
        idx = np.random.RandomState(42).choice(len(real_np), max_samples, replace=False)
        real_np = real_np[idx]
    if len(syn_np) > max_samples:
        idx = np.random.RandomState(42).choice(len(syn_np), max_samples, replace=False)
        syn_np = syn_np[idx]

    X = torch.tensor(real_np, dtype=torch.float32)
    Y = torch.tensor(syn_np, dtype=torch.float32)

    # Auto sigma: median heuristic
    if sigma is None:
        combined = torch.cat([X, Y], dim=0)
        n_sub = min(500, len(combined))
        sub_idx = torch.randperm(len(combined))[:n_sub]
        sub = combined[sub_idx]
        dists = torch.cdist(sub, sub, p=2)
        sigma = dists.median().item()
        if sigma == 0:
            sigma = 1.0

    gamma = 1.0 / (2.0 * sigma ** 2)

    # Kernel matrices
    XX = torch.cdist(X, X, p=2).pow(2)
    YY = torch.cdist(Y, Y, p=2).pow(2)
    XY = torch.cdist(X, Y, p=2).pow(2)

    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)

    m = X.shape[0]
    n = Y.shape[0]

    # Unbiased MMD^2 estimator
    mmd2 = (K_XX.sum() - K_XX.trace()) / (m * (m - 1)) + \
           (K_YY.sum() - K_YY.trace()) / (n * (n - 1)) - \
           2 * K_XY.sum() / (m * n)

    return max(float(mmd2), 0.0)


def compute_nndr(syn_th, real_th, batch_size=100):
    """Nearest Neighbor Distance Ratio: dist_1st / dist_2nd for each synthetic record."""
    ratios = []
    n_syn = syn_th.shape[0]

    for i in range((n_syn // batch_size) + 1):
        if i * batch_size >= n_syn:
            break
        if i != (n_syn // batch_size):
            batch = syn_th[i * batch_size: (i + 1) * batch_size]
        else:
            batch = syn_th[i * batch_size:]

        if batch.shape[0] == 0:
            continue

        # L1 distances to all real records
        dists = (batch[:, None] - real_th).abs().sum(dim=2)  # (batch, n_real)

        # Top-2 nearest
        top2 = torch.topk(dists, k=min(2, dists.shape[1]), dim=1, largest=False)
        d1 = top2.values[:, 0]
        if top2.values.shape[1] >= 2:
            d2 = top2.values[:, 1]
            # Avoid division by zero
            ratio = d1 / (d2 + 1e-10)
        else:
            ratio = torch.ones_like(d1)

        ratios.append(ratio)

    ratios = torch.cat(ratios)
    return ratios.cpu().numpy()


def compute_dcr_stats(syn_th, real_th, batch_size=100):
    """DCR statistics: mean, median, 5th percentile of nearest-neighbor distances."""
    distances = []
    n_syn = syn_th.shape[0]

    for i in range((n_syn // batch_size) + 1):
        if i * batch_size >= n_syn:
            break
        if i != (n_syn // batch_size):
            batch = syn_th[i * batch_size: (i + 1) * batch_size]
        else:
            batch = syn_th[i * batch_size:]

        if batch.shape[0] == 0:
            continue

        dist = (batch[:, None] - real_th).abs().sum(dim=2).min(dim=1).values
        distances.append(dist)

    distances = torch.cat(distances).cpu().numpy()
    return {
        'mean': float(np.mean(distances)),
        'median': float(np.median(distances)),
        '5th_percentile': float(np.percentile(distances, 5)),
    }


if __name__ == '__main__':
    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path

    real_path = f'synthetic/{dataname}/real.csv'
    data_dir = f'data/{dataname}'

    for fpath, desc in [(syn_path, "Synthetic"), (real_path, "Real")]:
        if not os.path.exists(fpath):
            print(f"ERROR: {desc} data not found at {fpath}")
            sys.exit(1)

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    # Standardize column names
    real_data.columns = list(np.arange(len(real_data.columns)))
    syn_data.columns = list(np.arange(len(real_data.columns)))

    print(f"Real data: {real_data.shape}, Synthetic data: {syn_data.shape}")

    # ========================================
    # 1. Wasserstein Distance
    # ========================================
    print("\n[1/5] Computing Wasserstein Distance...")
    wd = compute_wasserstein(real_data, syn_data, num_col_idx, cat_col_idx)
    print(f"  Wasserstein Distance: {wd:.4f}")

    # ========================================
    # 2. Jensen-Shannon Divergence
    # ========================================
    print("[2/5] Computing Jensen-Shannon Divergence...")
    jsd = compute_jsd(real_data, syn_data, num_col_idx, cat_col_idx)
    print(f"  JSD: {jsd:.4f}")

    # ========================================
    # Prepare encoded data for MMD, NNDR, DCR
    # ========================================
    num_ranges = []
    for i in num_col_idx:
        r = real_data[i].max() - real_data[i].min()
        num_ranges.append(r if r > 0 else 1.0)
    num_ranges = np.array(num_ranges)

    num_real = real_data[num_col_idx].to_numpy() / num_ranges
    num_syn = syn_data[num_col_idx].to_numpy() / num_ranges
    cat_real = real_data[cat_col_idx].to_numpy().astype('str')
    cat_syn = syn_data[cat_col_idx].to_numpy().astype('str')

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(cat_real)
    cat_real_oh = encoder.transform(cat_real)
    cat_syn_oh = encoder.transform(cat_syn)

    for arr in [num_real, num_syn]:
        arr[~np.isfinite(arr)] = 0.0

    real_np = np.concatenate([num_real, cat_real_oh], axis=1)
    syn_np = np.concatenate([num_syn, cat_syn_oh], axis=1)

    # ========================================
    # 3. MMD
    # ========================================
    print("[3/5] Computing MMD...")
    mmd = compute_mmd(real_np, syn_np)
    print(f"  MMD: {mmd:.6f}")

    # ========================================
    # 4. NNDR
    # ========================================
    print("[4/5] Computing NNDR...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    real_th = torch.tensor(real_np, dtype=torch.float32).to(device)
    syn_th = torch.tensor(syn_np, dtype=torch.float32).to(device)

    nndr_values = compute_nndr(syn_th, real_th, args.batch_size)
    nndr_mean = float(np.mean(nndr_values))
    print(f"  NNDR mean: {nndr_mean:.4f}")

    # ========================================
    # 5. DCR statistics
    # ========================================
    print("[5/5] Computing DCR statistics...")
    dcr_stats = compute_dcr_stats(syn_th, real_th, args.batch_size)
    print(f"  DCR mean: {dcr_stats['mean']:.4f}")
    print(f"  DCR median: {dcr_stats['median']:.4f}")
    print(f"  DCR 5th percentile: {dcr_stats['5th_percentile']:.4f}")

    # ========================================
    # Print parseable results
    # ========================================
    print(f"\n{'='*60}")
    print(f"STATISTICAL DISTANCE METRICS")
    print(f"{'='*60}")
    print(f"wasserstein={np.round(wd, 4)}")
    print(f"mmd={np.round(mmd, 6)}")
    print(f"jsd={np.round(jsd, 4)}")
    print(f"nndr={np.round(nndr_mean, 4)}")
    print(f"dcr_mean={np.round(dcr_stats['mean'], 4)}")
    print(f"dcr_median={np.round(dcr_stats['median'], 4)}")
    print(f"dcr_5th={np.round(dcr_stats['5th_percentile'], 4)}")

    # Save results
    if args.save_path:
        save_file = args.save_path
    else:
        save_dir = f'eval/statistical/{dataname}'
        os.makedirs(save_dir, exist_ok=True)
        save_file = f'{save_dir}/{model}.txt'

    print(f'\nSaving results to {save_file}')
    with open(save_file, 'w') as f:
        f.write(f'wasserstein={np.round(wd, 4)}\n')
        f.write(f'mmd={np.round(mmd, 6)}\n')
        f.write(f'jsd={np.round(jsd, 4)}\n')
        f.write(f'nndr={np.round(nndr_mean, 4)}\n')
        f.write(f'dcr_mean={np.round(dcr_stats["mean"], 4)}\n')
        f.write(f'dcr_median={np.round(dcr_stats["median"], 4)}\n')
        f.write(f'dcr_5th={np.round(dcr_stats["5th_percentile"], 4)}\n')
