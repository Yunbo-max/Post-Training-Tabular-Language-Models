"""
True Membership Inference Attack (MIA) Evaluation for Synthetic Data

The existing privacy_attack.py is WRONG - it computes a distinguishability score
(can a classifier tell real from synthetic), NOT a true MIA.

True MIA (Stadler et al.):
1. Generator G trained on D_train (real.csv)
2. G produces synthetic dataset S
3. D_test (test.csv) was never seen by G
4. For each query record r, compute distance to nearest neighbor in S
5. Members (r in D_train) may have lower distance if G memorized them
6. Non-members (r in D_test) should have higher distance
7. MIA AUC = how well NN-distance separates members from non-members
8. AUC close to 0.5 = good privacy (can't distinguish members from non-members)
"""

import numpy as np
import torch
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="True Membership Inference Attack evaluation")
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--path', type=str, default=None, help='Path to synthetic data CSV')
parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for distance computation')

args = parser.parse_args()


def compute_nn_distances(query_th, reference_th, batch_size=100):
    """Compute L1 distance from each query record to nearest record in reference set."""
    distances = []
    n_query = query_th.shape[0]

    for i in range((n_query // batch_size) + 1):
        if i * batch_size >= n_query:
            break
        if i != (n_query // batch_size):
            batch = query_th[i * batch_size: (i + 1) * batch_size]
        else:
            batch = query_th[i * batch_size:]

        if batch.shape[0] == 0:
            continue

        # L1 distance to nearest neighbor: (batch_size, 1, features) - (1, ref_size, features)
        dist = (batch[:, None] - reference_th).abs().sum(dim=2).min(dim=1).values
        distances.append(dist)

    return torch.cat(distances)


if __name__ == '__main__':
    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path

    real_path = f'synthetic/{dataname}/real.csv'   # Members (used to train generator)
    test_path = f'synthetic/{dataname}/test.csv'   # Non-members (never seen by generator)
    data_dir = f'data/{dataname}'

    # Validate files exist
    for fpath, desc in [(syn_path, "Synthetic"), (real_path, "Real/Member"), (test_path, "Test/Non-member")]:
        if not os.path.exists(fpath):
            print(f"ERROR: {desc} data not found at {fpath}")
            sys.exit(1)

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)   # Members
    test_data = pd.read_csv(test_path)   # Non-members

    # Column setup (same pattern as eval_dcr.py)
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
    test_data.columns = list(np.arange(len(real_data.columns)))

    # Compute numerical ranges for normalization
    num_ranges = []
    for i in num_col_idx:
        r = real_data[i].max() - real_data[i].min()
        num_ranges.append(r if r > 0 else 1.0)
    num_ranges = np.array(num_ranges)

    # Separate numerical and categorical
    num_real = real_data[num_col_idx].to_numpy()
    cat_real = real_data[cat_col_idx].to_numpy().astype('str')
    num_syn = syn_data[num_col_idx].to_numpy()
    cat_syn = syn_data[cat_col_idx].to_numpy().astype('str')
    num_test = test_data[num_col_idx].to_numpy()
    cat_test = test_data[cat_col_idx].to_numpy().astype('str')

    # One-hot encode categorical (fit on real)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(cat_real)

    cat_real_oh = encoder.transform(cat_real)
    cat_syn_oh = encoder.transform(cat_syn)
    cat_test_oh = encoder.transform(cat_test)

    # Normalize numerical by range
    num_real_norm = num_real / num_ranges
    num_syn_norm = num_syn / num_ranges
    num_test_norm = num_test / num_ranges

    # Handle NaN/inf
    for arr in [num_real_norm, num_syn_norm, num_test_norm]:
        arr[~np.isfinite(arr)] = 0.0

    # Combine features
    real_np = np.concatenate([num_real_norm, cat_real_oh], axis=1)
    syn_np = np.concatenate([num_syn_norm, cat_syn_oh], axis=1)
    test_np = np.concatenate([num_test_norm, cat_test_oh], axis=1)

    # Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    real_th = torch.tensor(real_np, dtype=torch.float32).to(device)
    syn_th = torch.tensor(syn_np, dtype=torch.float32).to(device)
    test_th = torch.tensor(test_np, dtype=torch.float32).to(device)

    print(f"Data shapes - Members: {real_th.shape}, Non-members: {test_th.shape}, Synthetic: {syn_th.shape}")
    print(f"Device: {device}")

    # ========================================
    # TRUE MIA: Compare NN-distance to synthetic
    # for members (real/train) vs non-members (test)
    # ========================================

    print("\nComputing nearest-neighbor distances to synthetic data...")

    # Distance from each MEMBER record to nearest synthetic record
    member_distances = compute_nn_distances(real_th, syn_th, args.batch_size)

    # Distance from each NON-MEMBER record to nearest synthetic record
    nonmember_distances = compute_nn_distances(test_th, syn_th, args.batch_size)

    member_dist_np = member_distances.cpu().numpy()
    nonmember_dist_np = nonmember_distances.cpu().numpy()

    print(f"\nMember NN-distances:     mean={member_dist_np.mean():.4f}, median={np.median(member_dist_np):.4f}, std={member_dist_np.std():.4f}")
    print(f"Non-member NN-distances: mean={nonmember_dist_np.mean():.4f}, median={np.median(nonmember_dist_np):.4f}, std={nonmember_dist_np.std():.4f}")

    # Labels: 1=member, 0=non-member
    labels = np.concatenate([
        np.ones(len(member_dist_np)),
        np.zeros(len(nonmember_dist_np))
    ])

    # Scores: NEGATIVE distance (lower distance = more likely member = higher score)
    scores = np.concatenate([
        -member_dist_np,
        -nonmember_dist_np
    ])

    # Compute MIA AUC
    mia_auc = roc_auc_score(labels, scores)
    attack_advantage = abs(mia_auc - 0.5) * 2  # Ranges from 0 (no advantage) to 1 (perfect attack)

    # Find optimal threshold for precision/recall
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Optimal threshold: maximize Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    # Compute precision/recall at optimal threshold
    predictions = (scores >= best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)

    # Print results
    print(f"\n{'='*60}")
    print(f"TRUE MEMBERSHIP INFERENCE ATTACK RESULTS")
    print(f"{'='*60}")
    print(f"MIA AUC:            {mia_auc:.4f}  (0.5 = perfect privacy, 1.0 = full memorization)")
    print(f"Attack Advantage:   {attack_advantage:.4f}  (0.0 = no advantage, 1.0 = perfect attack)")
    print(f"{'='*60}")
    print(f"At optimal threshold:")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall:           {recall:.4f}")
    print(f"  F1:               {f1:.4f}")
    print(f"{'='*60}")

    # Parseable output
    print(f"mia_auc={np.round(mia_auc, 4)}")
    print(f"attack_advantage={np.round(attack_advantage, 4)}")
    print(f"mia_precision={np.round(precision, 4)}")
    print(f"mia_recall={np.round(recall, 4)}")
    print(f"mia_f1={np.round(f1, 4)}")

    # Save results
    if args.save_path:
        save_file = args.save_path
    else:
        save_dir = f'eval/mia_proper/{dataname}'
        os.makedirs(save_dir, exist_ok=True)
        save_file = f'{save_dir}/{model}.txt'

    print(f'\nSaving results to {save_file}')
    with open(save_file, 'w') as f:
        f.write(f'mia_auc={np.round(mia_auc, 4)}\n')
        f.write(f'attack_advantage={np.round(attack_advantage, 4)}\n')
        f.write(f'mia_precision={np.round(precision, 4)}\n')
        f.write(f'mia_recall={np.round(recall, 4)}\n')
        f.write(f'mia_f1={np.round(f1, 4)}\n')
        f.write(f'member_dist_mean={np.round(member_dist_np.mean(), 4)}\n')
        f.write(f'nonmember_dist_mean={np.round(nonmember_dist_np.mean(), 4)}\n')
        f.write(f'member_dist_median={np.round(np.median(member_dist_np), 4)}\n')
        f.write(f'nonmember_dist_median={np.round(np.median(nonmember_dist_np), 4)}\n')
