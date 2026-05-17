"""
Generate DPO pairs using DCR (Distance to Closest Record) as the reward signal.

This addresses reviewer concerns about evaluation circularity: if we train the
reward with a non-classifier signal (DCR), then classifier-based evaluation
metrics (C2ST, MIA) become fair and independent.

DCR reward logic:
  - Compute L1 distance from each synthetic sample to its nearest real training record
    (after one-hot encoding categoricals + normalizing numericals by range).
  - Samples with very small DCR are memorizing real data (bad privacy) -> low reward.
  - Samples with moderate DCR have good utility/privacy balance -> high reward.
  - Samples with very large DCR are off-distribution (poor utility) -> moderate reward.
  - Reward = min(dcr, threshold) / threshold, where threshold = median DCR of reference.
    This caps at 1.0 for samples far enough away, and linearly penalizes memorizers.
  - Additionally provide a rank-based percentile reward as an alternative column.

DPO pairs are formed by pairing highest-reward samples (chosen) with lowest-reward
samples (rejected), identical to the structure in generate_dpo_synthetic.py.
"""

import pandas as pd
import numpy as np
import json
import os
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')


def row_to_str(row):
    """Convert a pandas Series (row) to a readable string in GReaT format."""
    parts = []
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            val = "?"
        elif isinstance(val, (int, float)):
            if np.isinf(val):
                val = "?"
            else:
                val = str(val)
        else:
            val = str(val).strip()
        parts.append(f"{col} is {val}")
    return ', '.join(parts)


def detect_dataname(path):
    """Detect dataset name from a file path.

    Recognised path shapes:
      .../synthetic/<dataname>/<file>.csv
      .../data/<dataname>/<file>.csv
      .../experiments/drift_<NAME>/<run>/synthetic_data/iteration_K/<file>.csv
        -> returns "<NAME>_drift" (matches data/<NAME>_drift/info.json)
      .../experiments/unlearn/<scenario>_seedN/synthetic_data/iteration_K/<file>.csv
        -> falls back to scenario-aware unlearn naming
    """
    parts = os.path.normpath(path).split(os.sep)
    # Form 1: synthetic/<NAME> or data/<NAME>
    for anchor in ('synthetic', 'data'):
        if anchor in parts:
            i = parts.index(anchor)
            if i + 1 < len(parts):
                return parts[i + 1]
    # Form 2: experiments/drift_<NAME>/...
    for p in parts:
        if p.startswith('drift_') and p != 'drift_data':
            return p[len('drift_'):] + '_drift'
    # Legacy substring fallback
    for name in ['adult', 'diabetes', 'beijing', 'news', 'shoppers', 'default', 'magic',
                 'elec2', 'german_credit']:
        if name in path:
            return name
    return 'adult'  # last-resort fallback


def load_column_info(dataname):
    """Load column type information from data/<dataname>/info.json.

    Returns num_col_idx, cat_col_idx lists (with target merged appropriately).
    """
    info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataname, 'info.json')
    if not os.path.exists(info_path):
        return None, None, None
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = list(info['num_col_idx'])
    cat_col_idx = list(info['cat_col_idx'])
    target_col_idx = list(info['target_col_idx'])
    task_type = info.get('task_type', 'binclass')

    # Merge target into the appropriate group
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    return num_col_idx, cat_col_idx, info


def preprocess_for_dcr(real_df, syn_df, num_col_idx, cat_col_idx):
    """One-hot encode categoricals, normalize numericals by range, return numpy arrays.

    Follows the same preprocessing as eval/eval_dcr.py:
      - Numerical columns: divide by (max - min) computed on real data.
      - Categorical columns: one-hot encode (fit on real data).
      - Concatenate [numerical | categorical_one_hot].

    Returns:
        real_np, syn_np  (float64 numpy arrays)
    """
    # Work on copies with integer column names (matching eval_dcr.py convention)
    real = real_df.copy()
    syn = syn_df.copy()

    col_names = list(real.columns)

    # --- Numerical ---
    num_ranges = []
    real_num_list = []
    syn_num_list = []

    for idx in num_col_idx:
        col = col_names[idx]
        real_col = pd.to_numeric(real[col], errors='coerce').fillna(0.0).values.astype(np.float64)
        syn_col = pd.to_numeric(syn[col], errors='coerce').fillna(0.0).values.astype(np.float64)
        col_range = real_col.max() - real_col.min()
        if col_range == 0:
            col_range = 1.0  # avoid division by zero
        num_ranges.append(col_range)
        real_num_list.append(real_col / col_range)
        syn_num_list.append(syn_col / col_range)

    if real_num_list:
        real_num_np = np.column_stack(real_num_list)
        syn_num_np = np.column_stack(syn_num_list)
    else:
        real_num_np = np.zeros((len(real), 0))
        syn_num_np = np.zeros((len(syn), 0))

    # --- Categorical ---
    if cat_col_idx:
        cat_cols = [col_names[i] for i in cat_col_idx]
        real_cat_np = real[cat_cols].fillna('?').astype(str).values
        syn_cat_np = syn[cat_cols].fillna('?').astype(str).values

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(real_cat_np)

        real_cat_oh = encoder.transform(real_cat_np)
        syn_cat_oh = encoder.transform(syn_cat_np)
    else:
        real_cat_oh = np.zeros((len(real), 0))
        syn_cat_oh = np.zeros((len(syn), 0))

    real_np = np.concatenate([real_num_np, real_cat_oh], axis=1)
    syn_np = np.concatenate([syn_num_np, syn_cat_oh], axis=1)

    return real_np, syn_np


def compute_dcr(syn_np, real_np, batch_size=100):
    """Compute DCR (L1 distance to closest real record) for each synthetic sample.

    Uses torch for GPU acceleration when available. Chunks the *real* set as well
    to keep peak memory bounded (peak ~ batch_size * real_chunk * D floats).

    Returns:
        dcr_values  (1-D numpy array of length len(syn_np))
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    real_th = torch.tensor(real_np, dtype=torch.float32).to(device)
    syn_th = torch.tensor(syn_np, dtype=torch.float32).to(device)

    n_real = real_th.shape[0]
    feat_dim = real_th.shape[1]
    # Auto-shrink syn batch to keep peak memory under ~4 GB:
    #     bytes ~= batch_size * real_chunk * feat_dim * 4
    # Use real_chunk = min(n_real, 4000) and batch_size capped accordingly.
    real_chunk = min(n_real, 4000)
    target_bytes = 4 * 1024**3  # 4 GB
    max_batch = max(1, target_bytes // (4 * real_chunk * max(feat_dim, 1)))
    if batch_size > max_batch:
        batch_size = int(max_batch)

    dcr_list = []
    n_batches = (syn_th.shape[0] + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, syn_th.shape[0])
        batch = syn_th[start:end]
        # L1 distance to every real record, take min — chunk the real set so the
        # transient (batch, real_chunk, feat) tensor stays bounded.
        running_min = None
        for r0 in range(0, n_real, real_chunk):
            r1 = min(n_real, r0 + real_chunk)
            distances = (batch[:, None, :] - real_th[None, r0:r1, :]).abs().sum(dim=2)
            chunk_min = distances.min(dim=1).values
            running_min = chunk_min if running_min is None else torch.minimum(running_min, chunk_min)
            del distances
        dcr_list.append(running_min.cpu())

    dcr_values = torch.cat(dcr_list).numpy()
    return dcr_values


def compute_dcr_rewards(dcr_values, mode='threshold', polarity='privacy'):
    """Compute rewards from DCR values.

    polarity:
      'privacy'   — higher DCR (further from real) gets higher reward. Used
                    when DCR plays the role of an anti-memorization signal
                    (paper §5.2 reward comparison ablation).
      'proximity' — LOWER DCR (closer to target distribution) gets higher reward.
                    Used for drift adaptation (§5.5.1) where we want generated
                    samples to fall near the *target* distribution.

    Two modes (orthogonal to polarity):
      'threshold' (default):
          base   = min(dcr, threshold) / threshold       (in [0,1], 1.0 = far)
      'rank':
          base   = rank_percentile                       (in [0,1], 1.0 = far)

    Returns:
        rewards  (1-D numpy array)
    """
    if mode == 'threshold':
        threshold = np.median(dcr_values)
        if threshold == 0:
            threshold = np.mean(dcr_values)
        if threshold == 0:
            threshold = 1.0
        base = np.minimum(dcr_values, threshold) / threshold
    elif mode == 'rank':
        ranks = dcr_values.argsort().argsort()
        base = ranks.astype(np.float64) / (len(ranks) - 1) if len(ranks) > 1 else np.ones_like(ranks, dtype=np.float64)
    else:
        raise ValueError(f"Unknown reward mode: {mode}")
    if polarity == 'privacy':
        rewards = base
    elif polarity == 'proximity':
        rewards = 1.0 - base  # invert: closer to target = higher reward
    else:
        raise ValueError(f"Unknown reward polarity: {polarity}")

    return rewards


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DPO pairs using DCR (Distance to Closest Record) as the reward signal. "
                    "Avoids evaluation circularity: classifier-based metrics (C2ST, MIA) remain independent."
    )
    parser.add_argument('--synthetic', default='synthetic/adult/great.csv',
                        help='Path to synthetic data CSV file')
    parser.add_argument('--original', default='synthetic/adult/real.csv',
                        help='Path to original/training data CSV file')
    parser.add_argument('--train_json', default='train_data_synthetic.json',
                        help='Output train JSON file (contains ALL pairs)')
    parser.add_argument('--test_json', default='test_data_synthetic.json',
                        help='Output test JSON file (kept for compatibility)')
    parser.add_argument('--top_percent', type=float, default=50.0,
                        help='Select top X%% of pairs with highest chosen rewards (default: 50.0)')
    parser.add_argument('--use_existing_synthetic', action='store_true',
                        help='Use existing synthetic data instead of generating NEW samples')
    parser.add_argument('--model_type', type=str, default='great',
                        choices=['great', 'great_gemma'],
                        help='Model type: great or great_gemma')
    parser.add_argument('--reward_mode', type=str, default='threshold',
                        choices=['threshold', 'rank'],
                        help='DCR reward mode: threshold (default) or rank')
    parser.add_argument('--polarity', type=str, default='auto',
                        choices=['auto', 'privacy', 'proximity'],
                        help="Reward direction: 'privacy' = far-from-real is good; "
                             "'proximity' = close-to-target is good (drift adaptation); "
                             "'auto' = proximity if dataname ends in '_drift', else privacy.")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Banner
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"DPO PAIR GENERATION  (DCR Reward Signal)")
    print(f"{'='*60}")

    iter_match = re.search(r'iter_(\d+)', args.train_json)
    current_iter = int(iter_match.group(1)) if iter_match else 1
    print(f"Processing iteration: {current_iter}")
    print(f"Reward mode: {args.reward_mode}")

    # Validate
    if args.top_percent <= 0 or args.top_percent > 100:
        raise ValueError("--top_percent must be between 0 and 100")

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    synthetic_data = pd.read_csv(args.synthetic)
    original_data = pd.read_csv(args.original)

    print(f"Loaded {len(synthetic_data)} synthetic samples and {len(original_data)} original samples")
    print(f"DCR REWARD APPROACH (no classifier in the loop):")
    print(f"   1. Compute DCR for each synthetic sample to real training data")
    print(f"   2. Reward based on distance: penalise memorisers, reward moderate distance")
    print(f"   3. Create DPO pairs from ranked samples")
    print(f"   4. Classifier-based eval (C2ST, MIA) stays fully independent")

    # Clean data
    columns = original_data.columns
    synthetic_data = synthetic_data[columns]

    for col in synthetic_data.columns:
        if synthetic_data[col].dtype == "object":
            synthetic_data[col] = synthetic_data[col].fillna("?")
            synthetic_data[col] = synthetic_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    for col in original_data.columns:
        if original_data[col].dtype == "object":
            original_data[col] = original_data[col].fillna("?")
            original_data[col] = original_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # ---------------------------------------------------------------
    # 2. Detect column types for DCR preprocessing
    # ---------------------------------------------------------------
    dataname = detect_dataname(args.synthetic)
    num_col_idx, cat_col_idx, info = load_column_info(dataname)

    if num_col_idx is None:
        # Fallback: treat all numeric-looking columns as numerical, rest as categorical
        print(f"WARNING: Could not load info.json for '{dataname}'. Inferring column types.")
        num_col_idx = []
        cat_col_idx = []
        for i, col in enumerate(original_data.columns):
            if pd.api.types.is_numeric_dtype(original_data[col]):
                num_col_idx.append(i)
            else:
                cat_col_idx.append(i)
    else:
        print(f"Loaded column info for '{dataname}' from info.json")

    print(f"  Numerical columns ({len(num_col_idx)}): {num_col_idx}")
    print(f"  Categorical columns ({len(cat_col_idx)}): {cat_col_idx}")

    # ---------------------------------------------------------------
    # 3. Generate or load NEW synthetic samples
    # ---------------------------------------------------------------
    print(f"\nStep 1: Preparing synthetic samples for DCR scoring...")

    if args.use_existing_synthetic:
        print(f"  Using existing synthetic data (--use_existing_synthetic flag)")
        new_synthetic_data = synthetic_data.copy()
        print(f"  Using {len(new_synthetic_data)} existing synthetic samples")
    else:
        print(f"  Generating NEW synthetic data using the current model...")

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_path = f'{curr_dir}/baselines/{args.model_type}/ckpt/{dataname}/model.pt'

        # Import the appropriate model class
        if args.model_type == 'great':
            from baselines.great.models.great import GReaT
            from baselines.great.models.great_utils import _array_to_dataframe
            model_class = GReaT
            base_model_name = "distilgpt2"
        elif args.model_type == 'great_gemma':
            from baselines.great_gemma.models.great_gemma import GReaTGemma
            from baselines.great_gemma.models.great_utils import _array_to_dataframe
            model_class = GReaTGemma
            base_model_name = "google/gemma-2b"

        if not os.path.exists(model_save_path):
            print(f"  Model not found at {model_save_path}")
            print(f"  Falling back to existing synthetic data...")
            new_synthetic_data = synthetic_data.copy()
        else:
            print(f"  Loading model from: {model_save_path}")
            try:
                great = model_class(base_model_name)
                great.model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
                great.load_finetuned_model(model_save_path)

                df = _array_to_dataframe(original_data, columns=None)
                great._update_column_information(df)
                great._update_conditional_information(df, conditional_col=None)

                n_samples = len(original_data)
                print(f"  Sampling {n_samples} NEW synthetic samples...")

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                new_synthetic_data = great.sample(
                    n_samples,
                    k=100,
                    device=device,
                    temperature=0.7
                )
                print(f"  Generated {len(new_synthetic_data)} NEW synthetic samples")

                # Fallback: model produced too few valid rows (e.g. drift case)
                if len(new_synthetic_data) < max(50, n_samples // 100):
                    print(f"  ⚠️ Fresh sampling produced only {len(new_synthetic_data)} valid rows; "
                          f"falling back to existing synthetic data.")
                    new_synthetic_data = synthetic_data.copy()

                new_synthetic_path = args.synthetic.replace('.csv', '_new.csv')
                new_synthetic_data.to_csv(new_synthetic_path, index=False)
                print(f"  Saved NEW synthetic data to: {new_synthetic_path}")

            except Exception as e:
                print(f"  Error during NEW synthetic data generation: {e}")
                print(f"  Falling back to existing synthetic data...")
                new_synthetic_data = synthetic_data.copy()

    print(f"  Synthetic samples for scoring: {len(new_synthetic_data)}")

    # ---------------------------------------------------------------
    # 4. Compute DCR for each synthetic sample
    # ---------------------------------------------------------------
    print(f"\nStep 2: Computing DCR (Distance to Closest Record)...")

    # Preprocess both real and new-synthetic into aligned numeric arrays
    real_np, syn_np = preprocess_for_dcr(original_data, new_synthetic_data, num_col_idx, cat_col_idx)
    print(f"  Preprocessed arrays: real {real_np.shape}, synthetic {syn_np.shape}")

    dcr_values = compute_dcr(syn_np, real_np, batch_size=100)
    print(f"  DCR statistics:")
    print(f"    Min:    {dcr_values.min():.6f}")
    print(f"    Max:    {dcr_values.max():.6f}")
    print(f"    Mean:   {dcr_values.mean():.6f}")
    print(f"    Median: {np.median(dcr_values):.6f}")
    print(f"    Std:    {dcr_values.std():.6f}")

    # ---------------------------------------------------------------
    # 5. Compute rewards from DCR
    # ---------------------------------------------------------------
    # Resolve polarity
    if args.polarity == 'auto':
        _polarity = 'proximity' if str(dataname).endswith('_drift') else 'privacy'
    else:
        _polarity = args.polarity
    print(f"\nStep 3: Computing DCR rewards (mode={args.reward_mode}, polarity={_polarity})...")

    rewards = compute_dcr_rewards(dcr_values, mode=args.reward_mode, polarity=_polarity)

    # Also compute rank-based reward for reference (saved in CSV)
    rank_rewards = compute_dcr_rewards(dcr_values, mode='rank')

    # Sort by reward (highest first)
    sorted_indices = np.argsort(rewards)[::-1]

    print(f"  Reward statistics (higher = better privacy, moderate utility):")
    print(f"    Mean reward:   {rewards.mean():.4f}")
    print(f"    Best reward:   {rewards.max():.4f}")
    print(f"    Worst reward:  {rewards.min():.4f}")
    print(f"    Median reward: {np.median(rewards):.4f}")
    print(f"    Std reward:    {rewards.std():.4f}")

    # ---------------------------------------------------------------
    # 6. Convert synthetic rows to GReaT format strings
    # ---------------------------------------------------------------
    print(f"\nStep 4: Converting synthetic rows to GReaT format strings...")
    new_synthetic_outputs = []
    batch_size = 100
    num_batches = (len(new_synthetic_data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(new_synthetic_data))
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"  Processing batch {batch_idx + 1}/{num_batches} (rows {start_idx}-{end_idx})")
        for idx in range(start_idx, end_idx):
            row_str = row_to_str(new_synthetic_data.iloc[idx])
            new_synthetic_outputs.append(row_str)

    print(f"  Converted {len(new_synthetic_outputs)} rows to strings")

    # ---------------------------------------------------------------
    # 7. Save DCR analysis CSV and plots
    # ---------------------------------------------------------------
    pairs_dir = os.path.dirname(args.train_json)
    if pairs_dir and not os.path.exists(pairs_dir):
        os.makedirs(pairs_dir, exist_ok=True)

    # Extract iteration id
    base_name = os.path.splitext(os.path.basename(args.train_json))[0]
    iteration_match = base_name.split('_')
    iteration_id = "unknown"
    for i, part in enumerate(iteration_match):
        if part == "iter" and i + 1 < len(iteration_match):
            iteration_id = iteration_match[i + 1]
            break

    # Save CSV
    dcr_data = pd.DataFrame({
        'sample_id': range(len(dcr_values)),
        'dcr': dcr_values,
        'dcr_reward': rewards,
        'dcr_rank_reward': rank_rewards,
        'sample_rank': [int(np.where(sorted_indices == i)[0][0]) + 1 for i in range(len(sorted_indices))]
    })

    csv_filename = f"dcr_scores_rewards_iter_{iteration_id}.csv"
    csv_path = os.path.join(pairs_dir, csv_filename) if pairs_dir else csv_filename

    try:
        dcr_data.to_csv(csv_path, index=False)
        print(f"  Saved DCR scores and rewards to: {csv_path}")
    except Exception as e:
        print(f"  Warning: Failed to save CSV file: {e}")
        csv_path = None

    # --- Distribution plots ---
    print(f"  Creating distribution plots...")

    try:
        plt.style.use('default')
        try:
            sns.set_palette("husl")
        except Exception:
            pass

        # Plot 1: DCR Distribution and Reward Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # DCR histogram
        ax1.hist(dcr_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        median_dcr = np.median(dcr_values)
        ax1.axvline(x=median_dcr, color='red', linestyle='--', linewidth=2,
                     label=f'Median: {median_dcr:.3f}')
        ax1.axvline(x=dcr_values.mean(), color='orange', linestyle='-', linewidth=2,
                     label=f'Mean: {dcr_values.mean():.3f}')
        ax1.set_xlabel('DCR (L1 Distance to Closest Real Record)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Distribution of DCR Values\nIteration {iteration_id}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        stats_text = (f'N = {len(dcr_values)}\n'
                      f'Mean = {dcr_values.mean():.3f}\n'
                      f'Std = {dcr_values.std():.3f}\n'
                      f'Median = {median_dcr:.3f}')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Reward histogram
        ax2.hist(rewards, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=rewards.mean(), color='orange', linestyle='-', linewidth=2,
                     label=f'Mean: {rewards.mean():.3f}')
        reward_label = 'min(dcr, threshold)/threshold' if args.reward_mode == 'threshold' else 'rank percentile'
        ax2.set_xlabel(f'DCR Reward ({reward_label})', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Distribution of DCR Rewards\nIteration {iteration_id}', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        reward_stats_text = (f'N = {len(rewards)}\n'
                             f'Mean = {rewards.mean():.3f}\n'
                             f'Std = {rewards.std():.3f}\n'
                             f'Median = {np.median(rewards):.3f}')
        ax2.text(0.02, 0.98, reward_stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        plot_filename = f"dcr_distributions_iter_{iteration_id}.png"
        plot_path = os.path.join(pairs_dir, plot_filename) if pairs_dir else plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved DCR distribution plot to: {plot_path}")

        # Plot 2: DCR vs Reward scatter + box-by-quartile
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        scatter = ax1.scatter(dcr_values, rewards, alpha=0.6, c=rewards,
                              cmap='RdYlGn', s=20)
        ax1.set_xlabel('DCR (L1 Distance to Closest Real Record)', fontsize=12)
        ax1.set_ylabel('DCR Reward', fontsize=12)
        ax1.set_title(f'DCR vs Reward\nIteration {iteration_id}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=median_dcr, color='red', linestyle='--', alpha=0.7, label='Median DCR')
        ax1.legend()
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Reward Value', fontsize=10)

        # Box plot by reward quartiles
        reward_quartiles = np.percentile(rewards, [25, 50, 75])
        q1_mask = rewards <= reward_quartiles[0]
        q2_mask = (rewards > reward_quartiles[0]) & (rewards <= reward_quartiles[1])
        q3_mask = (rewards > reward_quartiles[1]) & (rewards <= reward_quartiles[2])
        q4_mask = rewards > reward_quartiles[2]

        quartile_data = [
            dcr_values[q1_mask],
            dcr_values[q2_mask],
            dcr_values[q3_mask],
            dcr_values[q4_mask]
        ]
        quartile_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']

        ax2.boxplot(quartile_data, labels=quartile_labels)
        ax2.set_xlabel('Reward Quartiles', fontsize=12)
        ax2.set_ylabel('DCR Value', fontsize=12)
        ax2.set_title(f'DCR by Reward Quartile\nIteration {iteration_id}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=median_dcr, color='red', linestyle='--', alpha=0.7, label='Median DCR')
        ax2.legend()

        plt.tight_layout()

        analysis_plot_filename = f"dcr_analysis_iter_{iteration_id}.png"
        analysis_plot_path = os.path.join(pairs_dir, analysis_plot_filename) if pairs_dir else analysis_plot_filename
        plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved DCR analysis plot to: {analysis_plot_path}")

    except Exception as e:
        print(f"  Warning: Failed to create plots: {e}")
        plot_path = None
        analysis_plot_path = None

    # ---------------------------------------------------------------
    # 8. Print top/bottom samples
    # ---------------------------------------------------------------
    print(f"\nTop 10 highest reward synthetic samples:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"  Rank {i+1}: reward={rewards[idx]:.4f}, dcr={dcr_values[idx]:.4f}")

    print(f"\nTop 10 lowest reward synthetic samples:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[-(i+1)]
        print(f"  Rank {i+1}: reward={rewards[idx]:.4f}, dcr={dcr_values[idx]:.4f}")

    # ---------------------------------------------------------------
    # 9. Create DPO preference pairs
    # ---------------------------------------------------------------
    dpo_data = []
    num_pairs = len(sorted_indices) // 2

    print(f"\nStep 5: Creating {num_pairs} DPO preference pairs...")
    print(f"  Chosen (highest DCR reward) vs Rejected (lowest DCR reward)")

    for i in range(num_pairs):
        chosen_idx = sorted_indices[i]
        rejected_idx = sorted_indices[-(i+1)]

        entry = {
            "instruction": "",
            "input": "",
            "chosen": new_synthetic_outputs[chosen_idx],
            "rejected": new_synthetic_outputs[rejected_idx],
            "chosen_source": "new_synthetic",
            "rejected_source": "new_synthetic",
            "chosen_dcr": float(dcr_values[chosen_idx]),
            "rejected_dcr": float(dcr_values[rejected_idx]),
            "chosen_reward": float(rewards[chosen_idx]),
            "rejected_reward": float(rewards[rejected_idx])
        }
        dpo_data.append(entry)

    # Handle odd number of samples
    if len(sorted_indices) % 2 == 1:
        mid_idx = sorted_indices[num_pairs]
        entry = {
            "instruction": "",
            "input": "",
            "chosen": new_synthetic_outputs[mid_idx],
            "rejected": new_synthetic_outputs[mid_idx],
            "chosen_source": "new_synthetic",
            "rejected_source": "new_synthetic",
            "chosen_dcr": float(dcr_values[mid_idx]),
            "rejected_dcr": float(dcr_values[mid_idx]),
            "chosen_reward": float(rewards[mid_idx]),
            "rejected_reward": 0.0
        }
        dpo_data.append(entry)

    # ---------------------------------------------------------------
    # 10. Filter top X% of pairs
    # ---------------------------------------------------------------
    total_pairs_before = len(dpo_data)
    if args.top_percent < 100.0:
        dpo_data.sort(key=lambda x: x['chosen_reward'], reverse=True)
        num_pairs_to_keep = int(len(dpo_data) * args.top_percent / 100.0)
        dpo_data = dpo_data[:num_pairs_to_keep]

        print(f"\n  Filtered to top {args.top_percent}% of pairs (by chosen reward):")
        print(f"    Total pairs before: {total_pairs_before}")
        print(f"    Pairs kept: {len(dpo_data)}")
        print(f"    Pairs removed: {total_pairs_before - len(dpo_data)}")
        if dpo_data:
            print(f"    Best chosen reward: {dpo_data[0]['chosen_reward']:.4f}")
            print(f"    Worst (of kept) chosen reward: {dpo_data[-1]['chosen_reward']:.4f}")

    # ---------------------------------------------------------------
    # 11. Save DPO JSON
    # ---------------------------------------------------------------
    train_data = dpo_data

    print(f"\nSaving DPO data:")
    print(f"  Train pairs: {len(train_data)}")
    print(f"  Output file: {args.train_json}")

    with open(args.train_json, "w") as f:
        json.dump(train_data, f, indent=2)

    print(f"\n  Saved {args.train_json}")
    if args.top_percent < 100.0:
        print(f"  Filtered to top {args.top_percent}% of pairs with highest chosen rewards")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Summary of Generated Files:")
    print(f"{'='*60}")
    print(f"  DPO Training Data:")
    print(f"    - {args.train_json} ({len(train_data)} pairs)")

    csv_created = csv_path is not None
    plots_created = 'plot_path' in dir() and plot_path is not None
    analysis_created = 'analysis_plot_path' in dir() and analysis_plot_path is not None

    if csv_created or plots_created or analysis_created:
        print(f"  DCR Analysis Data:")
        if csv_created:
            print(f"    - {csv_path} (DCR scores & rewards for {len(dcr_data)} samples)")
        if plots_created:
            print(f"    - {plot_path} (DCR distribution plots)")
        if analysis_created:
            print(f"    - {analysis_plot_path} (DCR analysis plots)")

    print(f"\nDCR Reward Methodology:")
    if args.reward_mode == 'threshold':
        print(f"  reward = min(dcr, median_dcr) / median_dcr")
        print(f"  - Memorisers (dcr ~ 0): reward ~ 0 (penalised)")
        print(f"  - Moderate distance: reward scales linearly up to 1.0")
        print(f"  - Far samples (dcr > median): reward capped at 1.0 (good privacy)")
        print(f"  - Threshold (median DCR): {np.median(dcr_values):.4f}")
    else:
        print(f"  reward = rank_percentile (0 to 1)")
        print(f"  - Smallest DCR gets rank 0 (worst, memoriser)")
        print(f"  - Largest DCR gets rank 1 (best, most private)")

    print(f"\nKey advantage: DCR reward is independent of any classifier.")
    print(f"  Classifier-based evaluation metrics (C2ST, MIA) remain fair and unbiased.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
