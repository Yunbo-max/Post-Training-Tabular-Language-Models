"""
Generate DPO pairs using HOLDOUT split approach to address evaluation circularity.

Option 3: Training with a classifier on one data split, evaluating with a completely
separate classifier on a different split.

HOLDOUT APPROACH:
1. Split real data into two halves: split_A (for training reward) and split_B (reserved for evaluation)
2. Train RF classifier on split_A (real_A=1 vs synthetic=0) -> use this for DPO reward
3. Save split_B information so evaluation can use it independently
4. The evaluation side (eval/eval_holdout_classifier.py) trains its OWN classifier on split_B

This addresses reviewer concerns about evaluation circularity by ensuring the evaluation
classifier is independent of the training reward classifier.
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from rdt import HyperTransformer
from rdt.transformers import OneHotEncoder, GaussianNormalizer
from xgboost import XGBClassifier
import pickle
import re

try:
    from tabpfn import TabPFNClassifier  # not used here; tolerate missing torch>=2.3
except Exception:
    TabPFNClassifier = None

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


def update_config(config):
    """Update HyperTransformer config (same as privacy_attack.py)"""
    for col, sdtype in config["sdtypes"].items():
        if sdtype == "numerical":
            config["transformers"][col] = GaussianNormalizer(distribution="norm")
        if sdtype == "categorical":
            config["transformers"][col] = OneHotEncoder()
    return config


def preprocess_data(original_data, synthetic_data, hypertransformer=None):
    """Preprocess data using HyperTransformer (same as privacy_attack.py)"""
    combined_data = pd.concat([original_data, synthetic_data])

    if hypertransformer is None:
        ht = HyperTransformer()
        ht.detect_initial_config(data=combined_data)
        config = ht.get_config()
        config = update_config(config)
        ht.set_config(config)
        ht.fit(combined_data)
    else:
        ht = hypertransformer

    combined_data_transformed = ht.transform(combined_data)
    original_data_transformed = combined_data_transformed[:len(original_data)]
    synthetic_data_transformed = combined_data_transformed[len(original_data):]

    return original_data_transformed, synthetic_data_transformed, ht


def prepare_attack_data(original_data, synthetic_data):
    """Prepare MIA attack data (same as privacy_attack.py)"""
    original_data["member"] = 1
    synthetic_data["member"] = 0
    attack_data = pd.concat([original_data, synthetic_data])
    X = attack_data.drop("member", axis=1).values
    y = attack_data["member"].values
    return X, y


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DPO pairs using HOLDOUT split approach: "
                    "train reward classifier on split_A, reserve split_B for independent evaluation"
    )
    parser.add_argument('--synthetic', default='synthetic/adult/great.csv',
                        help='Path to synthetic data CSV file')
    parser.add_argument('--original', default='synthetic/adult/real.csv',
                        help='Path to original/training data CSV file')
    parser.add_argument('--train_json', default='train_data_synthetic.json',
                        help='Output train JSON file (contains ALL pairs)')
    parser.add_argument('--test_json', default='test_data_synthetic.json',
                        help='Output test JSON file (kept for compatibility but not used)')
    parser.add_argument('--top_percent', type=float, default=50.0,
                        help='Select top X percent of pairs with highest chosen rewards (default: 50.0)')
    parser.add_argument('--use_existing_synthetic', action='store_true',
                        help='Use existing synthetic data instead of generating NEW samples')
    parser.add_argument('--model_type', type=str, default='great',
                        choices=['great', 'great_gemma'],
                        help='Model type: great or great_gemma')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                        help='Fraction of real data for training reward classifier (split_A). '
                             'Remainder goes to split_B for evaluation. (default: 0.5)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducible splits (default: 42)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"DPO PAIR GENERATION - HOLDOUT SPLIT APPROACH")
    print(f"{'='*60}")

    # Extract and display iteration number
    iter_match = re.search(r'iter_(\d+)', args.train_json)
    current_iter = int(iter_match.group(1)) if iter_match else 1
    print(f"Processing iteration: {current_iter}")

    # Validate arguments
    if args.top_percent <= 0 or args.top_percent > 100:
        raise ValueError("--top_percent must be between 0 and 100")
    if args.split_ratio <= 0 or args.split_ratio >= 1:
        raise ValueError("--split_ratio must be between 0 and 1 (exclusive)")

    synthetic_data = pd.read_csv(args.synthetic)
    original_data = pd.read_csv(args.original)

    print(f"Loaded {len(synthetic_data)} synthetic samples and {len(original_data)} original samples")
    print(f"\nHOLDOUT SPLIT APPROACH:")
    print(f"   1. Split real data into split_A ({args.split_ratio*100:.0f}%) and split_B ({(1-args.split_ratio)*100:.0f}%)")
    print(f"   2. Train MIA classifier on split_A real + synthetic (for DPO reward)")
    print(f"   3. Reserve split_B for INDEPENDENT evaluation classifier")
    print(f"   4. Generate NEW synthetic samples and score with split_A classifier")
    print(f"   5. Save split_B indices so eval script can train its own classifier")

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

    # =====================================================
    # HOLDOUT SPLIT: Divide real data into split_A and split_B
    # =====================================================
    print(f"\nSplitting real data with random_state={args.random_state}, split_ratio={args.split_ratio}...")

    indices = np.arange(len(original_data))
    indices_A, indices_B = train_test_split(
        indices,
        train_size=args.split_ratio,
        random_state=args.random_state
    )

    split_A_data = original_data.iloc[indices_A].copy().reset_index(drop=True)
    split_B_data = original_data.iloc[indices_B].copy().reset_index(drop=True)

    print(f"  split_A (for training reward): {len(split_A_data)} samples ({len(indices_A)} indices)")
    print(f"  split_B (reserved for evaluation): {len(split_B_data)} samples ({len(indices_B)} indices)")

    # Save split_B indices to file so evaluation can use them independently
    pairs_dir = os.path.dirname(args.train_json)
    if pairs_dir and not os.path.exists(pairs_dir):
        os.makedirs(pairs_dir, exist_ok=True)

    split_info = {
        "split_ratio": args.split_ratio,
        "random_state": args.random_state,
        "split_A_indices": indices_A.tolist(),
        "split_B_indices": indices_B.tolist(),
        "split_A_size": len(indices_A),
        "split_B_size": len(indices_B),
        "total_real_samples": len(original_data),
        "original_data_path": args.original,
        "synthetic_data_path": args.synthetic,
        "iteration": current_iter
    }

    split_info_path = os.path.join(pairs_dir, "split_b_indices.json") if pairs_dir else "split_b_indices.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved split_B indices to: {split_info_path}")

    # =====================================================
    # Train MIA classifier ONLY on split_A real data + synthetic
    # =====================================================
    print(f"\nStep 1: Training MIA classifier on split_A real + ALL synthetic data...")

    synthetic_all = synthetic_data.copy()

    # Fit HyperTransformer on split_A + synthetic to capture all categories
    print("  Fitting HyperTransformer on split_A + synthetic data...")
    _, _, ht = preprocess_data(
        split_A_data, synthetic_all, hypertransformer=None
    )
    print("  HyperTransformer fitted on split_A + synthetic dataset")

    # Transform split_A data using the pre-fitted transformer
    print("  Transforming split_A data for MIA training...")
    split_A_transformed, synthetic_all_transformed, _ = preprocess_data(
        split_A_data, synthetic_all, hypertransformer=ht
    )

    # Prepare training data for MIA classifier (split_A only)
    X_train, y_train = prepare_attack_data(
        split_A_transformed, synthetic_all_transformed
    )

    # Train MIA classifier on split_A data only
    print("  Training MIA classifier on split_A data...")
    clf = RandomForestClassifier(n_estimators=100, random_state=args.random_state)
    clf.fit(X_train, y_train)

    print(f"  MIA classifier trained on {len(X_train)} samples")
    print(f"    split_A real samples: {len(split_A_transformed)}")
    print(f"    Synthetic samples: {len(synthetic_all_transformed)}")
    print(f"    NOTE: split_B ({len(split_B_data)} samples) is NOT used for training")

    # =====================================================
    # Step 2: Generate NEW synthetic samples
    # =====================================================
    print(f"\nStep 2: Generating NEW synthetic samples for MIA testing...")

    if args.use_existing_synthetic:
        print(f"  Using existing synthetic data instead of generating NEW samples")
        print(f"  Note: This is NOT recommended for iterative training")
        new_synthetic_data = synthetic_data.copy()
        print(f"  Using existing synthetic samples: {len(new_synthetic_data)}")
    else:
        print(f"  Generating NEW synthetic data using the current model...")

        # Detect dataset name from the synthetic path for model loading
        if 'adult' in args.synthetic:
            dataname = 'adult'
        elif 'diabetes' in args.synthetic:
            dataname = 'diabetes'
        elif 'beijing' in args.synthetic:
            dataname = 'beijing'
        elif 'news' in args.synthetic:
            dataname = 'news'
        elif 'shoppers' in args.synthetic:
            dataname = 'shoppers'
        else:
            dataname = 'adult'  # fallback

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_path = f'{curr_dir}/baselines/{args.model_type}/ckpt/{dataname}/model.pt'

        # Import the appropriate model class based on model_type
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
            print(f"  Falling back to using existing synthetic data...")
            print(f"  Train a base model first using: python main.py --dataname {dataname} --method great --mode train")
            new_synthetic_data = synthetic_data.copy()
        else:
            print(f"  Loading model from: {model_save_path}")

            try:
                # Initialize GReaT model
                great = model_class(base_model_name)

                # Load the trained model
                import torch
                great.model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
                great.load_finetuned_model(model_save_path)

                # Update column information from original data
                df = _array_to_dataframe(original_data, columns=None)
                great._update_column_information(df)
                great._update_conditional_information(df, conditional_col=None)

                # Sample NEW synthetic data with same number as real data
                n_samples = len(original_data)
                print(f"  Sampling {n_samples} NEW synthetic samples...")

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                new_synthetic_data = great.sample(
                    n_samples,
                    k=100,
                    device=device,
                    random_feature_order=True,
                    temperature=0.7
                )

                print(f"  Generated {len(new_synthetic_data)} NEW synthetic samples")

                if len(new_synthetic_data) < max(50, n_samples // 100):
                    print(f"  ⚠️ Fresh sampling produced only {len(new_synthetic_data)} valid rows; "
                          f"falling back to existing synthetic data.")
                    new_synthetic_data = synthetic_data.copy()

                # Save the NEW synthetic data for debugging/reference
                new_synthetic_path = args.synthetic.replace('.csv', '_new.csv')
                new_synthetic_data.to_csv(new_synthetic_path, index=False)
                print(f"  Saved NEW synthetic data to: {new_synthetic_path}")

            except Exception as e:
                print(f"  Error during NEW synthetic data generation: {e}")
                print(f"  Falling back to using existing synthetic data...")
                new_synthetic_data = synthetic_data.copy()

    print(f"  NEW synthetic samples (for testing): {len(new_synthetic_data)}")

    # =====================================================
    # Step 3: Score NEW synthetic using split_A classifier
    # =====================================================
    print("Step 3: Computing privacy scores for NEW synthetic data using split_A classifier...")

    # Transform NEW synthetic data using the pre-fitted HyperTransformer
    print("  Transforming NEW synthetic data with HyperTransformer...")

    try:
        new_synthetic_transformed = ht.transform(new_synthetic_data)
        print(f"  Transformed NEW synthetic data shape: {new_synthetic_transformed.shape}")

        if len(new_synthetic_transformed) == 0:
            raise ValueError("No data after transformation - check for data compatibility")

        new_synthetic_transformed_array = new_synthetic_transformed.values

    except Exception as e:
        print(f"  Direct transformation failed: {e}")
        print(f"  Trying alternative transformation approach...")

        small_original = split_A_data.head(10).copy()
        combined_for_transform = pd.concat([small_original, new_synthetic_data], ignore_index=True)

        combined_transformed = ht.transform(combined_for_transform)

        new_synthetic_transformed_array = combined_transformed.iloc[len(small_original):].values
        print(f"  Alternative transformation successful: {new_synthetic_transformed_array.shape}")

    # Predict membership probability for NEW synthetic samples using split_A classifier
    y_pred_new_synthetic = clf.predict_proba(new_synthetic_transformed_array)[:, 1]

    print(f"  Generated {len(y_pred_new_synthetic)} privacy scores for NEW synthetic data")
    print(f"  NOTE: Scores come from split_A classifier only (split_B not involved)")

    # Convert NEW synthetic rows to GReaT format strings
    print(f"\nConverting NEW synthetic rows to GReaT format strings...")
    new_synthetic_outputs = []
    batch_size = 100
    num_batches = (len(new_synthetic_data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(new_synthetic_data))

        print(f"  Processing batch {batch_idx + 1}/{num_batches} (rows {start_idx}-{end_idx})")

        for idx in range(start_idx, end_idx):
            row_str = row_to_str(new_synthetic_data.iloc[idx])
            new_synthetic_outputs.append(row_str)

    print(f"  Successfully converted {len(new_synthetic_outputs)} NEW synthetic rows to strings")

    # =====================================================
    # Calculate MIA rewards and create DPO pairs
    # =====================================================
    distances = np.abs(y_pred_new_synthetic - 0.5)
    rewards = 1.0 - distances  # Higher reward = better privacy (closer to 0.5)

    sorted_indices = np.argsort(rewards)[::-1]  # Highest reward first

    print(f"\nNEW Synthetic MIA Reward statistics (higher = better privacy):")
    print(f"  Mean reward: {rewards.mean():.4f}")
    print(f"  Best reward: {rewards.max():.4f}")
    print(f"  Worst reward: {rewards.min():.4f}")
    print(f"  Median reward: {np.median(rewards):.4f}")
    print(f"  Std reward: {rewards.std():.4f}")

    # Save MIA scores and rewards to CSV
    base_name = os.path.splitext(os.path.basename(args.train_json))[0]
    iteration_match = base_name.split('_')
    iteration_id = "unknown"
    for i, part in enumerate(iteration_match):
        if part == "iter" and i + 1 < len(iteration_match):
            iteration_id = iteration_match[i + 1]
            break

    mia_data = pd.DataFrame({
        'sample_id': range(len(y_pred_new_synthetic)),
        'mia_score': y_pred_new_synthetic,
        'mia_reward': rewards,
        'distance_from_0.5': distances,
        'sample_rank': [np.where(sorted_indices == i)[0][0] + 1 for i in range(len(sorted_indices))]
    })

    csv_filename = f"mia_scores_rewards_iter_{iteration_id}.csv"
    csv_path = os.path.join(pairs_dir, csv_filename) if pairs_dir else csv_filename

    try:
        mia_data.to_csv(csv_path, index=False)
        print(f"  Saved MIA scores and rewards to: {csv_path}")
    except Exception as e:
        print(f"  Warning: Failed to save CSV file: {e}")
        csv_path = None

    # Create and save distribution plots
    print(f"  Creating distribution plots...")

    try:
        plt.style.use('default')

        try:
            sns.set_palette("husl")
            use_seaborn = True
        except Exception:
            use_seaborn = False

        # Plot 1: MIA Score Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.hist(y_pred_new_synthetic, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Perfect Privacy (0.5)')
        ax1.axvline(x=y_pred_new_synthetic.mean(), color='orange', linestyle='-', linewidth=2,
                    label=f'Mean: {y_pred_new_synthetic.mean():.3f}')
        ax1.set_xlabel('MIA Score (Membership Probability)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Distribution of MIA Scores (Holdout split_A classifier)\nIteration {iteration_id}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        stats_text = (f'N = {len(y_pred_new_synthetic)}\n'
                      f'Mean = {y_pred_new_synthetic.mean():.3f}\n'
                      f'Std = {y_pred_new_synthetic.std():.3f}\n'
                      f'Median = {np.median(y_pred_new_synthetic):.3f}')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.hist(rewards, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=rewards.mean(), color='orange', linestyle='-', linewidth=2,
                    label=f'Mean: {rewards.mean():.3f}')
        ax2.set_xlabel('MIA Reward (1 - |score - 0.5|)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Distribution of MIA Rewards (Holdout split_A classifier)\nIteration {iteration_id}', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        reward_stats_text = (f'N = {len(rewards)}\n'
                             f'Mean = {rewards.mean():.3f}\n'
                             f'Std = {rewards.std():.3f}\n'
                             f'Median = {np.median(rewards):.3f}')
        ax2.text(0.02, 0.98, reward_stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        plot_filename = f"mia_distributions_holdout_iter_{iteration_id}.png"
        plot_path = os.path.join(pairs_dir, plot_filename) if pairs_dir else plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved MIA distribution plot to: {plot_path}")

        # Plot 2: Detailed Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        scatter = ax1.scatter(y_pred_new_synthetic, rewards, alpha=0.6, c=rewards,
                              cmap='RdYlGn', s=20)
        ax1.set_xlabel('MIA Score (Membership Probability)', fontsize=12)
        ax1.set_ylabel('MIA Reward (1 - |score - 0.5|)', fontsize=12)
        ax1.set_title(f'MIA Score vs Reward (Holdout split_A)\nIteration {iteration_id}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Perfect Privacy')
        ax1.legend()

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Reward Value', fontsize=10)

        reward_quartiles = np.percentile(rewards, [25, 50, 75])

        q1_mask = rewards <= reward_quartiles[0]
        q2_mask = (rewards > reward_quartiles[0]) & (rewards <= reward_quartiles[1])
        q3_mask = (rewards > reward_quartiles[1]) & (rewards <= reward_quartiles[2])
        q4_mask = rewards > reward_quartiles[2]

        quartile_data = [
            y_pred_new_synthetic[q1_mask],
            y_pred_new_synthetic[q2_mask],
            y_pred_new_synthetic[q3_mask],
            y_pred_new_synthetic[q4_mask]
        ]
        quartile_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']

        ax2.boxplot(quartile_data, labels=quartile_labels)
        ax2.set_xlabel('Reward Quartiles', fontsize=12)
        ax2.set_ylabel('MIA Score', fontsize=12)
        ax2.set_title(f'MIA Scores by Reward Quartile (Holdout split_A)\nIteration {iteration_id}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Perfect Privacy')
        ax2.legend()

        plt.tight_layout()

        analysis_plot_filename = f"mia_analysis_holdout_iter_{iteration_id}.png"
        analysis_plot_path = os.path.join(pairs_dir, analysis_plot_filename) if pairs_dir else analysis_plot_filename
        plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved MIA analysis plot to: {analysis_plot_path}")

    except Exception as e:
        print(f"  Warning: Failed to create plots: {e}")
        print(f"  Continuing with DPO pair generation...")
        plot_path = None
        analysis_plot_path = None

    print(f"\nTop 10 highest reward NEW synthetic samples:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        reward = rewards[idx]
        prob = y_pred_new_synthetic[idx]
        print(f"  Rank {i+1}: NEW_SYNTHETIC - reward={reward:.4f}, prob={prob:.4f}")

    print(f"\nTop 10 lowest reward NEW synthetic samples:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[-(i+1)]
        reward = rewards[idx]
        prob = y_pred_new_synthetic[idx]
        print(f"  Rank {i+1}: NEW_SYNTHETIC - reward={reward:.4f}, prob={prob:.4f}")

    # =====================================================
    # Create DPO preference pairs (identical logic to original)
    # =====================================================
    dpo_data = []
    num_pairs = len(sorted_indices) // 2
    num_samples = len(sorted_indices)

    print(f"\nCreating {num_pairs} DPO preference pairs from NEW synthetic data...")
    print(f"  Chosen (highest reward) - Rejected (lowest reward)")

    for i in range(num_pairs):
        chosen_idx = sorted_indices[i]       # Highest reward (best privacy)
        rejected_idx = sorted_indices[-(i+1)] # Lowest reward (worst privacy)

        entry = {
            "instruction": "",
            "input": "",
            "chosen": new_synthetic_outputs[chosen_idx],
            "rejected": new_synthetic_outputs[rejected_idx],
            "chosen_source": "new_synthetic",
            "rejected_source": "new_synthetic",
            "chosen_privacy_score": float(y_pred_new_synthetic[chosen_idx]),
            "rejected_privacy_score": float(y_pred_new_synthetic[rejected_idx]),
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
            "chosen_privacy_score": float(y_pred_new_synthetic[mid_idx]),
            "rejected_privacy_score": float(y_pred_new_synthetic[mid_idx]),
            "chosen_reward": float(rewards[mid_idx]),
            "rejected_reward": 0.0
        }
        dpo_data.append(entry)

    # Filter to top X% of pairs based on chosen sample's reward
    total_pairs_before = len(dpo_data)
    if args.top_percent < 100.0:
        dpo_data.sort(key=lambda x: x['chosen_reward'], reverse=True)

        num_pairs_to_keep = int(len(dpo_data) * args.top_percent / 100.0)
        dpo_data = dpo_data[:num_pairs_to_keep]

        print(f"\n  Filtered to top {args.top_percent}% of pairs (by chosen reward):")
        print(f"    Total pairs before: {total_pairs_before}")
        print(f"    Pairs kept: {len(dpo_data)}")
        print(f"    Pairs removed: {total_pairs_before - len(dpo_data)}")
        print(f"    Best chosen reward: {dpo_data[0]['chosen_reward']:.4f}")
        print(f"    Worst (of kept) chosen reward: {dpo_data[-1]['chosen_reward']:.4f}")

    # Analyze pair composition
    syn_syn = sum(1 for d in dpo_data if d['chosen_source'] == 'synthetic' and d['rejected_source'] == 'synthetic')
    syn_test = sum(1 for d in dpo_data if d['chosen_source'] == 'synthetic' and d['rejected_source'] == 'real_test')
    test_syn = sum(1 for d in dpo_data if d['chosen_source'] == 'real_test' and d['rejected_source'] == 'synthetic')
    test_test = sum(1 for d in dpo_data if d['chosen_source'] == 'real_test' and d['rejected_source'] == 'real_test')
    newsyn_newsyn = sum(1 for d in dpo_data if d['chosen_source'] == 'new_synthetic' and d['rejected_source'] == 'new_synthetic')

    print(f"\nPair composition:")
    print(f"  New_synthetic vs New_synthetic: {newsyn_newsyn}")
    print(f"  Synthetic (chosen) vs Synthetic (rejected): {syn_syn}")
    print(f"  Synthetic (chosen) vs Real_test (rejected): {syn_test}")
    print(f"  Real_test (chosen) vs Synthetic (rejected): {test_syn}")
    print(f"  Real_test (chosen) vs Real_test (rejected): {test_test}")

    # Save DPO data
    train_data = dpo_data

    print(f"\nSaving DPO data:")
    print(f"  Train pairs: {len(train_data)} (ALL pairs used for training)")
    print(f"  Total pairs: {len(dpo_data)}")
    print(f"  Train file: {args.train_json}")

    with open(args.train_json, "w") as f:
        json.dump(train_data, f, indent=2)

    print(f"\n  Saved {args.train_json}")

    # =====================================================
    # Summary
    # =====================================================
    print(f"\n{'='*60}")
    print(f"HOLDOUT SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Real data total: {len(original_data)} samples")
    print(f"  split_A (training reward): {len(split_A_data)} samples ({args.split_ratio*100:.0f}%)")
    print(f"  split_B (evaluation hold-out): {len(split_B_data)} samples ({(1-args.split_ratio)*100:.0f}%)")
    print(f"  Random state: {args.random_state}")
    print(f"")
    print(f"  Files created:")
    print(f"    - {args.train_json} ({len(train_data)} DPO pairs)")
    print(f"    - {split_info_path} (split_B indices for independent evaluation)")
    if csv_path:
        print(f"    - {csv_path} (MIA scores & rewards)")
    if 'plot_path' in locals() and plot_path is not None:
        print(f"    - {plot_path} (MIA distribution plots)")
    if 'analysis_plot_path' in locals() and analysis_plot_path is not None:
        print(f"    - {analysis_plot_path} (MIA analysis plots)")
    print(f"")
    print(f"  To run independent evaluation on split_B:")
    print(f"    python eval/eval_holdout_classifier.py \\")
    print(f"      --dataname <dataset> \\")
    print(f"      --path <synthetic_csv> \\")
    print(f"      --split_info_path {split_info_path}")
    print(f"")
    print(f"  This proves privacy improvement generalizes beyond the training classifier,")
    print(f"  addressing reviewer concerns about evaluation circularity.")


if __name__ == "__main__":
    main()
