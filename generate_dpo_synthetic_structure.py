"""
Generate DPO/GRAA preference pairs using Structural Fidelity as the reward signal.

Instead of MIA-based privacy reward, this scores synthetic samples by how well they
preserve the structural relationships (Markov blankets / causal dependencies) of the
real data. Inspired by TabStruct (ICLR 2026 Oral).

Reward: For each group of synthetic rows, train predictors on synthetic, test on real.
Higher prediction accuracy = better structural preservation = higher reward.
"""

import pandas as pd
import numpy as np
import json
import os
import re
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

warnings.filterwarnings("ignore")


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


def detect_column_types(df, info=None):
    """Detect which columns are numerical vs categorical."""
    if info is not None:
        num_cols = [df.columns[i] for i in info.get("num_col_idx", [])]
        cat_cols = [df.columns[i] for i in info.get("cat_col_idx", [])]
        target_cols = [df.columns[i] for i in info.get("target_col_idx", [])]
        return num_cols, cat_cols, target_cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols, []


def compute_structural_score(real_df, syn_df, num_cols, cat_cols, n_estimators=30, max_features=10):
    """
    Compute structural fidelity score for a synthetic sample group.

    For each feature as target: train RF on synthetic, evaluate on real.
    Compare against RF trained on real (reference).
    Score = mean ratio across features. Higher = better structural preservation.
    """
    common_cols = [c for c in real_df.columns if c in syn_df.columns]
    real = real_df[common_cols].copy()
    syn = syn_df[common_cols].copy()

    # Select features to test
    test_cols = list(real.columns)
    if max_features and len(test_cols) > max_features:
        np.random.seed(42)
        np.random.shuffle(test_cols)
        test_cols = test_cols[:max_features]

    ratios = []

    for col in test_cols:
        try:
            is_cat = col in cat_cols

            if syn[col].nunique() <= 1:
                continue
            if is_cat and real[col].nunique() > 20:
                continue

            # Encode features
            feature_cols = [c for c in common_cols if c != col]
            X_syn = pd.DataFrame(index=syn.index)
            X_real = pd.DataFrame(index=real.index)

            for fc in feature_cols:
                if fc in cat_cols:
                    le = LabelEncoder()
                    all_vals = pd.concat([syn[fc].astype(str), real[fc].astype(str)])
                    le.fit(all_vals)
                    X_syn[fc] = le.transform(syn[fc].astype(str))
                    X_real[fc] = le.transform(real[fc].astype(str))
                else:
                    X_syn[fc] = pd.to_numeric(syn[fc], errors='coerce').fillna(0)
                    X_real[fc] = pd.to_numeric(real[fc], errors='coerce').fillna(0)

            X_syn_arr = X_syn.values
            X_real_arr = X_real.values

            if is_cat:
                le_target = LabelEncoder()
                all_targets = pd.concat([syn[col].astype(str), real[col].astype(str)])
                le_target.fit(all_targets)
                y_syn = le_target.transform(syn[col].astype(str))
                y_real = le_target.transform(real[col].astype(str))

                clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                clf.fit(X_syn_arr, y_syn)
                score_syn = balanced_accuracy_score(y_real, clf.predict(X_real_arr))

                clf_real = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                clf_real.fit(X_real_arr, y_real)
                score_real = balanced_accuracy_score(y_real, clf_real.predict(X_real_arr))

                ratio = min(score_syn / score_real, 1.0) if score_real > 0 else (1.0 if score_syn == 0 else 0.0)
            else:
                y_syn = pd.to_numeric(syn[col], errors='coerce').fillna(0).values
                y_real = pd.to_numeric(real[col], errors='coerce').fillna(0).values

                scaler = StandardScaler()
                y_syn_s = scaler.fit_transform(y_syn.reshape(-1, 1)).ravel()
                y_real_s = scaler.transform(y_real.reshape(-1, 1)).ravel()

                reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                reg.fit(X_syn_arr, y_syn_s)
                rmse_syn = np.sqrt(mean_squared_error(y_real_s, reg.predict(X_real_arr)))

                reg_real = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                reg_real.fit(X_real_arr, y_real_s)
                rmse_real = np.sqrt(mean_squared_error(y_real_s, reg_real.predict(X_real_arr)))

                ratio = min(rmse_real / rmse_syn, 1.0) if rmse_syn > 0 else 1.0

            ratios.append(ratio)
        except Exception:
            continue

    return np.mean(ratios) if ratios else 0.0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DPO/GRAA pairs using Structural Fidelity reward"
    )
    parser.add_argument('--synthetic', default='synthetic/adult/great.csv',
                        help='Path to synthetic data CSV')
    parser.add_argument('--original', default='synthetic/adult/real.csv',
                        help='Path to original/training data CSV')
    parser.add_argument('--train_json', default='train_data_synthetic.json',
                        help='Output train JSON file')
    parser.add_argument('--test_json', default='test_data_synthetic.json',
                        help='Output test JSON file (kept for compatibility)')
    parser.add_argument('--top_percent', type=float, default=100.0,
                        help='Select top X percent of pairs')
    parser.add_argument('--use_existing_synthetic', action='store_true',
                        help='Use existing synthetic data instead of generating NEW samples')
    parser.add_argument('--model_type', type=str, default='great',
                        choices=['great', 'great_gemma'],
                        help='Model type')
    parser.add_argument('--group_size', type=int, default=None,
                        help='Group size for structural scoring (default: auto based on dataset)')
    parser.add_argument('--max_features', type=int, default=10,
                        help='Max features to test per group evaluation')
    parser.add_argument('--n_estimators', type=int, default=30,
                        help='Number of RF trees for structural scoring')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"STRUCTURAL FIDELITY DPO PAIR GENERATION")
    print(f"{'='*60}")

    iter_match = re.search(r'iter_(\d+)', args.train_json)
    current_iter = int(iter_match.group(1)) if iter_match else 1
    print(f"Processing iteration: {current_iter}")

    if args.top_percent <= 0 or args.top_percent > 100:
        raise ValueError("--top_percent must be between 0 and 100")

    synthetic_data = pd.read_csv(args.synthetic)
    original_data = pd.read_csv(args.original)

    print(f"Loaded {len(synthetic_data)} synthetic samples and {len(original_data)} original samples")

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

    # Load dataset info for column types
    info = None
    dataname = None
    for name in ['adult', 'diabetes', 'beijing', 'news', 'shoppers', 'magic', 'default']:
        if name in args.synthetic:
            dataname = name
            break

    if dataname:
        info_path = f'data/{dataname}/info.json'
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            print(f"Loaded dataset info for: {dataname}")

    num_cols, cat_cols, target_cols = detect_column_types(original_data, info)
    print(f"  Numerical columns: {len(num_cols)}")
    print(f"  Categorical columns: {len(cat_cols)}")

    # Step 1: Generate NEW synthetic samples (or use existing)
    if args.use_existing_synthetic:
        print(f"Using existing synthetic data")
        new_synthetic_data = synthetic_data.copy()
    else:
        print(f"Generating NEW synthetic data using the current model...")

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_path = f'{curr_dir}/baselines/{args.model_type}/ckpt/{dataname}/model.pt'

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
            print(f"Model not found at {model_save_path}, using existing synthetic data")
            new_synthetic_data = synthetic_data.copy()
        else:
            try:
                import torch
                great = model_class(base_model_name)
                great.model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
                great.load_finetuned_model(model_save_path)

                df = _array_to_dataframe(original_data, columns=None)
                great._update_column_information(df)
                great._update_conditional_information(df, conditional_col=None)

                n_samples = len(original_data)
                print(f"Sampling {n_samples} NEW synthetic samples...")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                new_synthetic_data = great.sample(
                    n_samples, k=100, device=device,
                    random_feature_order=True, temperature=0.7
                )
                print(f"Generated {len(new_synthetic_data)} NEW synthetic samples")

                new_synthetic_path = args.synthetic.replace('.csv', '_new.csv')
                new_synthetic_data.to_csv(new_synthetic_path, index=False)
            except Exception as e:
                print(f"Error during generation: {e}")
                new_synthetic_data = synthetic_data.copy()

    # Step 2: Compute per-row structural fidelity rewards
    # Strategy: Train RF on ALL synthetic data, predict on real data.
    # Then for each synthetic row, compute how "structurally consistent" it is
    # by measuring its prediction alignment with the real data patterns.
    print(f"\nStep 2: Computing per-row structural fidelity rewards...")
    print(f"  Strategy: Train on synthetic, test on real. Score each synthetic row")
    print(f"  by how well it aligns with learned structural patterns.")

    n_total = len(new_synthetic_data)
    row_rewards = np.zeros(n_total)
    n_features_scored = 0

    test_cols = list(original_data.columns)
    if args.max_features and len(test_cols) > args.max_features:
        np.random.seed(42)
        np.random.shuffle(test_cols)
        test_cols = test_cols[:args.max_features]

    for col in test_cols:
        try:
            is_cat = col in cat_cols
            if new_synthetic_data[col].nunique() <= 1:
                continue
            if is_cat and original_data[col].nunique() > 20:
                continue

            feature_cols = [c for c in original_data.columns if c != col]
            X_syn = pd.DataFrame(index=new_synthetic_data.index)
            X_real = pd.DataFrame(index=original_data.index)

            for fc in feature_cols:
                if fc in cat_cols:
                    le = LabelEncoder()
                    all_vals = pd.concat([new_synthetic_data[fc].astype(str), original_data[fc].astype(str)])
                    le.fit(all_vals)
                    X_syn[fc] = le.transform(new_synthetic_data[fc].astype(str))
                    X_real[fc] = le.transform(original_data[fc].astype(str))
                else:
                    X_syn[fc] = pd.to_numeric(new_synthetic_data[fc], errors='coerce').fillna(0)
                    X_real[fc] = pd.to_numeric(original_data[fc], errors='coerce').fillna(0)

            X_syn_arr = X_syn.values
            X_real_arr = X_real.values

            if is_cat:
                le_target = LabelEncoder()
                all_targets = pd.concat([new_synthetic_data[col].astype(str), original_data[col].astype(str)])
                le_target.fit(all_targets)
                y_syn = le_target.transform(new_synthetic_data[col].astype(str))
                y_real = le_target.transform(original_data[col].astype(str))

                # Train on synthetic with OOB enabled for honest per-row scores
                clf = RandomForestClassifier(
                    n_estimators=args.n_estimators, random_state=42, n_jobs=-1,
                    oob_score=True, bootstrap=True
                )
                clf.fit(X_syn_arr, y_syn)

                # Per-row reward using OOB predictions (out-of-bag = not seen during training)
                # This avoids the self-prediction overfitting bias
                if hasattr(clf, 'oob_decision_function_'):
                    oob_proba = clf.oob_decision_function_
                    for i in range(n_total):
                        true_class = y_syn[i]
                        class_idx = list(clf.classes_).index(true_class) if true_class in clf.classes_ else -1
                        if class_idx >= 0 and class_idx < oob_proba.shape[1]:
                            row_rewards[i] += oob_proba[i, class_idx]
                else:
                    # Fallback if OOB not available
                    syn_proba = clf.predict_proba(X_syn_arr)
                    for i in range(n_total):
                        true_class = y_syn[i]
                        class_idx = list(clf.classes_).index(true_class) if true_class in clf.classes_ else -1
                        if class_idx >= 0:
                            row_rewards[i] += syn_proba[i, class_idx]

                # Transfer score: train on real, predict on each synthetic row
                # Rows that are predictable by real-data patterns = structurally consistent
                clf_real = RandomForestClassifier(
                    n_estimators=args.n_estimators, random_state=42, n_jobs=-1
                )
                clf_real.fit(X_real_arr, y_real)
                real2syn_proba = clf_real.predict_proba(X_syn_arr)
                for i in range(n_total):
                    true_class = y_syn[i]
                    class_idx = list(clf_real.classes_).index(true_class) if true_class in clf_real.classes_ else -1
                    if class_idx >= 0 and class_idx < real2syn_proba.shape[1]:
                        row_rewards[i] += real2syn_proba[i, class_idx]

            else:
                y_syn = pd.to_numeric(new_synthetic_data[col], errors='coerce').fillna(0).values
                y_real = pd.to_numeric(original_data[col], errors='coerce').fillna(0).values

                scaler = StandardScaler()
                y_syn_s = scaler.fit_transform(y_syn.reshape(-1, 1)).ravel()
                y_real_s = scaler.transform(y_real.reshape(-1, 1)).ravel()

                # Train on synthetic with OOB for honest per-row scores
                reg = RandomForestRegressor(
                    n_estimators=args.n_estimators, random_state=42, n_jobs=-1,
                    oob_score=True, bootstrap=True
                )
                reg.fit(X_syn_arr, y_syn_s)

                # Per-row reward using OOB predictions
                if hasattr(reg, 'oob_prediction_'):
                    oob_pred = reg.oob_prediction_
                    syn_errors = np.abs(y_syn_s - oob_pred)
                else:
                    syn_pred = reg.predict(X_syn_arr)
                    syn_errors = np.abs(y_syn_s - syn_pred)
                max_err = syn_errors.max() if syn_errors.max() > 0 else 1.0
                row_rewards += (1.0 - syn_errors / max_err)

                # Transfer score: train on real, predict on each synthetic row
                # Low error = synthetic row follows real structural patterns
                reg_real = RandomForestRegressor(
                    n_estimators=args.n_estimators, random_state=42, n_jobs=-1
                )
                reg_real.fit(X_real_arr, y_real_s)
                real2syn_pred = reg_real.predict(X_syn_arr)
                syn_transfer_errors = np.abs(y_syn_s - real2syn_pred)
                max_te = syn_transfer_errors.max() if syn_transfer_errors.max() > 0 else 1.0
                row_rewards += (1.0 - syn_transfer_errors / max_te)

            n_features_scored += 1
            print(f"  Scored feature: {col} ({'cls' if is_cat else 'reg'})")

        except Exception as e:
            print(f"  Skipping {col}: {e}")
            continue

    # Normalize rewards to [0, 1]
    if row_rewards.max() > row_rewards.min():
        row_rewards = (row_rewards - row_rewards.min()) / (row_rewards.max() - row_rewards.min())

    print(f"\nStructural Fidelity Reward statistics ({n_features_scored} features scored):")
    print(f"  Mean: {row_rewards.mean():.4f}")
    print(f"  Std: {row_rewards.std():.4f}")
    print(f"  Best: {row_rewards.max():.4f}")
    print(f"  Worst: {row_rewards.min():.4f}")

    # Step 3: Convert rows to GReaT format strings
    print(f"\nConverting synthetic rows to GReaT format strings...")
    synthetic_outputs = []
    for idx in range(len(new_synthetic_data)):
        row_str = row_to_str(new_synthetic_data.iloc[idx])
        synthetic_outputs.append(row_str)
    print(f"Converted {len(synthetic_outputs)} rows")

    # Step 4: Create DPO pairs using structural fidelity rewards
    # Sort rows by reward (highest first)
    sorted_row_indices = np.argsort(row_rewards)[::-1]

    # Create DPO pairs: top half (chosen) vs bottom half (rejected)
    dpo_data = []
    num_pairs = len(sorted_row_indices) // 2

    print(f"\nCreating {num_pairs} DPO preference pairs...")
    print(f"  Chosen = high structural fidelity, Rejected = low structural fidelity")

    for i in range(num_pairs):
        chosen_idx = sorted_row_indices[i]
        rejected_idx = sorted_row_indices[-(i+1)]

        entry = {
            "instruction": "",
            "input": "",
            "chosen": synthetic_outputs[chosen_idx],
            "rejected": synthetic_outputs[rejected_idx],
            "chosen_source": "new_synthetic",
            "rejected_source": "new_synthetic",
            "chosen_privacy_score": float(row_rewards[chosen_idx]),
            "rejected_privacy_score": float(row_rewards[rejected_idx]),
            "chosen_reward": float(row_rewards[chosen_idx]),
            "rejected_reward": float(row_rewards[rejected_idx])
        }
        dpo_data.append(entry)

    # Handle odd sample
    if len(sorted_row_indices) % 2 == 1:
        mid_idx = sorted_row_indices[num_pairs]
        entry = {
            "instruction": "",
            "input": "",
            "chosen": synthetic_outputs[mid_idx],
            "rejected": synthetic_outputs[mid_idx],
            "chosen_source": "new_synthetic",
            "rejected_source": "new_synthetic",
            "chosen_privacy_score": float(row_rewards[mid_idx]),
            "rejected_privacy_score": float(row_rewards[mid_idx]),
            "chosen_reward": float(row_rewards[mid_idx]),
            "rejected_reward": 0.0
        }
        dpo_data.append(entry)

    # Apply top_percent filter
    if args.top_percent < 100.0:
        n_keep = int(len(dpo_data) * args.top_percent / 100.0)
        dpo_data = dpo_data[:n_keep]
        print(f"  Filtered to top {args.top_percent}%: {len(dpo_data)} pairs")

    # Save pairs
    pairs_dir = os.path.dirname(args.train_json)
    if pairs_dir:
        os.makedirs(pairs_dir, exist_ok=True)

    with open(args.train_json, 'w') as f:
        json.dump(dpo_data, f, indent=2)
    print(f"Saved {len(dpo_data)} DPO pairs to: {args.train_json}")

    # Save scores CSV
    iter_id = str(current_iter)
    scores_df = pd.DataFrame({
        'sample_id': range(len(row_rewards)),
        'structural_fidelity_reward': row_rewards,
    })
    csv_path = os.path.join(pairs_dir if pairs_dir else '.', f"structure_scores_iter_{iter_id}.csv")
    scores_df.to_csv(csv_path, index=False)
    print(f"Saved structural fidelity scores to: {csv_path}")

    # Save plots
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(row_rewards, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(x=row_rewards.mean(), color='orange', linestyle='-', linewidth=2,
                     label=f'Mean: {row_rewards.mean():.3f}')
        ax1.set_xlabel('Structural Fidelity Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Structural Fidelity Distribution\nIteration {iter_id}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Chosen vs rejected distribution
        chosen_rewards = [d['chosen_reward'] for d in dpo_data[:min(500, len(dpo_data))]]
        rejected_rewards = [d['rejected_reward'] for d in dpo_data[:min(500, len(dpo_data))]]
        ax2.hist(chosen_rewards, bins=20, alpha=0.6, label='Chosen', color='green', edgecolor='black')
        ax2.hist(rejected_rewards, bins=20, alpha=0.6, label='Rejected', color='red', edgecolor='black')
        ax2.set_xlabel('Structural Fidelity Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Chosen vs Rejected Distribution\nIteration {iter_id}', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(pairs_dir if pairs_dir else '.', f"structure_distributions_iter_{iter_id}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved distribution plot to: {plot_path}")
    except Exception as e:
        print(f"Warning: Failed to create plots: {e}")

    print(f"\n{'='*60}")
    print(f"STRUCTURAL FIDELITY PAIR GENERATION COMPLETE")
    print(f"  Pairs: {len(dpo_data)}")
    print(f"  Mean structural fidelity: {row_rewards.mean():.4f}")
    print(f"  Reward gap (chosen - rejected): {np.mean(chosen_rewards) - np.mean(rejected_rewards):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
