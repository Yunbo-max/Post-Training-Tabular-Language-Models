"""
Lightweight Structural Fidelity Metric (inspired by TabStruct's UtilityPerFeature).

Core idea: For each feature as target, train a predictor on synthetic data,
evaluate on real data. If synthetic preserves structural relationships (Markov
blankets / causal structure), models trained on synthetic should predict well on real.

This is a lightweight version using sklearn instead of AutoGluon, suitable for
use as a DPO reward signal in iterative training loops.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


def detect_column_types(df, info=None):
    """Detect which columns are numerical vs categorical."""
    if info is not None:
        num_cols = [df.columns[i] for i in info.get("num_col_idx", [])]
        cat_cols = [df.columns[i] for i in info.get("cat_col_idx", [])]
        target_cols = [df.columns[i] for i in info.get("target_col_idx", [])]
        return num_cols, cat_cols, target_cols

    # Auto-detect
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    target_cols = []
    return num_cols, cat_cols, target_cols


def encode_features(df_train, df_test, target_col, num_cols, cat_cols):
    """Encode features for sklearn, handling unseen categories gracefully."""
    feature_cols = [c for c in df_train.columns if c != target_col]

    X_train = pd.DataFrame(index=df_train.index)
    X_test = pd.DataFrame(index=df_test.index)

    encoders = {}

    for col in feature_cols:
        if col in cat_cols:
            le = LabelEncoder()
            # Fit on union of train+test values to avoid unseen labels
            all_vals = pd.concat([df_train[col].astype(str), df_test[col].astype(str)])
            le.fit(all_vals)
            X_train[col] = le.transform(df_train[col].astype(str))
            X_test[col] = le.transform(df_test[col].astype(str))
            encoders[col] = le
        else:
            X_train[col] = pd.to_numeric(df_train[col], errors="coerce").fillna(0)
            X_test[col] = pd.to_numeric(df_test[col], errors="coerce").fillna(0)

    return X_train.values, X_test.values, encoders


def compute_structural_fidelity(
    real_df,
    synthetic_df,
    info=None,
    max_features_to_test=None,
    n_estimators=50,
    verbose=False,
):
    """
    Compute structural fidelity score between real and synthetic data.

    For each feature (used as prediction target):
      - Train RF on synthetic data
      - Evaluate on real data
      - Compare to RF trained on real data (reference)

    Score = mean ratio of (synthetic-trained accuracy / real-trained accuracy)
    across all features. Score of 1.0 = perfect structural preservation.

    Args:
        real_df: DataFrame of real data
        synthetic_df: DataFrame of synthetic data
        info: dataset info dict (with num_col_idx, cat_col_idx, etc.)
        max_features_to_test: limit number of features to test (None = all)
        n_estimators: number of trees in RF
        verbose: print per-feature scores

    Returns:
        dict with:
            - 'score': aggregate structural fidelity score (0 to 1)
            - 'per_feature': dict of per-feature scores
            - 'details': dict with classification and regression breakdowns
    """
    # Align columns
    common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
    real_df = real_df[common_cols].copy()
    synthetic_df = synthetic_df[common_cols].copy()

    num_cols, cat_cols, target_cols = detect_column_types(real_df, info)

    # Determine which columns to test as targets
    test_cols = list(real_df.columns)
    if max_features_to_test and len(test_cols) > max_features_to_test:
        # Prioritize: target columns first, then sample rest
        priority = [c for c in test_cols if c in target_cols]
        remaining = [c for c in test_cols if c not in target_cols]
        np.random.seed(42)
        np.random.shuffle(remaining)
        test_cols = priority + remaining[:max_features_to_test - len(priority)]

    per_feature_scores = {}
    classification_scores = {}
    regression_scores = {}

    for col in test_cols:
        try:
            is_categorical = col in cat_cols

            # Skip if constant in synthetic
            if synthetic_df[col].nunique() <= 1:
                if verbose:
                    print(f"  Skipping {col}: constant in synthetic data")
                continue

            # Skip if too many categories (slow)
            if is_categorical and real_df[col].nunique() > 20:
                if verbose:
                    print(f"  Skipping {col}: too many categories ({real_df[col].nunique()})")
                continue

            # Encode features
            X_syn, X_real, _ = encode_features(
                synthetic_df, real_df, col, num_cols, cat_cols
            )

            if is_categorical:
                # Classification task
                le_target = LabelEncoder()
                all_targets = pd.concat([
                    synthetic_df[col].astype(str), real_df[col].astype(str)
                ])
                le_target.fit(all_targets)
                y_syn = le_target.transform(synthetic_df[col].astype(str))
                y_real = le_target.transform(real_df[col].astype(str))

                # Train on synthetic, test on real
                clf_syn = RandomForestClassifier(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                )
                clf_syn.fit(X_syn, y_syn)
                y_pred_syn = clf_syn.predict(X_real)
                score_syn = balanced_accuracy_score(y_real, y_pred_syn)

                # Train on real, test on real (reference via cross-val)
                clf_real = RandomForestClassifier(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                )
                try:
                    cv_scores = cross_val_score(
                        clf_real, X_real, y_real, cv=3,
                        scoring="balanced_accuracy"
                    )
                    score_real = cv_scores.mean()
                except Exception:
                    clf_real.fit(X_real, y_real)
                    score_real = balanced_accuracy_score(y_real, clf_real.predict(X_real))

                # Ratio (capped at 1.0)
                if score_real > 0:
                    ratio = min(score_syn / score_real, 1.0)
                else:
                    ratio = 1.0 if score_syn == 0 else 0.0

                per_feature_scores[col] = ratio
                classification_scores[col] = {
                    "syn_trained": score_syn,
                    "real_trained": score_real,
                    "ratio": ratio,
                }

            else:
                # Regression task
                y_syn = pd.to_numeric(synthetic_df[col], errors="coerce").fillna(0).values
                y_real = pd.to_numeric(real_df[col], errors="coerce").fillna(0).values

                # Normalize target for fair comparison
                scaler = StandardScaler()
                y_syn_scaled = scaler.fit_transform(y_syn.reshape(-1, 1)).ravel()
                y_real_scaled = scaler.transform(y_real.reshape(-1, 1)).ravel()

                # Train on synthetic, test on real
                reg_syn = RandomForestRegressor(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                )
                reg_syn.fit(X_syn, y_syn_scaled)
                y_pred_syn = reg_syn.predict(X_real)
                rmse_syn = np.sqrt(mean_squared_error(y_real_scaled, y_pred_syn))

                # Train on real (reference via cross-val)
                reg_real = RandomForestRegressor(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                )
                try:
                    cv_scores = cross_val_score(
                        reg_real, X_real, y_real_scaled, cv=3,
                        scoring="neg_root_mean_squared_error"
                    )
                    rmse_real = -cv_scores.mean()
                except Exception:
                    reg_real.fit(X_real, y_real_scaled)
                    rmse_real = np.sqrt(
                        mean_squared_error(y_real_scaled, reg_real.predict(X_real))
                    )

                # Convert RMSE to score: lower RMSE = better
                # Ratio: real_rmse / syn_rmse (capped at 1.0)
                if rmse_syn > 0:
                    ratio = min(rmse_real / rmse_syn, 1.0)
                else:
                    ratio = 1.0

                per_feature_scores[col] = ratio
                regression_scores[col] = {
                    "syn_rmse": rmse_syn,
                    "real_rmse": rmse_real,
                    "ratio": ratio,
                }

            if verbose:
                task = "cls" if is_categorical else "reg"
                print(f"  {col} ({task}): ratio={ratio:.4f}")

        except Exception as e:
            if verbose:
                print(f"  {col}: ERROR - {e}")
            continue

    # Aggregate score
    if per_feature_scores:
        aggregate_score = np.mean(list(per_feature_scores.values()))
    else:
        aggregate_score = 0.0

    return {
        "score": aggregate_score,
        "per_feature": per_feature_scores,
        "details": {
            "classification": classification_scores,
            "regression": regression_scores,
            "n_features_tested": len(per_feature_scores),
            "n_features_total": len(real_df.columns),
        },
    }


def compute_structural_fidelity_reward(
    real_df,
    synthetic_samples,
    info=None,
    group_size=4,
    n_estimators=30,
    max_features_to_test=10,
):
    """
    Compute per-group structural fidelity scores for DPO reward.

    Splits synthetic_samples into groups of `group_size`, computes structural
    fidelity for each group against real data, returns per-group scores.

    Args:
        real_df: DataFrame of real data
        synthetic_samples: DataFrame of all synthetic samples
        info: dataset info dict
        group_size: number of rows per group
        n_estimators: RF trees (lower = faster)
        max_features_to_test: limit features per evaluation

    Returns:
        list of (group_indices, score) tuples, sorted by score descending
    """
    n_samples = len(synthetic_samples)
    n_groups = n_samples // group_size

    group_scores = []

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_df = synthetic_samples.iloc[start:end]

        result = compute_structural_fidelity(
            real_df,
            group_df,
            info=info,
            max_features_to_test=max_features_to_test,
            n_estimators=n_estimators,
            verbose=False,
        )

        group_scores.append((list(range(start, end)), result["score"]))

    # Sort by score descending (higher = better structural fidelity)
    group_scores.sort(key=lambda x: x[1], reverse=True)

    return group_scores


# ============================================================
# Standalone CLI for evaluation
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate structural fidelity of synthetic tabular data"
    )
    parser.add_argument(
        "--real", required=True, help="Path to real data CSV"
    )
    parser.add_argument(
        "--synthetic", required=True, help="Path to synthetic data CSV"
    )
    parser.add_argument(
        "--dataname", type=str, default=None,
        help="Dataset name (to load info.json for column types)"
    )
    parser.add_argument(
        "--max_features", type=int, default=None,
        help="Max features to test (None = all)"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=50,
        help="Number of RF trees"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-feature scores"
    )

    args = parser.parse_args()

    real_df = pd.read_csv(args.real)
    syn_df = pd.read_csv(args.synthetic)

    info = None
    if args.dataname:
        info_path = f"data/{args.dataname}/info.json"
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)

    print(f"Real data: {real_df.shape}")
    print(f"Synthetic data: {syn_df.shape}")
    print(f"Computing structural fidelity...")
    print()

    result = compute_structural_fidelity(
        real_df,
        syn_df,
        info=info,
        max_features_to_test=args.max_features,
        n_estimators=args.n_estimators,
        verbose=args.verbose,
    )

    print(f"\n{'='*50}")
    print(f"Structural Fidelity Score: {result['score']:.4f}")
    print(f"Features tested: {result['details']['n_features_tested']}/{result['details']['n_features_total']}")
    print(f"{'='*50}")

    if result["details"]["classification"]:
        print(f"\nClassification features:")
        for col, scores in result["details"]["classification"].items():
            print(f"  {col}: syn={scores['syn_trained']:.4f}, real={scores['real_trained']:.4f}, ratio={scores['ratio']:.4f}")

    if result["details"]["regression"]:
        print(f"\nRegression features:")
        for col, scores in result["details"]["regression"].items():
            print(f"  {col}: syn_rmse={scores['syn_rmse']:.4f}, real_rmse={scores['real_rmse']:.4f}, ratio={scores['ratio']:.4f}")
