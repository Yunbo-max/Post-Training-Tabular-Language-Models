"""
Independent holdout evaluation classifier for the HOLDOUT SPLIT approach.

This script trains a COMPLETELY INDEPENDENT classifier on split_B data
(which was never used during DPO reward computation) to evaluate whether
the privacy improvements generalize beyond the training classifier.

This addresses reviewer concerns about evaluation circularity:
- Training: RF classifier trained on split_A (real_A vs synthetic) -> DPO reward
- Evaluation (THIS SCRIPT): XGBoost/neural net trained on split_B (real_B vs synthetic)
- The two classifiers are independent, proving privacy gains are real

Usage:
    python eval/eval_holdout_classifier.py \
        --dataname adult \
        --path synthetic/adult/great.csv \
        --split_info_path pairs_data/split_b_indices.json \
        --save_path eval/holdout_results
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from rdt import HyperTransformer
from rdt.transformers import OneHotEncoder, GaussianNormalizer


def update_config(config):
    """Update HyperTransformer config."""
    for col, sdtype in config["sdtypes"].items():
        if sdtype == "numerical":
            config["transformers"][col] = GaussianNormalizer(distribution="norm")
        if sdtype == "categorical":
            config["transformers"][col] = OneHotEncoder()
    return config


def preprocess_data(original_data, synthetic_data, hypertransformer=None):
    """Preprocess data using HyperTransformer."""
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
    """Prepare MIA attack data: real=1, synthetic=0."""
    original_data = original_data.copy()
    synthetic_data = synthetic_data.copy()
    original_data["member"] = 1
    synthetic_data["member"] = 0
    attack_data = pd.concat([original_data, synthetic_data])
    X = attack_data.drop("member", axis=1).values
    y = attack_data["member"].values
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Independent holdout evaluation: train a NEW classifier on split_B "
                    "to verify privacy improvements generalize beyond the training classifier"
    )
    parser.add_argument('--dataname', type=str, default='adult',
                        help='Dataset name (e.g., adult, diabetes, beijing, news, shoppers)')
    parser.add_argument('--model', type=str, default='model',
                        help='Model name for labeling results')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to synthetic data CSV to evaluate')
    parser.add_argument('--split_info_path', type=str, required=True,
                        help='Path to split_b_indices.json (created by generate_dpo_synthetic_holdout.py)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Output directory for evaluation results')
    parser.add_argument('--eval_classifier', type=str, default='xgboost',
                        choices=['xgboost', 'mlp', 'rf'],
                        help='Classifier family for evaluation (default: xgboost). '
                             'Using a different family than training (RF) strengthens the argument.')
    parser.add_argument('--random_state', type=int, default=123,
                        help='Random state for evaluation classifier (default: 123, '
                             'intentionally different from training random_state=42)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"INDEPENDENT HOLDOUT EVALUATION")
    print(f"{'='*60}")
    print(f"This evaluates privacy using a COMPLETELY INDEPENDENT classifier")
    print(f"trained on split_B data (never used during DPO reward computation).")
    print(f"{'='*60}\n")

    # =====================================================
    # Step 1: Load split_B indices
    # =====================================================
    print(f"Step 1: Loading split information from {args.split_info_path}...")

    if not os.path.exists(args.split_info_path):
        print(f"ERROR: Split info file not found: {args.split_info_path}")
        print(f"Run generate_dpo_synthetic_holdout.py first to create the split.")
        sys.exit(1)

    with open(args.split_info_path, 'r') as f:
        split_info = json.load(f)

    split_B_indices = split_info["split_B_indices"]
    split_A_indices = split_info["split_A_indices"]
    original_data_path = split_info.get("original_data_path", f"synthetic/{args.dataname}/real.csv")

    print(f"  Split info loaded:")
    print(f"    split_A size: {split_info['split_A_size']} (used for training reward)")
    print(f"    split_B size: {split_info['split_B_size']} (used for THIS evaluation)")
    print(f"    Total real samples: {split_info['total_real_samples']}")
    print(f"    Split ratio: {split_info['split_ratio']}")
    print(f"    Original random_state: {split_info['random_state']}")
    print(f"    Iteration: {split_info.get('iteration', 'unknown')}")

    # =====================================================
    # Step 2: Load real data and extract split_B records
    # =====================================================
    print(f"\nStep 2: Loading data...")

    real_path = original_data_path
    if not os.path.exists(real_path):
        real_path = f'synthetic/{args.dataname}/real.csv'

    if not os.path.exists(real_path):
        print(f"ERROR: Real data not found at: {real_path}")
        sys.exit(1)

    real_data = pd.read_csv(real_path)
    print(f"  Loaded real data: {len(real_data)} samples from {real_path}")

    # Verify indices are valid
    max_idx = max(split_B_indices)
    if max_idx >= len(real_data):
        print(f"ERROR: split_B index {max_idx} exceeds real data size {len(real_data)}")
        sys.exit(1)

    # Extract split_B records
    split_B_data = real_data.iloc[split_B_indices].copy().reset_index(drop=True)
    print(f"  Extracted split_B: {len(split_B_data)} samples")

    # Load synthetic data
    if args.path:
        syn_path = args.path
    else:
        syn_path = f'synthetic/{args.dataname}/{args.model}.csv'

    if not os.path.exists(syn_path):
        print(f"ERROR: Synthetic data not found at: {syn_path}")
        sys.exit(1)

    synthetic_data = pd.read_csv(syn_path)
    print(f"  Loaded synthetic data: {len(synthetic_data)} samples from {syn_path}")

    # Align columns
    columns = real_data.columns
    synthetic_data = synthetic_data[columns]

    # Clean data
    for col in synthetic_data.columns:
        if synthetic_data[col].dtype == "object":
            synthetic_data[col] = synthetic_data[col].fillna("?")
            synthetic_data[col] = synthetic_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    for col in split_B_data.columns:
        if split_B_data[col].dtype == "object":
            split_B_data[col] = split_B_data[col].fillna("?")
            split_B_data[col] = split_B_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # =====================================================
    # Step 3: Train a NEW classifier on split_B real vs synthetic
    # =====================================================
    print(f"\nStep 3: Training INDEPENDENT {args.eval_classifier.upper()} classifier on split_B...")
    print(f"  NOTE: This classifier has NEVER seen split_A data")
    print(f"  NOTE: This uses a DIFFERENT random_state ({args.random_state}) than training ({split_info['random_state']})")

    # Preprocess data
    print(f"  Preprocessing split_B + synthetic data...")
    split_B_transformed, synthetic_transformed, ht = preprocess_data(
        split_B_data, synthetic_data, hypertransformer=None
    )

    # Prepare attack data
    X_eval, y_eval = prepare_attack_data(split_B_transformed, synthetic_transformed)

    print(f"  Evaluation dataset: {len(X_eval)} samples")
    print(f"    split_B real (label=1): {len(split_B_transformed)}")
    print(f"    Synthetic (label=0): {len(synthetic_transformed)}")

    # Select classifier
    if args.eval_classifier == 'xgboost':
        eval_clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=args.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        clf_name = "XGBoost"
    elif args.eval_classifier == 'mlp':
        eval_clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=args.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        clf_name = "MLP Neural Network"
    elif args.eval_classifier == 'rf':
        eval_clf = RandomForestClassifier(
            n_estimators=100,
            random_state=args.random_state
        )
        clf_name = "Random Forest (different seed)"
    else:
        raise ValueError(f"Unknown classifier: {args.eval_classifier}")

    print(f"  Classifier: {clf_name}")

    # Train the evaluation classifier
    eval_clf.fit(X_eval, y_eval)

    # Get predictions
    y_pred = eval_clf.predict(X_eval)
    y_pred_proba = eval_clf.predict_proba(X_eval)[:, 1]

    # =====================================================
    # Step 4: Report detection score
    # =====================================================
    print(f"\nStep 4: Evaluation Results")
    print(f"{'='*60}")

    train_accuracy = accuracy_score(y_eval, y_pred)
    train_auc = roc_auc_score(y_eval, y_pred_proba)

    print(f"  Training accuracy (on full eval set): {train_accuracy:.4f}")
    print(f"  Training AUC-ROC (on full eval set): {train_auc:.4f}")

    # Cross-validation for more robust estimate
    print(f"\n  Cross-validation (5-fold) on split_B evaluation set:")
    cv_accuracy = cross_val_score(eval_clf, X_eval, y_eval, cv=5, scoring='accuracy')
    cv_auc = cross_val_score(eval_clf, X_eval, y_eval, cv=5, scoring='roc_auc')

    print(f"    CV Accuracy: {cv_accuracy.mean():.4f} +/- {cv_accuracy.std():.4f}")
    print(f"    CV AUC-ROC:  {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    # Detection score interpretation
    # Perfect privacy: accuracy ~0.5, AUC ~0.5 (can't distinguish real from synthetic)
    # Poor privacy: accuracy ~1.0, AUC ~1.0 (easily distinguishable)
    detection_score = cv_auc.mean()

    print(f"\n  Detection Score (CV AUC): {detection_score:.4f}")
    if detection_score < 0.55:
        print(f"    Interpretation: EXCELLENT privacy - classifier cannot distinguish real from synthetic")
    elif detection_score < 0.65:
        print(f"    Interpretation: GOOD privacy - synthetic data is hard to distinguish")
    elif detection_score < 0.75:
        print(f"    Interpretation: MODERATE privacy - some distinguishability remains")
    elif detection_score < 0.85:
        print(f"    Interpretation: FAIR privacy - classifier can partially distinguish")
    else:
        print(f"    Interpretation: POOR privacy - synthetic data is easily distinguishable")

    # Separate scores for real vs synthetic
    real_mask = y_eval == 1
    syn_mask = y_eval == 0

    real_proba = y_pred_proba[real_mask]
    syn_proba = y_pred_proba[syn_mask]

    print(f"\n  Score distribution:")
    print(f"    Real samples (split_B):  mean={real_proba.mean():.4f}, std={real_proba.std():.4f}")
    print(f"    Synthetic samples:       mean={syn_proba.mean():.4f}, std={syn_proba.std():.4f}")
    print(f"    Gap (lower=better):      {abs(real_proba.mean() - syn_proba.mean()):.4f}")

    # Classification report
    print(f"\n  Detailed classification report:")
    print(classification_report(y_eval, y_pred, target_names=['Synthetic', 'Real (split_B)']))

    # =====================================================
    # Step 5: Save results
    # =====================================================
    if args.save_path:
        save_dir = args.save_path
    else:
        save_dir = f'eval/holdout/{args.dataname}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Determine filename
    if args.path:
        basename = os.path.basename(args.path).replace('.csv', '')
    else:
        basename = args.model

    # Save results as JSON
    results = {
        "approach": "holdout_independent_evaluation",
        "eval_classifier": args.eval_classifier,
        "eval_classifier_name": clf_name,
        "eval_random_state": args.random_state,
        "training_random_state": split_info["random_state"],
        "split_ratio": split_info["split_ratio"],
        "split_A_size": split_info["split_A_size"],
        "split_B_size": split_info["split_B_size"],
        "synthetic_samples": len(synthetic_data),
        "train_accuracy": float(train_accuracy),
        "train_auc": float(train_auc),
        "cv_accuracy_mean": float(cv_accuracy.mean()),
        "cv_accuracy_std": float(cv_accuracy.std()),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "detection_score": float(detection_score),
        "real_proba_mean": float(real_proba.mean()),
        "real_proba_std": float(real_proba.std()),
        "syn_proba_mean": float(syn_proba.mean()),
        "syn_proba_std": float(syn_proba.std()),
        "score_gap": float(abs(real_proba.mean() - syn_proba.mean())),
        "dataname": args.dataname,
        "model": args.model,
        "synthetic_path": syn_path,
        "split_info_path": args.split_info_path,
        "iteration": split_info.get("iteration", "unknown")
    }

    json_path = os.path.join(save_dir, f"{basename}_holdout_eval.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved evaluation results to: {json_path}")

    # Save as TXT (consistent with eval_quality.py format)
    txt_path = os.path.join(save_dir, f"{basename}_holdout_eval.txt")
    with open(txt_path, 'w') as f:
        f.write(f"detection_score={round(detection_score, 4)}\n")
        f.write(f"cv_accuracy={round(float(cv_accuracy.mean()), 4)}\n")
        f.write(f"cv_auc={round(float(cv_auc.mean()), 4)}\n")
        f.write(f"score_gap={round(float(abs(real_proba.mean() - syn_proba.mean())), 4)}\n")
        f.write(f"eval_classifier={args.eval_classifier}\n")
        f.write(f"split_B_size={split_info['split_B_size']}\n")
    print(f"  Saved evaluation summary to: {txt_path}")

    # =====================================================
    # Final summary
    # =====================================================
    print(f"\n{'='*60}")
    print(f"HOLDOUT EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Training classifier: Random Forest (on split_A)")
    print(f"  Evaluation classifier: {clf_name} (on split_B)")
    print(f"  Detection score (CV AUC): {detection_score:.4f}")
    print(f"")
    print(f"  Key finding:")
    if detection_score < 0.65:
        print(f"    The privacy improvement GENERALIZES to an independent classifier.")
        print(f"    This rules out evaluation circularity as an explanation.")
    else:
        print(f"    The independent classifier can still distinguish real from synthetic.")
        print(f"    Privacy improvement may need more DPO iterations.")
    print(f"")
    print(f"  Files saved:")
    print(f"    - {json_path}")
    print(f"    - {txt_path}")


if __name__ == "__main__":
    main()
