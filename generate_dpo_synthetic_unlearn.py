"""
Generate DPO pairs for subgroup UNLEARNING.

Reward logic:
  reward(row) = +1.0 if row does NOT match forget_condition
  reward(row) = -1.0 if row DOES match forget_condition

DPO pairs are formed by sorting on this reward:
  - chosen:   non-matching (desirable, keep generating)
  - rejected: matching    (forget-target, stop generating)

Model learns to prefer non-matching → forgets the target subgroup.

Forget-condition spec (--forget_cond):
  - A pandas-eval-style string, e.g.:
      "education == 'Doctorate'"
      "(age < 25) & (`marital.status` == 'Never-married')"
      "year < 2014"
  - Backticks (`) allowed for column names with dots or spaces.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch


def row_to_str(row):
    """GReaT row format: 'col1 is val1, col2 is val2, ...'."""
    return ", ".join([f"{k} is {v}" for k, v in row.items()])


def detect_dataname(path):
    """Extract <dataname> from `.../synthetic/<dataname>/<file>.csv`,
    `.../data/<dataname>/...`, or `.../experiments/drift_<NAME>/...`."""
    parts = os.path.normpath(path).split(os.sep)
    for anchor in ("synthetic", "data"):
        if anchor in parts:
            i = parts.index(anchor)
            if i + 1 < len(parts):
                return parts[i + 1]
    for p in parts:
        if p.startswith("drift_") and p != "drift_data":
            return p[len("drift_"):] + "_drift"
    for name in ["adult", "diabetes", "beijing", "news", "shoppers", "default", "magic",
                 "elec2", "german_credit"]:
        if name in path:
            return name
    return "adult"


def apply_forget_cond(df, cond_str):
    """Return a boolean array of whether each row matches the forget condition.

    Uses pandas DataFrame.eval; supports backticks for column names with dots.
    """
    try:
        mask = df.eval(cond_str)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        return mask.values
    except Exception as e:
        print(f"[forget-cond] eval failed: {e}")
        print(f"[forget-cond] cond_str: {cond_str}")
        print(f"[forget-cond] columns: {list(df.columns)}")
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--top_percent", type=float, default=100.0)
    ap.add_argument("--model_type", default="great", choices=["great", "great_gemma"])
    ap.add_argument("--forget_cond", required=True,
                    help="pandas-eval string defining the forget subgroup")
    ap.add_argument("--use_existing_synthetic", action="store_true",
                    help="use --synthetic file instead of sampling fresh")
    args = ap.parse_args()

    print("=" * 60)
    print(f"DPO PAIR GENERATION (UNLEARNING)")
    print(f"forget_cond: {args.forget_cond}")
    print("=" * 60)

    synthetic_data = pd.read_csv(args.synthetic)
    original_data = pd.read_csv(args.original)
    print(f"Loaded synthetic={len(synthetic_data)} original={len(original_data)}")

    # Align synthetic schema to original
    synthetic_data = synthetic_data[original_data.columns]
    for col in synthetic_data.columns:
        if synthetic_data[col].dtype == "object":
            synthetic_data[col] = synthetic_data[col].fillna("?")
            synthetic_data[col] = synthetic_data[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x)
    for col in original_data.columns:
        if original_data[col].dtype == "object":
            original_data[col] = original_data[col].fillna("?")
            original_data[col] = original_data[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x)

    # --------------------------------------------------------------
    # Generate fresh synthetic samples from the current checkpoint
    # --------------------------------------------------------------
    dataname = detect_dataname(args.synthetic)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = f"{curr_dir}/baselines/{args.model_type}/ckpt/{dataname}/model.pt"

    if args.use_existing_synthetic:
        new_synthetic_data = synthetic_data.copy()
    else:
        if args.model_type == "great":
            from baselines.great.models.great import GReaT
            from baselines.great.models.great_utils import _array_to_dataframe
            model_class = GReaT
            base_model_name = "distilgpt2"
        else:
            from baselines.great_gemma.models.great_gemma import GReaTGemma
            from baselines.great_gemma.models.great_utils import _array_to_dataframe
            model_class = GReaTGemma
            base_model_name = "google/gemma-2b"

        if not os.path.exists(model_save_path):
            print(f"model_save_path missing: {model_save_path} — falling back to existing synthetic")
            new_synthetic_data = synthetic_data.copy()
        else:
            try:
                great = model_class(base_model_name)
                great.model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
                great.load_finetuned_model(model_save_path)

                df = _array_to_dataframe(original_data, columns=None)
                great._update_column_information(df)
                great._update_conditional_information(df, conditional_col=None)

                # Cap n_samples for efficiency — 3000 is plenty for preference-pair formation
                n_samples = min(len(original_data), 3000)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Sampling {n_samples} NEW synthetic rows on {device} ...")
                new_synthetic_data = great.sample(
                    n_samples, k=100, device=device, temperature=0.7)
                print(f"Generated {len(new_synthetic_data)} rows")

                if len(new_synthetic_data) < max(50, n_samples // 100):
                    print(f"⚠️ Too few valid rows ({len(new_synthetic_data)}); falling back to existing synthetic")
                    new_synthetic_data = synthetic_data.copy()

                new_syn_path = args.synthetic.replace(".csv", "_new.csv")
                new_synthetic_data.to_csv(new_syn_path, index=False)
                print(f"Saved new synthetic to {new_syn_path}")
            except Exception as e:
                print(f"sample() error: {e}  — falling back to existing synthetic")
                new_synthetic_data = synthetic_data.copy()

    # --------------------------------------------------------------
    # Score each sample: -1 if matches forget cond, +1 otherwise
    # --------------------------------------------------------------
    print(f"\nApplying forget condition: {args.forget_cond}")
    forget_mask = apply_forget_cond(new_synthetic_data, args.forget_cond)
    rewards = np.where(forget_mask, -1.0, 1.0)

    n_forget = int(forget_mask.sum())
    n_keep = int((~forget_mask).sum())
    print(f"  matched forget: {n_forget}  ({100.0*n_forget/len(forget_mask):.2f}%)")
    print(f"  non-matching (keep): {n_keep}")

    if n_forget == 0:
        print("[warn] zero rows match forget_cond — model may already have unlearned, "
              "or cond is too strict. Creating empty pair set.")
    if n_keep == 0:
        print("[warn] all rows match forget_cond — no retain samples. Aborting.")
        sys.exit(2)

    # Sort descending: +1 rewards first, then -1
    sorted_indices = np.argsort(rewards)[::-1]

    # Row strings
    print("Converting synthetic rows to GReaT format strings...")
    row_strs = [row_to_str(new_synthetic_data.iloc[i]) for i in range(len(new_synthetic_data))]

    # --------------------------------------------------------------
    # Build pairs: chosen=keep, rejected=forget (opposite-ends pairing)
    # --------------------------------------------------------------
    # We pair up to min(n_forget, n_keep) — one chosen and one rejected each.
    num_pairs = min(n_forget, n_keep)
    print(f"\nCreating {num_pairs} DPO pairs (chosen=keep, rejected=forget)")

    keep_indices = np.where(rewards > 0)[0]      # +1 reward
    forget_indices = np.where(rewards < 0)[0]    # -1 reward

    # Shuffle to avoid ordering bias
    rng = np.random.default_rng(0)
    rng.shuffle(keep_indices)
    rng.shuffle(forget_indices)

    dpo_data = []
    for i in range(num_pairs):
        ci = int(keep_indices[i])
        ri = int(forget_indices[i])
        dpo_data.append({
            "instruction": "",
            "input": "",
            "chosen": row_strs[ci],
            "rejected": row_strs[ri],
            "chosen_source": "new_synthetic",
            "rejected_source": "new_synthetic",
            "chosen_reward": float(rewards[ci]),
            "rejected_reward": float(rewards[ri]),
            "chosen_is_forget": False,
            "rejected_is_forget": True,
        })

    # top_percent currently unused (all pairs already perfectly separated)
    # but keep the arg for compatibility with iterative_dpo_trainer.py

    print(f"Total pairs: {len(dpo_data)}")

    # --------------------------------------------------------------
    # Train/test split (85/15)
    # --------------------------------------------------------------
    n_total = len(dpo_data)
    n_test = max(1, int(n_total * 0.15))
    rng.shuffle(dpo_data)
    test_pairs = dpo_data[:n_test]
    train_pairs = dpo_data[n_test:]

    os.makedirs(os.path.dirname(args.train_json) or ".", exist_ok=True)
    with open(args.train_json, "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open(args.test_json, "w") as f:
        json.dump(test_pairs, f, indent=2)

    print(f"\nSaved {len(train_pairs)} train pairs -> {args.train_json}")
    print(f"Saved {len(test_pairs)} test pairs -> {args.test_json}")

    # Quick CSV summary
    pairs_dir = os.path.dirname(args.train_json)
    iter_m = re.search(r"iter_(\d+)", args.train_json)
    iter_id = iter_m.group(1) if iter_m else "unknown"
    summary = pd.DataFrame({
        "sample_id": range(len(rewards)),
        "reward": rewards,
        "is_forget": forget_mask.astype(int),
    })
    summary_path = os.path.join(pairs_dir, f"unlearn_scores_iter_{iter_id}.csv")
    try:
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary CSV -> {summary_path}")
    except Exception as e:
        print(f"summary CSV save failed: {e}")


if __name__ == "__main__":
    main()
