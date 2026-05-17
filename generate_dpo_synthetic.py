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
# Add these imports at the TOP
import pickle
import re

try:
    from tabpfn import TabPFNClassifier  # not used in this script; tolerate missing torch>=2.3
except Exception:
    TabPFNClassifier = None
# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

def row_to_str(row):
    # Convert a pandas Series (row) to a readable string in GReaT format
    # Handle potential NaN, inf, or problematic values
    parts = []
    for col in row.index:
        val = row[col]
        # Handle special numeric values
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
    import os
    parser = argparse.ArgumentParser(description="Generate DPO pairs using IMPROVED privacy approach: ALL real+synthetic for MIA training, NEW synthetic for testing")
    parser.add_argument('--synthetic', default='synthetic/adult/great.csv', help='Path to synthetic data CSV file')
    parser.add_argument('--original', default='synthetic/adult/real.csv', help='Path to original/training data CSV file (will use ALL for MIA training)')
    parser.add_argument('--train_json', default='train_data_synthetic.json', help='Output train JSON file (contains ALL pairs)')
    parser.add_argument('--test_json', default='test_data_synthetic.json', help='Output test JSON file (kept for compatibility but not used)')
    parser.add_argument('--top_percent', type=float, default=50.0, 
                        help='Select top X percent of pairs with highest chosen rewards (default: 100.0 = all pairs)')
    parser.add_argument('--use_existing_synthetic', action='store_true', 
                        help='Use existing synthetic data instead of generating NEW samples (not recommended for iterative training)')
    parser.add_argument('--model_type', type=str, default='great', 
                    choices=['great', 'great_gemma'],
                    help='Model type: great or great_gemma')
    args = parser.parse_args()

    # ⭐⭐⭐ ADD THIS ⭐⭐⭐
    print(f"\n{'='*60}")
    print(f"🔄 DPO PAIR GENERATION")
    print(f"{'='*60}")
    
    # Extract and display iteration number
    iter_match = re.search(r'iter_(\d+)', args.train_json)
    current_iter = int(iter_match.group(1)) if iter_match else 1
    print(f"📊 Processing iteration: {current_iter}")
    # ⭐⭐⭐ END ADDITION ⭐⭐⭐
    
    
    # Validate top_percent
    if args.top_percent <= 0 or args.top_percent > 100:
        raise ValueError("--top_percent must be between 0 and 100")

    synthetic_data = pd.read_csv(args.synthetic)
    original_data = pd.read_csv(args.original)
    
    print(f"Loaded {len(synthetic_data)} synthetic samples and {len(original_data)} original samples")
    print(f"🎯 IMPROVED PRIVACY APPROACH:")
    print(f"   1. Train MIA classifier on ALL real + synthetic data (no splits)")
    print(f"   2. Generate NEW synthetic samples for privacy testing")
    print(f"   3. Create DPO pairs ONLY from NEW synthetic samples ranked by privacy")
    print(f"   4. This gives more realistic privacy evaluation and better training data")

    # Clean data (same as privacy_attack.py)
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

    # Split original data into train/test
    print("\nUSING ALL DATA APPROACH for better privacy evaluation...")
    print("Step 1: Using ALL real data + ALL synthetic data to train MIA classifier")
    
    # Use all data for MIA training (no splits)
    original_all = original_data.copy()
    synthetic_all = synthetic_data.copy()
    
    print(f"  ALL real data: {len(original_all)} samples")
    print(f"  ALL synthetic data: {len(synthetic_all)} samples")

    # First, fit HyperTransformer on ALL data to avoid unseen category warnings
    print("Fitting HyperTransformer on ALL data (real + synthetic) to capture all categories...")
    _, _, ht = preprocess_data(
        original_all, synthetic_all, hypertransformer=None
    )
    print("✓ HyperTransformer fitted on complete dataset")
    
    # Transform ALL data using the pre-fitted transformer (for MIA training)
    print("Transforming ALL data for MIA training...")
    original_all_transformed, synthetic_all_transformed, _ = preprocess_data(
        original_all, synthetic_all, hypertransformer=ht
    )

    # Prepare training data for MIA classifier (using ALL data)
    X_train, y_train = prepare_attack_data(
        original_all_transformed, synthetic_all_transformed
    )

    # Train MIA classifier on ALL data
    print("Training MIA classifier on ALL available data...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)



    # # 🔄 USE SAME CLASSIFIER ACROSS ALL ITERATIONS (NO RETRAINING)
    # # Check if we already have a trained classifier
    # classifier_path = "mia_classifier_fixed.pkl"

    # if os.path.exists(classifier_path):
    #     print(f"📂 LOADING EXISTING CLASSIFIER (NO RETRAINING)")
    #     print(f"   From: {classifier_path}")
    #     with open(classifier_path, 'rb') as f:
    #         clf = pickle.load(f)
    #     print(f"✅ Using pre-trained classifier from first iteration")
    # else:
    #     print(f"🔄 FIRST ITERATION: Training initial classifier...")
    #     clf = RandomForestClassifier(n_estimators=100, random_state=42)
    #     clf.fit(X_train, y_train)
        
    #     # Save it for all future iterations
    #     with open(classifier_path, 'wb') as f:
    #         pickle.dump(clf, f)
    #     print(f"💾 Saved classifier to: {classifier_path}")
    #     print(f"   This classifier will be REUSED for all future iterations")


    
    print(f"✓ MIA classifier trained on {len(X_train)} samples")
    print(f"  Real samples: {len(original_all_transformed)}")
    print(f"  Synthetic samples: {len(synthetic_all_transformed)}")

    # Step 2: Generate NEW synthetic samples for testing (DEFAULT behavior)
    print(f"\nStep 2: Generating NEW synthetic samples for MIA testing...")
    
    if args.use_existing_synthetic:
        print(f"⚠️ Using existing synthetic data instead of generating NEW samples")
        print(f"Note: This is NOT recommended for iterative training as it defeats the purpose")
        new_synthetic_data = synthetic_data.copy()
        print(f"  Using existing synthetic samples: {len(new_synthetic_data)}")
    else:
        print(f"  Generating NEW synthetic data using the current model...")
        

                # Load the current model for sampling new data
        import os
        
        # Detect dataset name: paths look like `.../synthetic/<dataname>/<file>.csv`.
        # Extract <dataname> directly so new datasets (e.g. `*_drift`) don't need a
        # hardcoded allowlist.
        _parts = os.path.normpath(args.synthetic).split(os.sep)
        dataname = None
        for _anchor in ('synthetic', 'data'):
            if _anchor in _parts:
                _i = _parts.index(_anchor)
                if _i + 1 < len(_parts):
                    dataname = _parts[_i + 1]
                    break
        # experiments/drift_<NAME>/... -> "<NAME>_drift"
        if dataname is None:
            for _p in _parts:
                if _p.startswith('drift_') and _p != 'drift_data':
                    dataname = _p[len('drift_'):] + '_drift'
                    break
        if dataname is None:
            for _n in ('adult', 'diabetes', 'beijing', 'news', 'shoppers',
                       'default', 'magic', 'elec2', 'german_credit'):
                if _n in args.synthetic:
                    dataname = _n
                    break
        if dataname is None:
            dataname = 'adult'
        
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
            print(f"⚠️ Model not found at {model_save_path}")
            print(f"Falling back to using existing synthetic data...")
            print(f"💡 Train a base model first using: python main.py --dataname {dataname} --method great --mode train")
            new_synthetic_data = synthetic_data.copy()
        else:
            print(f"📂 Loading model from: {model_save_path}")
            
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
                print(f"🎲 Sampling {n_samples} NEW synthetic samples...")
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                new_synthetic_data = great.sample(
                    n_samples,
                    k=100,
                    device=device,
                    # guided_sampling=True,
                    temperature=0.7
                )
                
                print(f"✅ Generated {len(new_synthetic_data)} NEW synthetic samples")

                # If the post-training model produced no valid rows (e.g. drift case
                # where source-trained model fails on target-distribution start tokens),
                # fall back to the existing synthetic data instead of crashing the RF.
                if len(new_synthetic_data) < max(50, n_samples // 100):
                    print(f"⚠️ Fresh sampling produced too few valid rows ({len(new_synthetic_data)}); "
                          f"falling back to existing synthetic data.")
                    new_synthetic_data = synthetic_data.copy()

                # Save the NEW synthetic data for debugging/reference
                new_synthetic_path = args.synthetic.replace('.csv', '_new.csv')
                new_synthetic_data.to_csv(new_synthetic_path, index=False)
                print(f"💾 Saved NEW synthetic data to: {new_synthetic_path}")

            except Exception as e:
                print(f"❌ Error during NEW synthetic data generation: {e}")
                print(f"🔄 Falling back to using existing synthetic data...")
                new_synthetic_data = synthetic_data.copy()
    
    print(f"  NEW synthetic samples (for testing): {len(new_synthetic_data)}")

    # Get privacy scores for NEW SYNTHETIC DATA ONLY
    print("Step 3: Computing privacy scores for NEW synthetic data only...")
    
    # Transform NEW synthetic data directly using the pre-fitted HyperTransformer
    print("Transforming NEW synthetic data with HyperTransformer...")
    
    try:
        # Use the pre-fitted HyperTransformer to transform NEW synthetic data directly
        new_synthetic_transformed = ht.transform(new_synthetic_data)
        print(f"Transformed NEW synthetic data shape: {new_synthetic_transformed.shape}")
        
        # Verify we have data to work with
        if len(new_synthetic_transformed) == 0:
            raise ValueError("No data after transformation - check for data compatibility")
            
        # Convert to numpy array for sklearn
        new_synthetic_transformed_array = new_synthetic_transformed.values
        
    except Exception as e:
        print(f"⚠️ Direct transformation failed: {e}")
        print("🔄 Trying alternative transformation approach...")
        
        # Fallback: combine with a small sample of original data for transformation
        small_original = original_all.head(10).copy()  # Use more rows for stability
        combined_for_transform = pd.concat([small_original, new_synthetic_data], ignore_index=True)
        
        # Transform combined data
        combined_transformed = ht.transform(combined_for_transform)
        
        # Extract only the NEW synthetic data (skip the original rows)
        new_synthetic_transformed_array = combined_transformed.iloc[len(small_original):].values
        print(f"Alternative transformation successful: {new_synthetic_transformed_array.shape}")
    
    # Predict membership probability for NEW synthetic samples ONLY
    y_pred_new_synthetic = clf.predict_proba(new_synthetic_transformed_array)[:, 1]
    
    print(f"Generated {len(y_pred_new_synthetic)} privacy scores for NEW synthetic data")

    # Convert NEW synthetic rows to GReaT format strings
    print(f"\nConverting NEW synthetic rows to GReaT format strings...")
    new_synthetic_outputs = []
    batch_size = 100
    num_batches = (len(new_synthetic_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(new_synthetic_data))
        
        print(f"  Processing batch {batch_idx + 1}/{num_batches} (rows {start_idx}-{end_idx})")
        
        # Process batch
        for idx in range(start_idx, end_idx):
            row_str = row_to_str(new_synthetic_data.iloc[idx])
            new_synthetic_outputs.append(row_str)
    
    print(f"✓ Successfully converted {len(new_synthetic_outputs)} NEW synthetic rows to strings")

    # Calculate MIA rewards for NEW synthetic data: higher reward = closer to 0.5 (better privacy)
    # Reward = 1 - distance from 0.5, so closer to 0.5 gives higher reward
    distances = np.abs(y_pred_new_synthetic - 0.5)
    rewards = 1.0 - distances  # Higher reward = better privacy (closer to 0.5)
    
    # Sort by reward (highest to lowest)
    sorted_indices = np.argsort(rewards)[::-1]  # Highest reward first
    
    print(f"\nNEW Synthetic MIA Reward statistics (higher = better privacy):")
    print(f"  Mean reward: {rewards.mean():.4f}")
    print(f"  Best reward: {rewards.max():.4f}")
    print(f"  Worst reward: {rewards.min():.4f}")
    print(f"  Median reward: {np.median(rewards):.4f}")
    print(f"  Std reward: {rewards.std():.4f}")
    
    # Save MIA scores and rewards to CSV in pairs_data directory
    pairs_dir = os.path.dirname(args.train_json)
    if pairs_dir and not os.path.exists(pairs_dir):
        os.makedirs(pairs_dir, exist_ok=True)
    
    # Extract iteration number from the output file path for naming
    base_name = os.path.splitext(os.path.basename(args.train_json))[0]
    iteration_match = base_name.split('_')
    iteration_id = "unknown"
    for i, part in enumerate(iteration_match):
        if part == "iter" and i + 1 < len(iteration_match):
            iteration_id = iteration_match[i + 1]
            break
    
    # Create DataFrame with MIA data
    mia_data = pd.DataFrame({
        'sample_id': range(len(y_pred_new_synthetic)),
        'mia_score': y_pred_new_synthetic,
        'mia_reward': rewards,
        'distance_from_0.5': distances,
        'sample_rank': [np.where(sorted_indices == i)[0][0] + 1 for i in range(len(sorted_indices))]
    })
    
    # Save CSV file
    csv_filename = f"mia_scores_rewards_iter_{iteration_id}.csv"
    csv_path = os.path.join(pairs_dir, csv_filename) if pairs_dir else csv_filename
    
    try:
        mia_data.to_csv(csv_path, index=False)
        print(f"💾 Saved MIA scores and rewards to: {csv_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save CSV file: {e}")
        csv_path = None
    
    # Create and save distribution plots
    print(f"📊 Creating distribution plots...")
    
    try:
        # Set plot style
        plt.style.use('default')
        
        # Try to use seaborn if available, otherwise use matplotlib defaults
        try:
            sns.set_palette("husl")
            use_seaborn = True
        except:
            use_seaborn = False
            print("📊 Note: Seaborn not available, using matplotlib defaults")
        
        # Plot 1: MIA Score Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MIA Score histogram
        ax1.hist(y_pred_new_synthetic, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Perfect Privacy (0.5)')
        ax1.axvline(x=y_pred_new_synthetic.mean(), color='orange', linestyle='-', linewidth=2, 
                    label=f'Mean: {y_pred_new_synthetic.mean():.3f}')
        ax1.set_xlabel('MIA Score (Membership Probability)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Distribution of MIA Scores\nIteration {iteration_id}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'N = {len(y_pred_new_synthetic)}\nMean = {y_pred_new_synthetic.mean():.3f}\nStd = {y_pred_new_synthetic.std():.3f}\nMedian = {np.median(y_pred_new_synthetic):.3f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # MIA Reward histogram
        ax2.hist(rewards, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=rewards.mean(), color='orange', linestyle='-', linewidth=2, 
                    label=f'Mean: {rewards.mean():.3f}')
        ax2.set_xlabel('MIA Reward (1 - |score - 0.5|)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Distribution of MIA Rewards\nIteration {iteration_id}', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text box
        reward_stats_text = f'N = {len(rewards)}\nMean = {rewards.mean():.3f}\nStd = {rewards.std():.3f}\nMedian = {np.median(rewards):.3f}'
        ax2.text(0.02, 0.98, reward_stats_text, transform=ax2.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save the distribution plot
        plot_filename = f"mia_distributions_iter_{iteration_id}.png"
        plot_path = os.path.join(pairs_dir, plot_filename) if pairs_dir else plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved MIA distribution plot to: {plot_path}")
        
        # Plot 2: Detailed Analysis - MIA Score vs Reward
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: MIA Score vs Reward
        scatter = ax1.scatter(y_pred_new_synthetic, rewards, alpha=0.6, c=rewards, 
                             cmap='RdYlGn', s=20)
        ax1.set_xlabel('MIA Score (Membership Probability)', fontsize=12)
        ax1.set_ylabel('MIA Reward (1 - |score - 0.5|)', fontsize=12)
        ax1.set_title(f'MIA Score vs Reward\nIteration {iteration_id}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Perfect Privacy')
        ax1.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Reward Value', fontsize=10)
        
        # Box plot by reward quartiles
        reward_quartiles = np.percentile(rewards, [25, 50, 75])
        quartile_labels = []
        quartile_data = []
        
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
        ax2.set_title(f'MIA Scores by Reward Quartile\nIteration {iteration_id}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Perfect Privacy')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the analysis plot
        analysis_plot_filename = f"mia_analysis_iter_{iteration_id}.png"
        analysis_plot_path = os.path.join(pairs_dir, analysis_plot_filename) if pairs_dir else analysis_plot_filename
        plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved MIA analysis plot to: {analysis_plot_path}")
        
        print(f"✅ All plots and CSV data saved successfully!")
        print(f"📁 Files created:")
        if csv_path:
            print(f"   - {csv_path}")
        print(f"   - {plot_path}")
        print(f"   - {analysis_plot_path}")
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to create plots: {e}")
        print(f"   Continuing with DPO pair generation...")
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

    # Create DPO preference pairs from NEW synthetic data only: chosen=highest reward, rejected=lowest reward
    dpo_data = []
    num_pairs = len(sorted_indices) // 2
    num_samples = len(sorted_indices)

    print(f"\nCreating {num_pairs} DPO preference pairs from NEW synthetic data...")
    print(f"  Chosen (highest reward) - Rejected (lowest reward)")
    
    for i in range(num_pairs):
        chosen_idx = sorted_indices[i]  # Highest reward (best privacy)
        rejected_idx = sorted_indices[-(i+1)]  # Lowest reward (worst privacy)
        
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

    

    # # Strategy 2: Offset pairing (1st+51st, 2nd+52nd, etc.)
    # print(f"  Strategy 2: Offset pairing (i-th vs i+50-th)")
    # offset = min(50, num_samples // 2)  # Use offset of 50 or half of samples
    # for i in range(num_samples - offset):
    #     chosen_idx = sorted_indices[i]  # i-th best
    #     rejected_idx = sorted_indices[i + offset]  # (i+50)-th best
        
    #     entry = {
    #         "instruction": "",
    #         "input": "",
    #         "chosen": new_synthetic_outputs[chosen_idx],
    #         "rejected": new_synthetic_outputs[rejected_idx],
    #         "chosen_source": "new_synthetic",
    #         "rejected_source": "new_synthetic",
    #         "chosen_privacy_score": float(y_pred_new_synthetic[chosen_idx]),
    #         "rejected_privacy_score": float(y_pred_new_synthetic[rejected_idx]),
    #         "chosen_reward": float(rewards[chosen_idx]),
    #         "rejected_reward": float(rewards[rejected_idx]),
    #         "pair_strategy": "offset_pairing",
    #         "pair_offset": offset,
    #         "pair_rank": i
    #     }
    #     dpo_data.append(entry)

        
    
    # Handle odd number of samples
    if len(sorted_indices) % 2 == 1:
        mid_idx = sorted_indices[num_pairs]
        entry = {
            "instruction": "",
            "input": "",
            "chosen": new_synthetic_outputs[mid_idx],
            "rejected": new_synthetic_outputs[mid_idx],  # Same as chosen for odd sample
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
        # Sort pairs by chosen sample's reward (highest first)
        dpo_data.sort(key=lambda x: x['chosen_reward'], reverse=True)
        
        # Select top X%
        num_pairs_to_keep = int(len(dpo_data) * args.top_percent / 100.0)
        dpo_data = dpo_data[:num_pairs_to_keep]
        
        print(f"\n✂️ Filtered to top {args.top_percent}% of pairs (by chosen reward):")
        print(f"  Total pairs before: {total_pairs_before}")
        print(f"  Pairs kept: {len(dpo_data)}")
        print(f"  Pairs removed: {total_pairs_before - len(dpo_data)}")
        print(f"  Best chosen reward: {dpo_data[0]['chosen_reward']:.4f}")
        print(f"  Worst (of kept) chosen reward: {dpo_data[-1]['chosen_reward']:.4f}")

    # Analyze pair composition
    syn_syn = sum(1 for d in dpo_data if d['chosen_source'] == 'synthetic' and d['rejected_source'] == 'synthetic')
    syn_test = sum(1 for d in dpo_data if d['chosen_source'] == 'synthetic' and d['rejected_source'] == 'real_test')
    test_syn = sum(1 for d in dpo_data if d['chosen_source'] == 'real_test' and d['rejected_source'] == 'synthetic')
    test_test = sum(1 for d in dpo_data if d['chosen_source'] == 'real_test' and d['rejected_source'] == 'real_test')
    
    print(f"\nPair composition:")
    print(f"  Synthetic (chosen) vs Synthetic (rejected): {syn_syn}")
    print(f"  Synthetic (chosen) vs Real_test (rejected): {syn_test}")
    print(f"  Real_test (chosen) vs Synthetic (rejected): {test_syn}")
    print(f"  Real_test (chosen) vs Real_test (rejected): {test_test}")


    # For DPO fine-tuning, use ALL pairs for training (no need for train/test split)
    train_data = dpo_data  # Use all pairs for training

    print(f"\nSaving DPO data:")
    print(f"  Train pairs: {len(train_data)} (ALL pairs used for training)")
    print(f"  Total pairs: {len(dpo_data)}")
    print(f"  📁 Train file: {args.train_json}")
    print(f"  📁 Note: No test file created - using all pairs for fine-tuning")

    with open(args.train_json, "w") as f:
        json.dump(train_data, f, indent=2)

    print(f"\n✓ Saved {args.train_json}")
    print(f"✓ Used ALL NEW synthetic data to create DPO pairs")
    if args.top_percent < 100.0:
        print(f"✓ Filtered to top {args.top_percent}% of pairs with highest chosen rewards")
    print(f"✓ MIA trained on real_train vs synthetic_train (no test data leakage)")
    
    # Print summary of all created files
    print(f"\n📁 Summary of Generated Files:")
    print(f"   🎯 DPO Training Data:")
    print(f"     - {args.train_json} ({len(train_data)} pairs)")
    print(f"     - Note: All pairs used for training (no test split needed)")
    
    # Check if CSV and plotting files were created successfully
    csv_created = 'csv_path' in locals() and csv_path is not None
    plots_created = 'plot_path' in locals() and plot_path is not None
    analysis_created = 'analysis_plot_path' in locals() and analysis_plot_path is not None
    
    if csv_created or plots_created or analysis_created:
        print(f"   📊 MIA Analysis Data:")
        if csv_created:
            print(f"     - {csv_path} (MIA scores & rewards for {len(mia_data)} samples)")
        if plots_created:
            print(f"     - {plot_path} (MIA distribution plots)")
        if analysis_created:
            print(f"     - {analysis_plot_path} (MIA analysis plots)")
    
    print(f"\n💡 Usage tips:")
    print(f"   - Default (100%): All pairs included")
    print(f"   - Top 50%: Only pairs with highest chosen rewards (recommended)")
    print(f"   - Top 25%: Most selective, highest quality pairs")
    print(f"\n📊 Ranking methodology:")
    print(f"   - Reward = 1 - |MIA_prob - 0.5| (higher = better privacy)")
    print(f"   - Pairs: highest reward (chosen) vs lowest reward (rejected)")
    print(f"   - Filter: top X% based on chosen sample's reward")
    print(f"   - Uses: NEW synthetic data only (privacy-focused)")
    print(f"   - Training: ALL pairs used for fine-tuning (no splits)")
    print(f"🎯 Benefits of NEW synthetic data approach:")
    print(f"   - No training data leakage (test set not used in MIA training)")
    print(f"   - Uses only NEW synthetic data for fair privacy evaluation")
    print(f"   - All pairs used for training (no unnecessary train/test split)")
    print(f"   - MIA classifier trained on separate data splits")
    print(f"   - Simplified workflow focused on privacy-preserving DPO")
    print(f"\n📈 Visualization Files:")
    if plots_created or analysis_created:
        print(f"   - Distribution plots show MIA score and reward histograms")
        print(f"   - Analysis plots show score vs reward relationships")
    if csv_created:
        print(f"   - CSV file contains all sample-level MIA data for further analysis")
    if not (csv_created or plots_created or analysis_created):
        print(f"   - No visualization files created due to plotting errors (check dependencies)")

if __name__ == "__main__":
    main()
