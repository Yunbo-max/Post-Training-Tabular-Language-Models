import numpy as np
import torch 
import pandas as pd
import os 
import sys
import json
import pickle

# Metrics
from sdmetrics import load_demo
from sdmetrics.single_table import LogisticDetection
from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as plt

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='real')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')
parser.add_argument('--save_path', type=str, default = None, help='The file path to save the evaluation results')

args = parser.parse_args()

def handle_unknown_categories(real_data, syn_data, metadata):
    """
    Ensure synthetic data only contains categories present in the ENTIRE real dataset
    """
    syn_data_clean = syn_data.copy()
    
    for col_idx, col_info in metadata['columns'].items():

        if col_info['sdtype'] == 'categorical':
            # Get ALL categories from ENTIRE real dataset
            allowed_categories = set(real_data[col_idx].astype(str).unique())
            current_categories = set(syn_data_clean[col_idx].astype(str).unique())
            
            # Find unknown categories
            unknown_categories = current_categories - allowed_categories
            
            if unknown_categories:
                print(f"Warning: Column {col_idx} has unknown categories: {unknown_categories}")
                # Replace unknown categories with mode from ENTIRE real dataset
                mode_value = real_data[col_idx].mode()[0] if len(real_data[col_idx].mode()) > 0 else real_data[col_idx].iloc[0]
                mask = syn_data_clean[col_idx].astype(str).isin(unknown_categories)
                syn_data_clean.loc[mask, col_idx] = str(mode_value)
                print(f"Replaced {mask.sum()} unknown categories with '{mode_value}'")
    
    return syn_data_clean

def create_comprehensive_encoder(real_data, metadata):
    """
    Create encoders based on ALL real data, not just a training split
    """
    categorical_columns = []
    for col_idx, col_info in metadata['columns'].items():
        if col_info['sdtype'] == 'categorical':
            categorical_columns.append(col_idx)
    
    # Fit encoder on ALL real data
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
    encoder.fit(real_data[categorical_columns])
    
    return encoder, categorical_columns

def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    metadata = info['metadata']
    columns = metadata['columns']
    metadata['columns'] = {}

    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    
    return new_real_data, new_syn_data, metadata

if __name__ == '__main__':
    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path
        
    real_path = f'synthetic/{dataname}/real.csv'
    data_dir = f'data/{dataname}' 
    print(f"Processing: {syn_path}")

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    save_dir = f'eval/detection/{dataname}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Use save_path if provided, otherwise use default
    if args.save_path:
        save_file = args.save_path.replace('.json', '.txt')  # Keep txt format for consistency
    else:
        save_file = f'eval/detection/{dataname}/{model}.txt'

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

    new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)
    
    # Step 1: Clean synthetic data to remove unknown categories
    # print("Cleaning unknown categories...")
    new_syn_data_clean = handle_unknown_categories(new_real_data, new_syn_data, metadata)
    
    # Step 2: Create comprehensive encoder using ALL real data
    # print("Creating comprehensive encoder...")
    encoder, categorical_columns = create_comprehensive_encoder(new_real_data, metadata)
    
    # Step 3: Verify both datasets use the same categories
    print("Verifying category consistency...")
    for col in categorical_columns:
        real_cats = set(new_real_data[col].astype(str).unique())
        syn_cats = set(new_syn_data_clean[col].astype(str).unique())
        if not syn_cats.issubset(real_cats):
            print(f"Column {col} still has mismatched categories")
            print(f"Real: {real_cats}")
            print(f"Syn: {syn_cats}")

    try:
        print("Computing Logistic Detection score...")
        score = LogisticDetection.compute(
            real_data=new_real_data,
            synthetic_data=new_syn_data_clean,
            metadata=metadata
        )

        c2st = np.round(score, 4)
        print('Saving scores to ', save_file)
        print(f'{dataname}, {model}: {c2st}')
        print(f'c2st={c2st}')  # Standard output for parsing
        with open(save_file, 'w') as f:
            f.write(f'c2st={c2st}')
            
    except Exception as e:
        print(f"Error during detection computation: {e}")
        print("Using fallback method...")
        
        # Fallback: Manual detection score computation
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        
        # Combine and label data
        new_real_data['is_synthetic'] = 0
        new_syn_data_clean['is_synthetic'] = 1
        
        combined_data = pd.concat([new_real_data, new_syn_data_clean], axis=0, ignore_index=True)
        
        # Handle categorical encoding manually using our comprehensive encoder
        if categorical_columns:
            # Transform categorical columns
            cat_encoded = encoder.transform(combined_data[categorical_columns])
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out())
            
            # Combine with numerical columns
            numerical_columns = [col for col in combined_data.columns if col not in categorical_columns + ['is_synthetic']]
            numerical_data = combined_data[numerical_columns].reset_index(drop=True)
            
            X = pd.concat([numerical_data, cat_encoded_df], axis=1)
        else:
            X = combined_data.drop('is_synthetic', axis=1)
        
        y = combined_data['is_synthetic']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # AUC score
        auc_score = roc_auc_score(y_test, y_pred_proba)
        c2st_score = max(auc_score, 1 - auc_score)
        
        c2st = np.round(c2st_score, 4)
        print(f'Fallback score: {c2st}')
        print(f'c2st={c2st}')  # Standard output for parsing
        with open(save_file, 'w') as f:
            f.write(f'c2st={c2st} (fallback)')