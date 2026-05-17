import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
import logging
import time

import matplotlib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# import torch
# from flair.data import Sentence
# from flair.models import SequenceTagger

# def load_flair_model(model_name='flair/ner-english-large'):
#     model_path = f'./cache/{model_name.replace("/", "_")}_flair_model.pt'
#     if os.path.exists(model_path):
#         tagger = SequenceTagger.load(model_path)
#     else:
#         tagger = SequenceTagger.load(model_name)
#         tagger.eval()
#         tagger = tagger.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#         tagger.save(model_path)
#     return tagger

def prepare_features(synthetic_data, original_data, metadata, vectorizer=None, fit_vectorizer=False):
    text_columns = [col for col, info in metadata['columns'].items()
                   if info['sdtype'] == 'text' and col in synthetic_data.columns]
    if text_columns:
        print("Vectorizing text columns:", text_columns)
        syn_text = synthetic_data[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        orig_text = original_data[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        if fit_vectorizer or vectorizer is None:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            print("Fitting TF-IDF vectorizer...")
            vectorizer.fit(pd.concat([syn_text, orig_text]))
        print("Transforming synthetic text...")
        syn_text_features = vectorizer.transform(syn_text)
        print("Transforming original text...")
        orig_text_features = vectorizer.transform(orig_text)
        non_text_cols = [col for col in synthetic_data.columns if col not in text_columns]
        if non_text_cols:
            print("Encoding non-text columns:", non_text_cols)
            combined = pd.concat([synthetic_data[non_text_cols], original_data[non_text_cols]], axis=0)
            combined_encoded = pd.get_dummies(combined).fillna(0)
            # Convert boolean columns to integers (0/1)
            for col in combined_encoded.columns:
                if combined_encoded[col].dtype == bool:
                    combined_encoded[col] = combined_encoded[col].astype(int)
            combined_encoded = combined_encoded.apply(pd.to_numeric)

            combined_encoded = combined_encoded.apply(pd.to_numeric) 
            syn_non_text = combined_encoded.iloc[:len(synthetic_data), :]
            orig_non_text = combined_encoded.iloc[len(synthetic_data):, :]
            synthetic_features = np.hstack([syn_non_text.values, syn_text_features.toarray()])
            original_features = np.hstack([orig_non_text.values, orig_text_features.toarray()])
        else:
            synthetic_features = syn_text_features.toarray()
            original_features = orig_text_features.toarray()
    else:
        print("No text columns found, encoding all columns.")
        combined = pd.concat([synthetic_data, original_data], axis=0)
        combined_encoded = pd.get_dummies(combined).fillna(0)
        # Convert boolean columns to integers (0/1)
        for col in combined_encoded.columns:
            if combined_encoded[col].dtype == bool:
                combined_encoded[col] = combined_encoded[col].astype(int)
        combined_encoded = combined_encoded.apply(pd.to_numeric)


        combined_encoded = combined_encoded.apply(pd.to_numeric)        
        synthetic_features = combined_encoded.iloc[:len(synthetic_data), :].values
        original_features = combined_encoded.iloc[len(synthetic_data):, :].values
    return synthetic_features, original_features, vectorizer



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Standalone Membership Inference Attack")
    parser.add_argument('--synthetic', required=True, help='Path to synthetic data CSV file')
    parser.add_argument('--original', required=True, help='Path to original data CSV file')
    parser.add_argument('--metadata', required=True, help='Path to metadata JSON file')
    parser.add_argument('--cache', default='./cache', help='Directory to save/load classifier and vectorizer')
    parser.add_argument('--retrain', action='store_true', help='Force retrain and overwrite cached model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    start_time = time.time()

    logger.info(f"Loading synthetic data from {args.synthetic}")
    synthetic_data = pd.read_csv(args.synthetic)
    logger.info(f"Loading original data from {args.original}")
    original_data = pd.read_csv(args.original)
    logger.info(f"Loading metadata from {args.metadata}")
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)

    os.makedirs(args.cache, exist_ok=True)
    clf_path = os.path.join(args.cache, 'mia_rf_classifier.pkl')
    vec_path = os.path.join(args.cache, 'mia_tfidf_vectorizer.pkl')


   

   

    # logger.info("Loading Flair NER model (for compatibility)...")
    # flair_model = load_flair_model()

    # Retrain if --retrain is set, or if cache files do not exist
    if args.retrain or not (os.path.exists(clf_path) and os.path.exists(vec_path)):
        logger.info("Training classifier and saving to cache...")
        syn_features, orig_features, vectorizer = prepare_features(synthetic_data, original_data, metadata, vectorizer=None, fit_vectorizer=True)
        X = np.vstack([syn_features, orig_features])
        y = np.concatenate([np.ones(len(syn_features)), np.zeros(len(orig_features))])

        split = train_test_split(X, y, test_size=0.2, random_state=42)
        x_train, x_test, y_train, y_test = split

        print("Training RandomForestClassifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(x_train, y_train)
        with open(clf_path, 'wb') as f:
            pickle.dump(clf, f)
        with open(vec_path, 'wb') as f:
            pickle.dump(vectorizer, f)
    else:
        logger.info("Loading cached classifier and vectorizer...")
        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)
        with open(vec_path, 'rb') as f:
            vectorizer = pickle.load(f)
        syn_features, orig_features, _ = prepare_features(synthetic_data, original_data, metadata, vectorizer, fit_vectorizer=False)

    print("Predicting membership probabilities...")
    # X_test = np.vstack([syn_features, orig_features])
    y_pred = clf.predict_proba(x_test)[:, 1]
    # After X_test = np.vstack([syn_features, orig_features])

    auc = roc_auc_score(y_test, y_pred)
    print('mia-score', auc)

    elapsed = time.time() - start_time
    print(f"\nProcess completed in {elapsed:.2f} seconds.")
    print("\nMembership Inference Probabilities (y_pred):")


    print('y_test',y_test.tolist())
    print('y_pred',y_pred.tolist())


    

    # # Calculate abs distance from 0.5 for each sample
    # distances = np.abs(y_pred - 0.5)
    # sorted_indices = np.argsort(distances)  # Smallest to largest

    # print("\nSamples ranked by privacy risk (closest to 0.5 are best):")
    # for rank, idx in enumerate(sorted_indices):
    #     print(f"Rank {rank+1}: Sample {idx}, Probability {y_pred[idx]:.4f}, Distance from 0.5: {distances[idx]:.4f}")

    # unique_values = np.unique(y_pred)
    # print(f"Number of unique y_pred values: {len(unique_values)}")
    # print(f"Unique y_pred values: {unique_values}")



    # unique_values, counts = np.unique(y_pred, return_counts=True)
    # print(f"Number of unique y_pred values: {len(unique_values)}")
    # print(f"Unique y_pred values: {unique_values}")
    # print(f"Frequencies: {counts}")

    # import matplotlib
    # matplotlib.use('Agg')  # Use non-interactive backend if needed
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 4))
    # plt.scatter(unique_values, counts, color='blue', alpha=0.7)
    # plt.xlabel('Unique Predicted Probability (Synthetic)')
    # plt.ylabel('Frequency')
    # plt.title('Frequency of Unique Membership Inference Probabilities')
    # plt.tight_layout()
    # plt.savefig("mia_unique_scatter.png")
    # print("Scatter plot saved as mia_unique_scatter.png")



# ...existing code...


    # ...existing code for ranking...
if __name__ == "__main__":
    main()