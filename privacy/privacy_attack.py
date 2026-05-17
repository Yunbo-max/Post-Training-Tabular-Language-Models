from rdt import HyperTransformer
from rdt.transformers import OneHotEncoder, GaussianNormalizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import pickle
import argparse
import logging
import time
import os


def update_config(config):
    for col, sdtype in config["sdtypes"].items():
        if sdtype == "numerical":
            config["transformers"][col] = GaussianNormalizer(distribution="norm")
        if sdtype == "categorical":
            config["transformers"][col] = OneHotEncoder()
    return config


def preprocess_data(original_data, synthetic_data, hypertransformer=None):
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
    original_data_transformed = combined_data_transformed[: len(original_data)]
    synthetic_data_transformed = combined_data_transformed[len(original_data) :]

    return original_data_transformed, synthetic_data_transformed, ht


def prepare_attack_data(original_data, synthetic_data):
    original_data["member"] = 1
    synthetic_data["member"] = 0

    attack_data = pd.concat([original_data, synthetic_data])

    X = attack_data.drop("member", axis=1).values
    y = attack_data["member"].values
    return X, y


def plot_roc(clf, X_test, y_test):
    y_pred = clf.predict_proba(X_test)[:, 1]
    disp = RocCurveDisplay.from_estimator(clf, X_test, y_test, plot_chance_level=True)
    disp.line_.set_color("orange")
    plt.show()


def detection_attack(
    original_data,           # All original data
    synthetic_data,          # Synthetic data to evaluate
    random_state=42,
    test_size=0.3,           # Split original into train/holdout
    inference_mode=False,
    hypertransformer=None,
    show_roc_plot=False,
    clf_path="mia_model.pt",
    ht_path="ht.pt",
):
    """
    True MIA (Privacy Evaluation):
    - Split original data: train (member=1) vs holdout (member=0)
    - Train classifier on real_train vs real_holdout
    - Test on synthetic_data: high probability = privacy risk
    """
    columns = original_data.columns
    synthetic_data = synthetic_data[columns]

    # Data cleaning (keep your existing logic)
    for col in synthetic_data.columns:
        if synthetic_data[col].dtype == "object":
            synthetic_data[col] = synthetic_data[col].fillna("?")
            synthetic_data[col] = synthetic_data[col].apply(lambda x: x.strip())

    for col in original_data.columns:
        if original_data[col].dtype == "object":
            original_data[col] = original_data[col].fillna("?")
            original_data[col] = original_data[col].apply(lambda x: x.strip())

    if not inference_mode:
        # Split original data into train (members) and holdout (non-members)
        original_train, original_holdout = train_test_split(
            original_data, test_size=test_size, random_state=random_state
        )
        
        # Preprocess training data (real_train vs real_holdout)
        train_transformed, holdout_transformed, ht = preprocess_data(
            original_train, original_holdout, hypertransformer=None
        )

        # Train MIA classifier: train=1, holdout=0
        train_transformed["member"] = 1
        holdout_transformed["member"] = 0
        attack_data = pd.concat([train_transformed, holdout_transformed])
        
        X_train = attack_data.drop("member", axis=1).values
        y_train = attack_data["member"].values
        
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf.fit(X_train, y_train)

        # Save model
        with open(clf_path, "wb") as f:
            pickle.dump(clf, f)
        with open(ht_path, "wb") as f:
            pickle.dump(ht, f)
    else:
        # Load cached model
        assert clf_path is not None, "please specify path to load trained model."
        assert ht_path is not None, "please specify path to load hypertransformer."

        print("Loading cached MIA classifier and hypertransformer...")
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
        with open(ht_path, "rb") as f:
            ht = pickle.load(f)

    # Test on synthetic data (key MIA step)
    # Preprocess synthetic data with same transformer
    synthetic_transformed, _, _ = preprocess_data(
        synthetic_data, synthetic_data.iloc[:1], hypertransformer=ht
    )
    
    # Predict on synthetic data: high probability = looks like training member (privacy risk)
    y_pred_synthetic = clf.predict_proba(synthetic_transformed)[:, 1]
    
    # Calculate MIA score: average probability that synthetic samples look like training members
    mia_score = y_pred_synthetic.mean()
    print(f"\n{'='*50}")
    print(f"MIA Privacy Score: {mia_score:.4f} ({mia_score*100:.2f}%)")
    print(f"{'='*50}")
    print(f"Interpretation: Higher score → more synthetic data resembles training data")
    print(f"                → Higher privacy risk (memorization)")
    print(f"{'='*50}")

    attack_score = {"mia_score": mia_score, "mia_risk_percentage": mia_score * 100}

    if show_roc_plot:
        # For visualization
        plot_roc(clf, synthetic_transformed, y_pred_synthetic > 0.5)

    return attack_score, clf



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Privacy (Membership Inference) Attack"
    )
    parser.add_argument(
        "--synthetic", required=True, help="Path to synthetic data CSV file"
    )
    parser.add_argument(
        "--original", required=True, help="Path to original data CSV file"
    )
    parser.add_argument(
        "--cache",
        default="./cache",
        help="Directory to save/load classifier and vectorizer",
    )
    parser.add_argument(
        "--inference_mode",
        action="store_true",
        help="Run in inference using the cached model",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    start_time = time.time()

    logger.info(f"Loading synthetic data from {args.synthetic}")
    synthetic_data = pd.read_csv(args.synthetic)
    logger.info(f"Loading original data from {args.original}")
    original_data = pd.read_csv(args.original)

    os.makedirs(args.cache, exist_ok=True)
    clf_path = os.path.join(args.cache, "mia_rf_classifier.pkl")
    ht_path = os.path.join(args.cache, "mia_hypertransformer.pkl")

    inference_mode = args.inference_mode
    logger.info(inference_mode)
    attack_score, clf = detection_attack(
        original_data,
        synthetic_data,
        inference_mode=inference_mode,
        clf_path=clf_path,
        ht_path=ht_path,
    )

    elapsed = time.time() - start_time
    print(f"\nProcess completed in {elapsed:.2f} seconds.")
