"""
Unified Reward Scoring Framework for Tabular DPO/GRAA.

Implements multiple reward signals across different categories:
1. Distributional: Classifier, Density (KDE/GMM)
2. Pointwise: DCR (Distance to Closest Record)
3. Marginal: Per-feature marginal likelihood
4. Relational: Correlation consistency
5. OOD: Isolation Forest, Autoencoder, Mahalanobis, One-Class SVM, LOF
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")


def _encode_dataframe(real_df, syn_df):
    """Encode mixed-type dataframe to numeric for scoring."""
    scaler = StandardScaler()
    encoders = {}
    real_enc = pd.DataFrame(index=real_df.index)
    syn_enc = pd.DataFrame(index=syn_df.index)

    for col in real_df.columns:
        if real_df[col].dtype == "object" or real_df[col].nunique() < 15:
            le = LabelEncoder()
            all_vals = pd.concat([real_df[col].astype(str), syn_df[col].astype(str)])
            le.fit(all_vals)
            real_enc[col] = le.transform(real_df[col].astype(str))
            syn_enc[col] = le.transform(syn_df[col].astype(str))
            encoders[col] = le
        else:
            real_enc[col] = pd.to_numeric(real_df[col], errors="coerce").fillna(0)
            syn_enc[col] = pd.to_numeric(syn_df[col], errors="coerce").fillna(0)

    real_arr = scaler.fit_transform(real_enc.values)
    syn_arr = scaler.transform(syn_enc.values)
    return real_arr, syn_arr, scaler, encoders


# ============================================================
# 1. DISTRIBUTIONAL: Classifier Reward
# ============================================================
def score_classifier(real_df, syn_df, n_estimators=100):
    """
    Train RF to distinguish real vs synthetic.
    Score = P(synthetic | row). Lower = more realistic = better.
    We invert: reward = 1 - P(synthetic) so higher = better.
    """
    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)

    X = np.vstack([real_arr, syn_arr])
    y = np.array([0] * len(real_arr) + [1] * len(syn_arr))

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    # Score synthetic rows: P(real | row)
    syn_proba = clf.predict_proba(syn_arr)
    p_real = syn_proba[:, 0] if clf.classes_[0] == 0 else syn_proba[:, 1]

    return p_real  # higher = more realistic


# ============================================================
# 2. DISTRIBUTIONAL: Density Estimation (GMM)
# ============================================================
def score_density(real_df, syn_df, n_components=10):
    """
    Fit GMM on real data, score synthetic rows by log-likelihood.
    Higher log-likelihood = row fits real distribution better.
    """
    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)

    gmm = GaussianMixture(n_components=min(n_components, len(real_arr) // 10),
                           random_state=42, covariance_type="full")
    gmm.fit(real_arr)

    scores = gmm.score_samples(syn_arr)
    # Normalize to [0, 1]
    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


# ============================================================
# 3. POINTWISE: DCR (Distance to Closest Record)
# ============================================================
def score_dcr(real_df, syn_df, batch_size=1000):
    """
    For each synthetic row, compute distance to closest real row.
    Higher distance = more private but potentially less realistic.
    We invert: reward = 1 - normalized_distance (closer to real = better quality).
    """
    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)

    min_dists = np.full(len(syn_arr), np.inf)
    for i in range(0, len(syn_arr), batch_size):
        batch = syn_arr[i : i + batch_size]
        dists = pairwise_distances(batch, real_arr, metric="euclidean")
        min_dists[i : i + batch_size] = dists.min(axis=1)

    # Invert and normalize: closer to real = higher reward
    if min_dists.max() > min_dists.min():
        scores = 1.0 - (min_dists - min_dists.min()) / (min_dists.max() - min_dists.min())
    else:
        scores = np.ones(len(syn_arr))
    return scores


# ============================================================
# 4. MARGINAL: Per-feature Marginal Likelihood
# ============================================================
def score_marginal(real_df, syn_df):
    """
    For each feature, fit KDE on real data.
    Score each synthetic row = average log-likelihood across features.
    Captures marginal quality but IGNORES dependencies.
    """
    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)
    n_features = real_arr.shape[1]

    row_scores = np.zeros(len(syn_arr))

    for j in range(n_features):
        real_col = real_arr[:, j]
        syn_col = syn_arr[:, j]

        # Skip constant columns
        if real_col.std() < 1e-8:
            continue

        try:
            kde = gaussian_kde(real_col, bw_method="scott")
            col_scores = kde(syn_col)
            # Log-transform and clip
            col_scores = np.log(col_scores + 1e-10)
            # Normalize per feature
            col_scores = col_scores - col_scores.min()
            if col_scores.max() > 0:
                col_scores = col_scores / col_scores.max()
            row_scores += col_scores
        except Exception:
            continue

    # Average across features
    row_scores /= max(n_features, 1)
    return row_scores


# ============================================================
# 5. RELATIONAL: Correlation Consistency
# ============================================================
def score_correlation(real_df, syn_df):
    """
    Learn real correlation matrix.
    For each synthetic row, check if feature pairs covary consistently.
    Score = how well the row's feature-pair products match real correlations.
    """
    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)
    n_features = real_arr.shape[1]

    # Real correlation matrix
    real_corr = np.corrcoef(real_arr, rowvar=False)
    real_mean = real_arr.mean(axis=0)
    real_std = real_arr.std(axis=0) + 1e-8

    # Z-score synthetic rows
    syn_z = (syn_arr - real_mean) / real_std

    # For each row, compute how well feature products match real correlations
    row_scores = np.zeros(len(syn_arr))

    # Sample feature pairs for efficiency
    np.random.seed(42)
    n_pairs = min(n_features * (n_features - 1) // 2, 100)
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairs.append((i, j))
    if len(pairs) > n_pairs:
        idx = np.random.choice(len(pairs), n_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    for i, j in pairs:
        expected_corr = real_corr[i, j]
        # Row's contribution: z_i * z_j should be close to expected correlation
        observed = syn_z[:, i] * syn_z[:, j]
        # Deviation from expected
        deviation = np.abs(observed - expected_corr)
        # Convert to score: lower deviation = higher score
        pair_score = np.exp(-deviation)
        row_scores += pair_score

    row_scores /= len(pairs)
    return row_scores


# ============================================================
# 6. OOD: Isolation Forest
# ============================================================
def score_isolation_forest(real_df, syn_df, n_estimators=100):
    """
    Fit Isolation Forest on real data.
    Score synthetic rows: higher = more in-distribution.
    """
    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)

    iso = IsolationForest(n_estimators=n_estimators, random_state=42, contamination=0.1)
    iso.fit(real_arr)

    # score_samples: higher = more normal (in-distribution)
    scores = iso.score_samples(syn_arr)
    # Normalize to [0, 1]
    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


# ============================================================
# 7. OOD: Autoencoder Reconstruction Error
# ============================================================
def score_autoencoder(real_df, syn_df, encoding_dim=8, epochs=50):
    """
    Train autoencoder on real data.
    Score = 1 - normalized reconstruction error.
    Low error = row is reconstructable from real patterns = in-distribution.
    """
    import torch
    import torch.nn as nn

    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)
    input_dim = real_arr.shape[1]
    hidden_dim = max(encoding_dim, input_dim // 2)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, encoding_dim), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train on real data
    real_t = torch.FloatTensor(real_arr).to(device)
    dataset = torch.utils.data.TensorDataset(real_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            pred = model(batch)
            loss = loss_fn(pred, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Score synthetic rows
    model.eval()
    with torch.no_grad():
        syn_t = torch.FloatTensor(syn_arr).to(device)
        recon = model(syn_t).cpu().numpy()
        errors = np.mean((syn_arr - recon) ** 2, axis=1)

    # Invert: lower error = higher score
    if errors.max() > errors.min():
        scores = 1.0 - (errors - errors.min()) / (errors.max() - errors.min())
    else:
        scores = np.ones(len(syn_arr))
    return scores


# ============================================================
# 8. OOD: Mahalanobis Distance
# ============================================================
def score_mahalanobis(real_df, syn_df):
    """
    Compute Mahalanobis distance of each synthetic row from real data centroid.
    Lower distance = more in-distribution = higher score.
    """
    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)

    mean = real_arr.mean(axis=0)
    cov = np.cov(real_arr, rowvar=False)
    # Regularize covariance
    cov += np.eye(cov.shape[0]) * 1e-6
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    dists = np.array([mahalanobis(row, mean, cov_inv) for row in syn_arr])

    # Invert and normalize
    if dists.max() > dists.min():
        scores = 1.0 - (dists - dists.min()) / (dists.max() - dists.min())
    else:
        scores = np.ones(len(syn_arr))
    return scores


# ============================================================
# 9. OOD: One-Class SVM
# ============================================================
def score_ocsvm(real_df, syn_df, max_train=5000):
    """
    Fit One-Class SVM on real data.
    Score synthetic rows by decision function (higher = more in-distribution).
    """
    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)

    # Subsample for speed
    if len(real_arr) > max_train:
        idx = np.random.RandomState(42).choice(len(real_arr), max_train, replace=False)
        train_arr = real_arr[idx]
    else:
        train_arr = real_arr

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    ocsvm.fit(train_arr)

    scores = ocsvm.decision_function(syn_arr)
    # Normalize to [0, 1]
    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


# ============================================================
# 10. OOD: Local Outlier Factor
# ============================================================
def score_lof(real_df, syn_df, n_neighbors=20, max_train=5000):
    """
    Fit LOF on real data, score synthetic rows.
    Higher score = more in-distribution.
    """
    real_arr, syn_arr, _, _ = _encode_dataframe(real_df, syn_df)

    # Subsample for speed
    if len(real_arr) > max_train:
        idx = np.random.RandomState(42).choice(len(real_arr), max_train, replace=False)
        train_arr = real_arr[idx]
    else:
        train_arr = real_arr

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(train_arr)

    scores = lof.decision_function(syn_arr)
    # Normalize to [0, 1]
    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


# ============================================================
# Master scoring function
# ============================================================
ALL_SCORERS = {
    "classifier": score_classifier,
    "density": score_density,
    "dcr": score_dcr,
    "marginal": score_marginal,
    "correlation": score_correlation,
    "isolation_forest": score_isolation_forest,
    "autoencoder": score_autoencoder,
    "mahalanobis": score_mahalanobis,
    "ocsvm": score_ocsvm,
    "lof": score_lof,
}

REWARD_CATEGORIES = {
    "classifier": "distributional",
    "density": "distributional",
    "dcr": "pointwise",
    "marginal": "marginal",
    "correlation": "relational",
    "isolation_forest": "ood",
    "autoencoder": "ood",
    "mahalanobis": "ood",
    "ocsvm": "ood",
    "lof": "ood",
}


def score_all(real_df, syn_df, scorers=None):
    """Run all (or selected) reward scorers and return results dict."""
    if scorers is None:
        scorers = list(ALL_SCORERS.keys())

    results = {}
    for name in scorers:
        if name not in ALL_SCORERS:
            print(f"  Unknown scorer: {name}")
            continue
        try:
            print(f"  Scoring: {name} ({REWARD_CATEGORIES[name]})...", end=" ", flush=True)
            scores = ALL_SCORERS[name](real_df, syn_df)
            results[name] = {
                "scores": scores,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "category": REWARD_CATEGORIES[name],
            }
            print(f"mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {"scores": None, "mean": None, "std": None, "error": str(e)}
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test all reward scorers")
    parser.add_argument("--real", required=True)
    parser.add_argument("--synthetic", required=True)
    args = parser.parse_args()

    real_df = pd.read_csv(args.real)
    syn_df = pd.read_csv(args.synthetic)

    # Align columns
    common = [c for c in real_df.columns if c in syn_df.columns]
    real_df = real_df[common]
    syn_df = syn_df[common]

    print(f"Real: {real_df.shape}, Synthetic: {syn_df.shape}")
    print()

    results = score_all(real_df, syn_df)

    print(f"\n{'='*60}")
    print(f"{'Reward':<20} {'Category':<15} {'Mean':>8} {'Std':>8}")
    print(f"{'='*60}")
    for name, r in sorted(results.items(), key=lambda x: x[1].get("category", "")):
        if r["mean"] is not None:
            print(f"{name:<20} {r['category']:<15} {r['mean']:>8.4f} {r['std']:>8.4f}")
