import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def train_and_optimize_DBSCAN(data, feature_cols, eps_min=0.1, eps_max=1.0, eps_step=0.1, min_samples_min=2, min_samples_max=10, min_samples_step=1):
    # Preprocess data using standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Initialize variables to store optimal hyperparameters and metrics
    best_eps = 0
    best_min_samples = 0
    best_silhouette_score = -1

    # Perform grid search to find optimal hyperparameters
    for eps in np.arange(eps_min, eps_max + eps_step, eps_step):
        for min_samples in range(min_samples_min, min_samples_max + min_samples_step, min_samples_step):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = dbscan.fit_predict(X_scaled)
            if len(set(y_pred)) > 1:
                # Compute Silhouette Score for non-trivial solutions
                silhouette = silhouette_score(X_scaled, y_pred)
                if silhouette > best_silhouette_score:
                    # Update best hyperparameters and Silhouette Score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_silhouette_score = silhouette

    # Train DBSCAN model using best hyperparameters
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    y_pred = dbscan.fit_predict(X_scaled)

    # Return predicted cluster labels and best hyperparameters
    return y_pred, {'eps': best_eps, 'min_samples': best_min_samples}