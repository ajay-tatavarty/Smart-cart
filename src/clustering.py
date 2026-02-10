"""Clustering analysis utilities."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


def encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Encode categorical features using one-hot encoding.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        
    Returns:
        DataFrame with encoded categorical features
    """
    df_encoded = df.copy()
    
    available_cols = [col for col in categorical_cols if col in df_encoded.columns]
    
    if available_cols:
        ohe = OneHotEncoder(sparse_output=False)
        enc_array = ohe.fit_transform(df_encoded[available_cols])
        
        enc_df = pd.DataFrame(
            enc_array,
            columns=ohe.get_feature_names_out(available_cols),
            index=df_encoded.index
        )
        
        df_encoded = pd.concat([df_encoded.drop(columns=available_cols), enc_df], axis=1)
    
    return df_encoded


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Scale features using StandardScaler.
    
    Args:
        X: Input features DataFrame
        
    Returns:
        Tuple of scaled features array and scaler object
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def apply_pca(X_scaled: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, PCA]:
    """Apply PCA for dimensionality reduction.
    
    Args:
        X_scaled: Scaled features array
        n_components: Number of PCA components
        
    Returns:
        Tuple of PCA-transformed data and PCA object
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def find_optimal_k(X: np.ndarray, k_range: range = None) -> int:
    """Find optimal K using elbow method.
    
    Args:
        X: Input features array
        k_range: Range of K values to test
        
    Returns:
        Optimal K value
    """
    if k_range is None:
        k_range = range(1, 11)
    
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    knee = KneeLocator(list(k_range), wcss, curve="convex", direction="decreasing")
    return knee.elbow if knee.elbow else 3


def calculate_silhouette_scores(X: np.ndarray, k_range: range = None) -> list:
    """Calculate silhouette scores for different K values.
    
    Args:
        X: Input features array
        k_range: Range of K values to test
        
    Returns:
        List of silhouette scores
    """
    if k_range is None:
        k_range = range(2, 11)
    
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    
    return scores


def kmeans_clustering(X: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    """Perform K-means clustering.
    
    Args:
        X: Input features array
        n_clusters: Number of clusters
        
    Returns:
        Cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels


def agglomerative_clustering(X: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    """Perform agglomerative (hierarchical) clustering.
    
    Args:
        X: Input features array
        n_clusters: Number of clusters
        
    Returns:
        Cluster labels
    """
    agg_clf = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = agg_clf.fit_predict(X)
    return labels


def analyze_clusters(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Analyze cluster characteristics.
    
    Args:
        X: Original features DataFrame
        labels: Cluster labels
        
    Returns:
        DataFrame with cluster summary statistics
    """
    X_with_clusters = X.copy()
    X_with_clusters["cluster"] = labels
    
    cluster_summary = X_with_clusters.groupby("cluster").mean()
    return cluster_summary
