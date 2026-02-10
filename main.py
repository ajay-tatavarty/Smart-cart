"""
Smart Cart - Customer Segmentation Analysis
Main analysis script that orchestrates the complete workflow.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import data_preprocessing as dp
from src import feature_engineering as fe
from src import clustering as clust
from src import visualization as viz


def main():
    """Main analysis pipeline."""
    
    print("="*60)
    print("Smart Cart - Customer Segmentation Analysis")
    print("="*60)
    
    # Step 1: Load Data
    print("\n[1/8] Loading data...")
    data_path = "data/smartcart_customers.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please place your 'smartcart_customers.csv' in the 'data' folder.")
        return
    
    df = dp.load_data(data_path)
    print(f"✓ Loaded {df.shape[0]} records with {df.shape[1]} features")
    print(f"Data shape: {df.shape}")
    
    # Step 2: Data Preprocessing
    print("\n[2/8] Preprocessing data...")
    df = dp.handle_missing_values(df)
    print(f"✓ Handled missing values")
    
    # Step 3: Feature Engineering
    print("\n[3/8] Engineering features...")
    df = fe.engineer_features(df)
    print(f"✓ Created engineered features (Age, Tenure, Total_Spending, etc.)")
    
    # Step 4: Data Cleaning
    print("\n[4/8] Removing outliers...")
    original_size = len(df)
    df = dp.remove_outliers(df, age_limit=90, income_limit=600000)
    removed = original_size - len(df)
    print(f"✓ Removed {removed} outliers ({100*removed/original_size:.1f}%)")
    
    # Step 5: Drop unnecessary columns
    print("\n[5/8] Preparing features for clustering...")
    cols_to_drop = [
        "ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", 
        "Dt_Customer", "MntWines", "MntFruits", "MntMeatProducts", 
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    df_cleaned = dp.drop_columns(df, cols_to_drop)
    
    # Step 6: Encode categorical features
    print("\n[6/8] Encoding categorical features...")
    df_encoded = clust.encode_categorical_features(df_cleaned, ["Education", "Living_With"])
    print(f"✓ Encoded categorical features. Shape: {df_encoded.shape}")
    
    # Step 7: Scale features
    print("\n[7/8] Scaling features...")
    X_scaled, scaler = clust.scale_features(df_encoded)
    print(f"✓ Scaled features using StandardScaler")
    
    # Step 8: Apply PCA
    print("\n[8/8] Applying PCA for visualization...")
    X_pca, pca = clust.apply_pca(X_scaled, n_components=3)
    print(f"✓ Applied PCA. Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Clustering Analysis
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)
    
    # Find optimal K
    print("\nFinding optimal number of clusters...")
    k_range = range(1, 11)
    wcss = []
    for k in k_range:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        wcss.append(kmeans.inertia_)
    
    optimal_k = clust.find_optimal_k(X_pca)
    print(f"✓ Optimal K (Elbow method): {optimal_k}")
    
    # Calculate silhouette scores
    silhouette_scores = clust.calculate_silhouette_scores(X_pca)
    best_k_silhouette = np.argmax(silhouette_scores) + 2
    print(f"✓ Best K (Silhouette): {best_k_silhouette} (score: {max(silhouette_scores):.3f})")
    
    # Use K=4 as suggested by optimal_k
    n_clusters = optimal_k if optimal_k else 4
    print(f"\nUsing K={n_clusters} for final clustering")
    
    # K-means Clustering
    print("\nPerforming K-means clustering...")
    labels_kmeans = clust.kmeans_clustering(X_pca, n_clusters=n_clusters)
    print(f"✓ K-means complete")
    
    # Agglomerative Clustering
    print("Performing hierarchical clustering...")
    labels_agg = clust.agglomerative_clustering(X_pca, n_clusters=n_clusters)
    print(f"✓ Hierarchical clustering complete")
    
    # Cluster Analysis
    print("\nAnalyzing cluster characteristics...")
    cluster_summary = clust.analyze_clusters(df_encoded, labels_agg)
    print("\nCluster Summary Statistics:")
    print(cluster_summary)
    
    # Visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 3D PCA visualization
    print("\n1. Creating 3D PCA visualization...")
    fig = viz.plot_3d_scatter(X_pca, labels_agg, "3D Clustering Visualization")
    viz.save_figure(f"{results_dir}/01_3d_clustering.png")
    plt.close()
    
    # Elbow curve
    print("2. Creating elbow curve...")
    viz.plot_elbow_curve(k_range, wcss)
    viz.save_figure(f"{results_dir}/02_elbow_curve.png")
    plt.close()
    
    # Silhouette scores
    print("3. Creating silhouette score plot...")
    viz.plot_silhouette_scores(range(2, 11), silhouette_scores)
    viz.save_figure(f"{results_dir}/03_silhouette_scores.png")
    plt.close()
    
    # Cluster distribution
    print("4. Creating cluster distribution...")
    viz.plot_cluster_distribution(labels_agg)
    viz.save_figure(f"{results_dir}/04_cluster_distribution.png")
    plt.close()
    
    # Spending vs Income
    df_with_labels = df_encoded.copy()
    df_with_labels["cluster"] = labels_agg
    print("5. Creating spending vs income plot...")
    viz.plot_spending_vs_income(df_with_labels, labels_agg)
    viz.save_figure(f"{results_dir}/05_spending_vs_income.png")
    plt.close()
    
    # Correlation heatmap
    print("6. Creating correlation heatmap...")
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
    viz.plot_correlation_heatmap(df_encoded[numeric_cols])
    viz.save_figure(f"{results_dir}/06_correlation_heatmap.png")
    plt.close()
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nVisualizations saved to: {results_dir}/")
    print("\nFiles generated:")
    print("  - 01_3d_clustering.png")
    print("  - 02_elbow_curve.png")
    print("  - 03_silhouette_scores.png")
    print("  - 04_cluster_distribution.png")
    print("  - 05_spending_vs_income.png")
    print("  - 06_correlation_heatmap.png")
    
    return df_encoded, labels_agg, X_pca


if __name__ == "__main__":
    main()
