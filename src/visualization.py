"""Visualization utilities for clustering analysis."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd


def plot_pair_plot(df: pd.DataFrame, columns: list = None, title: str = "Pair Plot"):
    """Create a pair plot of selected columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to plot
        title: Plot title
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
    
    plt.figure(figsize=(10, 8))
    sns.pairplot(df[columns])
    plt.suptitle(title, y=1.01)
    return plt


def plot_correlation_heatmap(df: pd.DataFrame, figsize: tuple = (8, 6), title: str = "Correlation Matrix"):
    """Create a correlation heatmap.
    
    Args:
        df: Input DataFrame
        figsize: Figure size
        title: Plot title
    """
    corr = df.corr(numeric_only=True)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        annot_kws={"size": 8},
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title(title)
    plt.tight_layout()
    return plt


def plot_3d_scatter(X_pca: np.ndarray, labels: np.ndarray = None, title: str = "3D PCA Projection"):
    """Create a 3D scatter plot of PCA components.
    
    Args:
        X_pca: PCA-transformed data
        labels: Cluster labels for coloring
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    if labels is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap="viridis")
        plt.colorbar(scatter, label="Cluster")
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
    
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.set_title(title)
    
    return fig


def plot_elbow_curve(k_range: range, wcss: list, figsize: tuple = (8, 6), title: str = "Elbow Curve"):
    """Plot elbow curve for optimal K selection.
    
    Args:
        k_range: Range of K values
        wcss: Within-cluster sum of squares values
        figsize: Figure size
        title: Plot title
    """
    plt.figure(figsize=figsize)
    plt.plot(list(k_range), wcss, marker='o', linewidth=2, markersize=6)
    plt.xlabel("K (Number of Clusters)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_silhouette_scores(k_range: range, scores: list, figsize: tuple = (8, 6), title: str = "Silhouette Scores"):
    """Plot silhouette scores for different K values.
    
    Args:
        k_range: Range of K values
        scores: Silhouette scores
        figsize: Figure size
        title: Plot title
    """
    plt.figure(figsize=figsize)
    plt.plot(list(k_range), scores, marker='o', linewidth=2, markersize=6, color='green')
    plt.xlabel("K (Number of Clusters)")
    plt.ylabel("Silhouette Score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_cluster_distribution(labels: np.ndarray, figsize: tuple = (8, 6), title: str = "Cluster Distribution"):
    """Plot distribution of samples across clusters.
    
    Args:
        labels: Cluster labels
        figsize: Figure size
        title: Plot title
    """
    plt.figure(figsize=figsize)
    unique_labels = np.unique(labels)
    cluster_counts = [np.sum(labels == label) for label in unique_labels]
    
    sns.countplot(x=labels, palette="husl")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.tight_layout()
    return plt


def plot_spending_vs_income(df: pd.DataFrame, labels: np.ndarray = None, figsize: tuple = (10, 6), 
                            title: str = "Total Spending vs Income"):
    """Plot spending vs income scatter plot.
    
    Args:
        df: Input DataFrame with spending and income data
        labels: Cluster labels for coloring
        figsize: Figure size
        title: Plot title
    """
    plt.figure(figsize=figsize)
    
    if labels is not None:
        scatter = plt.scatter(df["Total_Spending"], df["Income"], c=labels, cmap="viridis", alpha=0.6, s=50)
        plt.colorbar(scatter, label="Cluster")
    else:
        plt.scatter(df["Total_Spending"], df["Income"], alpha=0.6, s=50)
    
    plt.xlabel("Total Spending")
    plt.ylabel("Income")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def save_figure(filepath: str):
    """Save current figure to file.
    
    Args:
        filepath: Path to save the figure
    """
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")
