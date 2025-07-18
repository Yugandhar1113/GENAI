import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_data(path):
    """C:/Users/yugay/OneDrive/Desktop/Data Science/Sample - Superstore (1).xls"""
    return pd.read_excel('C:/Users/yugay/OneDrive/Desktop/Data Science/Sample - Superstore (1).xls')


def preprocess_data(df, features):
    """Select features, handle missing values, and scale the data."""
    X = df[features].copy()
    X = X.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.index


def find_best_k(X_scaled, k_range=(2, 11)):
    """Try different k values and return the one with the highest silhouette score."""
    sil_scores = []
    best_score = -1
    best_k = 2
    for k in range(*k_range):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)
        if sil > best_score:
            best_score = sil
            best_k = k

    # Plot silhouette scores
    plt.figure(figsize=(8, 4))
    plt.plot(range(*k_range), sil_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores vs Number of Clusters')
    plt.savefig('silhouette_scores.png')
    plt.show()

    print(f"/n✅ Best number of clusters based on silhouette score: {best_k} (Score: {best_score:.3f})")
    return best_k


def fit_kmeans(X_scaled, n_clusters):
    """Fit KMeans to data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans


def evaluate_clustering(X_scaled, labels):
    """Print clustering evaluation metrics."""
    print('/n--- Clustering Validation Metrics ---')
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    print(f'Silhouette Score: {sil:.3f}')
    print(f'Calinski-Harabasz Index: {ch:.3f}')
    print(f'Davies-Bouldin Index: {db:.3f}')


def visualize_clusters(df, features, cluster_col):
    """Pairplot visualization of clusters."""
    sns.pairplot(df, vars=features, hue=cluster_col, palette='tab10')
    plt.suptitle('Cluster Visualization', y=1.02)
    plt.savefig('cluster_visualization.png')
    plt.show()


def cluster_profiling(df, features, cluster_col):
    """Show summary statistics for each cluster."""
    for c in sorted(df[cluster_col].unique()):
        print(f'/nCluster {c}:')
        print(df[df[cluster_col] == c][features].describe())


def main():
    data_path = r"C:/Users/yugay/OneDrive/Desktop/Data Science/Sample - Superstore (1).xls"
    df = load_data(data_path)

    # Date conversions
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Ship_Duration'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Derived features
    df['Profit/Sales'] = df.apply(lambda row: row['Profit'] / row['Sales'] if row['Sales'] != 0 else np.nan, axis=1)
    df['Quantity/Profit'] = df.apply(lambda row: row['Quantity'] / row['Profit'] if row['Profit'] != 0 else np.nan, axis=1)
    df['Profit_Margin'] = df['Profit'] / df['Sales']
    df['Efficiency'] = df.apply(lambda row: row['Quantity'] / row['Profit'] if row['Profit'] != 0 else np.nan, axis=1)

    features = ['Sales', 'Quantity', 'Discount', 'Profit', 'Profit_Margin', 'Efficiency', 'Ship_Duration']
    X_scaled, valid_idx = preprocess_data(df, features)
    print('✅ Data loaded and preprocessed.')

    print('/n--- Finding Best k with Silhouette Score ---')
    best_k = find_best_k(X_scaled)

    labels, kmeans = fit_kmeans(X_scaled, best_k)
    df.loc[valid_idx, 'Cluster'] = labels

    evaluate_clustering(X_scaled, labels)

    print('/n--- Cluster Means ---')
    print(df.groupby('Cluster')[features].mean().round(2))

    print('/n--- Cluster Visualization ---')
    visualize_clusters(df.loc[valid_idx], features, 'Cluster')

    print('/n--- Cluster Profiling ---')
    cluster_profiling(df.loc[valid_idx], features, 'Cluster')


if __name__ == '__main__':
    main()