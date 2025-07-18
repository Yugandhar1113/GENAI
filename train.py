import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

def load_data(path):
    """C:/Users/yugay/OneDrive/Desktop/Data Science/Sample - Superstore (1).xls"""
    return pd.read_excel("C:/Users/yugay/OneDrive/Desktop/Data Science/Sample - Superstore (1).xls")


def preprocess_data(df, features):
    """Select features, handle missing values, and scale the data."""
    X = df[features].copy()
    X = X.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.index


def plot_elbow(X_scaled):
    inertia = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 11), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    plt.savefig('elbow_method.png')
    with open('elbow_method.pkl', 'wb') as f:
        pickle.dump(kmeans,f)
    print("kmeans model saved as 'elbow_method.pkl'.")



def fit_kmeans(X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans


def evaluate_clustering(X_scaled, labels):
    print('/n--- Clustering Validation Metrics ---')
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    print(f'Silhouette Score: {sil:.3f}')
    print(f'Calinski-Harabasz Index: {ch:.3f}')
    print(f'Davies-Bouldin Index: {db:.3f}')


def visualize_clusters(df, features, cluster_col):
    sns.pairplot(df, vars=features, hue=cluster_col, palette='tab10')
    plt.suptitle('Cluster Visualization', y=1.02)
    plt.savefig('cluster_visualization.png')
    plt.show()
    
    with open('cluster_visualization.pkl', 'wb') as f:
        pickle.dump(KMeans,f)
    print("kmeans model saved as 'cluster_visualization.pkl'.")
    



def cluster_profiling(df, features, cluster_col):
    for c in sorted(df[cluster_col].unique()):
        print(f'/nCluster {c}:')
        print(df[df[cluster_col] == c][features].describe())


def main():
    data_path = 'C:/Users/yugay/OneDrive/Desktop/Data Science/Sample - Superstore (1).xls'  # Update path if needed
    features = ['Sales', 'Quantity', 'Discount', 'Profit']
    df = load_data(data_path)
    X_scaled, valid_idx = preprocess_data(df, features)
    print('Data loaded and preprocessed.')

    print('/n--- Elbow Method ---')
    plot_elbow(X_scaled)

    n_clusters = 4  # Set based on elbow plot
    labels, kmeans = fit_kmeans(X_scaled, n_clusters)
    df.loc[valid_idx, 'Cluster'] = labels

    evaluate_clustering(X_scaled, labels)

    print('/n--- Cluster Means ---')
    print(df.groupby('Cluster')[features].mean())

    print('/n--- Cluster Visualization ---')
    visualize_clusters(df.loc[valid_idx], features, 'Cluster')

    print('/n--- Cluster Profiling ---')
    cluster_profiling(df.loc[valid_idx], features, 'Cluster')


if __name__ == '__main__':
    main()