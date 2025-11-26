import numpy as np
import hdbscan
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def _reduce_and_normalize(vectors: np.ndarray) -> np.ndarray:
    """Drop the dimensionality and normalize, improving HDBSCAN stability."""
    n_samples, n_features = vectors.shape
    if n_samples >= 15 and n_features > 100:
        max_components = min(100, n_features, n_samples - 1)
        if max_components >= 10:
            pca = PCA(n_components=max_components, random_state=42)
            vectors = pca.fit_transform(vectors)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def perform_hdbscan(vectors):
    min_cluster_size = max(5, int(len(vectors) * 0.03))
    min_samples = max(2, int(min_cluster_size * 0.6))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    return clusterer.fit_predict(vectors)

def perform_kmeans(vectors, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    return kmeans.fit_predict(vectors)

def assign_noise_to_nearest(vectors, labels):
    # This is a placeholder for the logic we will implement
    # For now, just return labels as is
    return labels

def run_clustering_simulation():
    print("Generating synthetic data...")
    # Generate 3 dense clusters
    cluster1 = np.random.normal(loc=[10, 10], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[20, 20], scale=0.5, size=(50, 2))
    cluster3 = np.random.normal(loc=[30, 10], scale=0.5, size=(50, 2))
    
    # Generate scattered noise
    noise = np.random.uniform(low=0, high=100, size=(200, 2))
    
    all_vectors = np.vstack([cluster1, cluster2, cluster3, noise])
    # Shuffle to mix them up
    np.random.shuffle(all_vectors)
    
    print(f"Total points: {len(all_vectors)}")
    
    # Normalize (simulating the app's behavior)
    vectors_normalized = _reduce_and_normalize(all_vectors)
    
    print("Running HDBSCAN...")
    labels = perform_hdbscan(vectors_normalized)
    
    # FORCE HIGH NOISE FOR TESTING
    # Set the first 100 points to be noise (-1)
    labels[:100] = -1
    print("Manually injected noise for testing.")
    
    noise_indices = [i for i, l in enumerate(labels) if l == -1]
    noise_ratio = len(noise_indices) / len(all_vectors)
    print(f"Noise ratio: {noise_ratio:.2%}")
    
    if noise_ratio > 0.25:
        print(f"High noise detected ({noise_ratio:.1%}). Re-clustering noise...")
        noise_vectors = vectors_normalized[noise_indices]
        k = max(2, len(noise_vectors) // 20)
        print(f"Secondary clustering with k={k}")
        secondary_labels = perform_kmeans(noise_vectors, k=k)
        
        # Merge labels
        max_label = max(labels)
        for i, idx in enumerate(noise_indices):
            labels[idx] = secondary_labels[i] + max_label + 1
            
    # Check for remaining noise
    remaining_noise = [l for l in labels if l == -1]
    print(f"Remaining noise points: {len(remaining_noise)}")
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    print(f"Final cluster count: {len(unique_labels)}")

if __name__ == "__main__":
    run_clustering_simulation()
