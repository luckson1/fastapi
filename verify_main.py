import numpy as np
import sys
import os

# Add current directory to path so we can import main
sys.path.append(os.getcwd())

try:
    from main import perform_hdbscan, perform_kmeans, assign_noise_to_nearest, _reduce_and_normalize
except ImportError as e:
    print(f"Error importing from main: {e}")
    sys.exit(1)

def run_verification():
    print("Generating synthetic data...")
    # Generate 3 dense clusters
    cluster1 = np.random.normal(loc=[10, 10], scale=0.5, size=(50, 2))
    cluster2 = np.random.normal(loc=[20, 20], scale=0.5, size=(50, 2))
    cluster3 = np.random.normal(loc=[30, 10], scale=0.5, size=(50, 2))
    
    # Generate scattered noise
    noise = np.random.uniform(low=0, high=100, size=(200, 2))
    
    all_vectors = np.vstack([cluster1, cluster2, cluster3, noise])
    np.random.shuffle(all_vectors)
    
    print(f"Total points: {len(all_vectors)}")
    
    # Normalize
    vectors_normalized = _reduce_and_normalize(all_vectors)
    
    print("Testing perform_hdbscan...")
    labels = perform_hdbscan(vectors_normalized)
    print(f"HDBSCAN labels unique: {np.unique(labels)}")
    
    # Force some noise to test assign_noise_to_nearest
    labels[:10] = -1
    
    print("Testing assign_noise_to_nearest...")
    new_labels = assign_noise_to_nearest(vectors_normalized, labels.copy())
    
    if -1 in new_labels:
        print("FAILURE: assign_noise_to_nearest left some noise points.")
    else:
        print("SUCCESS: assign_noise_to_nearest assigned all noise points.")
        
    print("Testing perform_kmeans...")
    kmeans_labels = perform_kmeans(vectors_normalized, k=5)
    print(f"KMeans labels unique: {np.unique(kmeans_labels)}")
    
    if len(np.unique(kmeans_labels)) == 5:
        print("SUCCESS: perform_kmeans returned correct number of clusters.")
    else:
        print(f"FAILURE: perform_kmeans returned {len(np.unique(kmeans_labels))} clusters, expected 5.")

if __name__ == "__main__":
    run_verification()
