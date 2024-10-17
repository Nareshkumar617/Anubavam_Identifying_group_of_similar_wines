import numpy as np
import pandas as pd

class Matrix:
    def __init__(self, filename=None):
        self.array_2d = None
        if filename is not None:
            self.load_from_csv(filename)

    def load_from_csv(self, filename):
        self.array_2d = pd.read_csv(filename).to_numpy()
        print("Data Loaded Successfully")

    def standardise(self):
        mean = np.mean(self.array_2d, axis=0)
        std_dev = np.std(self.array_2d, axis=0)
        self.array_2d = (self.array_2d - mean) / std_dev
        print("Data Standardized")

    def get_distance(self, other_matrix, row_i):
        #Euclidean distance between row_i of this matrix and all rows of other_matrix
        distances = np.linalg.norm(self.array_2d[row_i] - other_matrix.array_2d, axis=1)
        return distances.reshape(-1, 1)

    def get_weighted_distance(self, other_matrix, weights, row_i):
        #Weighted Euclidean distance between row_i and all rows of other_matrix using weight
        diff = (self.array_2d[row_i] - other_matrix.array_2d) * weights
        distances = np.linalg.norm(diff, axis=1)
        return distances.reshape(-1, 1)

    def get_count_frequency(self):
        if self.array_2d.shape[1] == 1:
            unique, counts = np.unique(self.array_2d, return_counts=True)
            return dict(zip(unique.flatten(), counts))
        return 0

def get_initial_weights(m):
    #Return a row matrix with random values adding to 1
    weights = np.random.rand(m)
    return weights / np.sum(weights)

def get_centroids(data_matrix, S, K):
    #Calculate centroids based on the groups formed in S.
    centroids = []
    for k in range(K):
        cluster_data = data_matrix.array_2d[S.flatten() == k]
        centroid = np.mean(cluster_data, axis=0)
        centroids.append(centroid)
    return np.array(centroids)

def get_separation_within(data_matrix, centroids, S, K):
    separation = np.zeros(data_matrix.array_2d.shape[1])
    for k in range(K):
        cluster_data = data_matrix.array_2d[S.flatten() == k]
        cluster_distances = np.linalg.norm(cluster_data - centroids[k], axis=1)
        separation += np.sum(cluster_distances)
    return separation.reshape(1, -1)

def get_separation_between(data_matrix, centroids, S, K):
    overall_centroid = np.mean(centroids, axis=0)
    separation = np.zeros(data_matrix.array_2d.shape[1])
    for k in range(K):
        separation += np.linalg.norm(centroids[k] - overall_centroid)
    return separation.reshape(1, -1)

def get_groups(data_matrix, K):
    #Assign data points to clusters based on the distances to centroids.
    # Initialize centroids randomly
    initial_centroids_indices = np.random.choice(data_matrix.array_2d.shape[0], K, replace=False)
    centroids = data_matrix.array_2d[initial_centroids_indices]

    # Perform K-means clustering
    for _ in range(100):  # Number of iterations
        # Calculate distances from each point to each centroid
        distances = np.zeros((data_matrix.array_2d.shape[0], K))

        for k in range(K):
            centroid_matrix = Matrix()
            centroid_matrix.array_2d = centroids[k].reshape(1, -1)  # Create a matrix instance with one row
            distances[:, k] = data_matrix.get_distance(centroid_matrix, slice(None)).flatten()

        # Assign clusters based on closest centroid
        S = np.argmin(distances, axis=1)
        centroids = get_centroids(data_matrix, S.reshape(-1, 1), K)

    return S

def get_new_weights(data_matrix, centroids, old_weights, S, K):
    ##Update weights based on the clustering.
    new_weights = np.zeros_like(old_weights)
    for k in range(K):
        cluster_data = data_matrix.array_2d[S.flatten() == k]
        cluster_distances = np.linalg.norm(cluster_data - centroids[k], axis=1)
        new_weights[k] = np.mean(cluster_distances)
    return new_weights.reshape(1, -1)

def run_test():
    m = Matrix('/Count_AI_task/Data_anubavam.csv')
    for k in range(2, 11):
        for i in range(20):
            S = get_groups(m, k)
            print(str(k), str(S))

run_test()

