import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn import metrics
np.random.seed(1)

# Get your mentioned graph
G = nx.karate_club_graph()

# Get ground-truth: club-labels -> transform to 0/1 np-array
#     (possible overcomplicated networkx usage here)
gt_dict = nx.get_node_attributes(G, 'club')
gt = [gt_dict[i] for i in G.nodes()]
gt = np.array([0 if i == 'Mr. Hi' else 1 for i in gt])

# Get adjacency-matrix as numpy-array
adj_mat = nx.to_numpy_matrix(G)

# print('ground truth')
# print(gt)

# Cluster
sc = SpectralClustering(2, affinity='precomputed', n_init=100)
sc.fit(adj_mat)


def get_L_matrix(A):
    shape = A.shape[0]
    D = np.zeros(shape=(shape, shape))
    for i in range(0, shape):
        D[i, i] = np.sum(A[i])

    L = D - A

    return L

L = get_L_matrix(adj_mat)
eig_values, eig_vectors = np.linalg.eig(L)
sorted_eig_values = np.sort(eig_values)
second = sorted_eig_values[1]
second_idx = eig_values.tolist().index(second)
values = eig_vectors[second_idx]
print(sc.labels_)
for v in eig_vectors:
    reduced = np.array(v[0].tolist())[0]
    temp = reduced > np.median(reduced)
    result = [1 if i else 0 for i in temp]
    print(result)






# Compare ground-truth and clustering-results
print('spectral clustering')
print(sc.labels_)
print('just for better-visualization: invert clusters (permutation)')
print(np.abs(sc.labels_ - 1))

# Calculate some clustering metrics
print(metrics.adjusted_rand_score(gt, sc.labels_))
print(metrics.adjusted_mutual_info_score(gt, sc.labels_))