import numpy as np
from sklearn.cluster import SpectralClustering
import networkx as nx
import matplotlib.pyplot as plt

data = np.loadtxt('./CA-HepTh.txt', dtype=int)
data = data[0:2000]

## list of [fromNodeId to toNodeId]
def get_adj_matrix(edges):
    vertices = np.unique(edges.flatten()).tolist()
    shape = len(vertices)
    matrix = np.zeros(shape=(shape, shape))

    for fromId, toId in edges:
        fromIdx = vertices.index(fromId)
        toIdx = vertices.index(toId)
        matrix[fromIdx][toIdx] = 1

    degree_matrix = np.zeros(shape=(shape, shape))
    for i in range(0, shape):
        degree_matrix[i, i] = np.sum(matrix[i])

    laplacian_matrix = degree_matrix - matrix

    return matrix, degree_matrix, laplacian_matrix, vertices

# Nodes: 9877 Edges: 51971
# print(data)
A, D, L, vertices = get_adj_matrix(data)

# eig_values, eig_vectors = np.linalg.eig(L)

sc = SpectralClustering(2, affinity='precomputed', n_init=100)
sc.fit(A)
labels = sc.labels_
G = nx.from_numpy_matrix(A)
nx.draw(G,pos=nx.spring_layout(G))
plt.draw()
plt.show()

foo = 5