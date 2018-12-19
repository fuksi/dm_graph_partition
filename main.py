import numpy as np
from spectral_lib import get_adj_matrix_A, get_L_and_D, get_clusters
np.random.seed(1)

data = np.loadtxt('./CA-HepTh.txt', dtype=int)

def normalized_spectral_with_normalized_L(edges):
    A, vertices_mapping = get_adj_matrix_A(edges)
    L, D = get_L_and_D(A, normalize_L=True)
    labels = get_clusters(L, D, 2, normalize_spectral=True)
    print(labels)

normalized_spectral_with_normalized_L(data)


# sc = SpectralClustering(2, affinity='precomputed', n_init=100)
# sc.fit(A)