import numpy as np
from spectral_lib import get_adj_matrix_A, get_L_and_D, get_clusters
from sklearn.cluster import SpectralClustering
from evaluate import main as eval_phi, parse_clustering_file, parse_graphfile
np.random.seed(1)

# data = np.loadtxt('./data/CA-HepTh.txt', dtype=int)

def get_clusters_all_algorithm(edges, n_clusters):
    A, vertices_mapping = get_adj_matrix_A(edges)

    # Algorithm 1
    # L, D = get_L_and_D(A, normalize_L=False)
    # labels = get_clusters(L, D, n_clusters, normalize_spectral=False)
    # with open('unnorm_spec_unnorm_L.txt', 'a') as f:
    #     for i in range(0, len(labels)):
    #         f.write(f'{vertices_mapping[i]} {labels[i]}\n')
        
    # Algorithm 2
    L, D = get_L_and_D(A, normalize_L=True)
    labels = get_clusters(L, D, n_clusters, normalize_spectral=True)
    with open('unnorm_spec_norm_L.txt', 'a') as f:
        for i in range(0, len(labels)):
            f.write(f'{vertices_mapping[i]} {labels[i]}\n')

    # Algorithm 3
    L, D = get_L_and_D(A, normalize_L=True)
    labels = get_clusters(L, D, n_clusters, normalize_spectral=True)
    with open('norm_spec_norm_L.txt', 'a') as f:
        for i in range(0, len(labels)):
            f.write(f'{vertices_mapping[i]} {labels[i]}\n')

# get_clusters_all_algorithm(data, 20)

foo = parse_graphfile('./data/CA-HepTh.txt')
bar = parse_clustering_file('./norm_spec_norm_L.txt')
foo = 9

# sc = SpectralClustering(2, affinity='precomputed', n_init=100)
# sc.fit(A)