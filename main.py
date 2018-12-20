import numpy as np
from spectral_lib import get_adj_matrix_A, get_L_and_D, get_clusters, get_edges_and_param_line_from_file
np.random.seed(1)

# One method to wrap the whole process from source dataset to clustering output
def get_clusters_all_algorithm(dataset_name, n_clusters, algorithm):

    # Load and create adjajency matrix, param_line will be used for output file
    filepath = f'./data/{dataset_name}.txt'
    edges, param_line = get_edges_and_param_line_from_file(filepath)  
    A, vertices_mapping = get_adj_matrix_A(edges)

    # Algorithm 2
    L, D = get_L_and_D(A, normalize_L=True)
    labels = get_clusters(L, D, n_clusters, normalize_spectral=False)
    with open(F'./result/unnorm_{dataset_name}.output', 'w') as f:
        f.write(param_line)
        for i in range(0, len(labels)):
            f.write(f'{vertices_mapping[i]} {labels[i]}\n')

    # Algorithm 3
    L, D = get_L_and_D(A, normalize_L=True)
    labels = get_clusters(L, D, n_clusters, normalize_spectral=True)
    with open(F'./result/{dataset_name}.output', 'w') as f:
        f.write(param_line)
        for i in range(0, len(labels)):
            f.write(f'{vertices_mapping[i]} {labels[i]}\n')

datasets_k = {
    'ca-AstroPh': 50,
    'ca-CondMat': 100, 
    'ca-GrQc': 2, 
    'ca-HepPh': 25, 
    'ca-HepTh': 20
}

for dataset_name, k in datasets:
    get_clusters_all_algorithm(dataset_name, k)
