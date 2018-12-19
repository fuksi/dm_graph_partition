import numpy as np
import networkx as nx
from sklearn.cluster import k_means
from scipy.sparse.linalg import eigsh

# adj_mat = np.loadtxt('example.txt')

def get_adj_matrix_A(edges):
    ''' Convert an input graph of edges to adjajency matrix
    
    Arguments:
        edges {list of tuple/list} -- Collection of [fromVertexId, toVertexId]
    
    Returns:
        matrix -- adjajency matrix representation of graph
        vertices -- mapping from original vertex ID to vertex index of the adjajency matrix 
    '''
    vertices = np.unique(edges.flatten()).tolist()
    shape = len(vertices)
    matrix = np.zeros(shape=(shape, shape))

    for fromId, toId in edges:
        fromIdx = vertices.index(fromId)
        toIdx = vertices.index(toId)
        matrix[fromIdx][toIdx] = 1

    return matrix, vertices

def get_L_and_D(graph, normalize_L=True):
    ''' Find laplacian matrix and degree matrix for a matrix
    
    Arguments:
        graph {n x n matrix} -- diagonally symmetric 

    Returns:
        L: laplacian matrix n x n
        D: degree matrix n x n
    '''

    # Better to clone, but got out of memory error
    # A = np.array(graph)
    A = graph


    D = A.sum(axis=0)
    shape = A.shape[0]

    if normalize_L:
        I = np.identity(shape)

        # necessary since graph is not 100% connected
        diagonal_mask = (D == 1)
        D_sqrt = np.where(diagonal_mask, 1, np.sqrt(D))

        # calc normalized L
        L = A / D_sqrt / D_sqrt[:, np.newaxis]
        L *= -1

        # unmask diag
        np.fill_diagonal(L, 1)
    else:
        # unnormalized L
        L = D - A 

    return L, D 

def get_clusters(L, D, n_clusters, normalize_spectral=True):
    ''' Group nodes into a number of cluster using spetral graph method
    
    Arguments:
        L {m square matrix} --[Laplacian matrix of the adjajency matrix
        n_clusters {n} -- Number of cluster
    '''

    # Here we'll need to find up to nth smallest eigenvalues
    # However, the eigsh/eigs method is really slow for this objective
    # so we'll use shift-inverted mode instead
    # https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    L_inverse = L * -1
    values, vectors = eigsh(L_inverse, n_clusters, sigma=1, which='LM')

    # Each eigenvectors vectors[i] correlates to eigenvalues[i]
    # We'll take eigenvectors up to nth eigenvalues (inclusive)
    # Notes: eigenvalues output of eigsh are in descending order-> need to reverse order
    embedding = vectors.T[::-1]

    # If normalized, normalize by degree so that rows have norm 1
    if normalize_spectral:
        embedding = embedding / D

    # Transpose back to get m x p matrix where m is number of nodes, p is number of feature
    embedding = embedding.T

    # Embedding now has shape: m x p, where m is number of nodes, p is number of feature
    # Simply use k_means from here
    _, labels, _ = k_means(embedding, n_clusters)

    return labels

# def normalized_spectral_with_normalized_L(agraph):
#     L, D = get_L_and_D(adj_mat, normalize_L=True)
#     labels = get_clusters(L, D, 2, normalize_spectral=True)
#     print(labels)
#     foo = 8

# normalized_spectral_with_normalized_L(adj_mat)





    
    

