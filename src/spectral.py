import numpy as np
from numerics import euclidean_distance, diag_rev_sqrt
from kmeans import kmeanspp, kmeans, nkmeans, nkmeanspp

import matplotlib.pyplot as plt


def spectral_clustering(X: np.ndarray, k: int, sigma=1, dist_f=euclidean_distance, n=20, m=2, max_iter=1000):
    A = dist_f(X[:, None], X, axis=2)
    A = np.exp(-A**2 / (2 * sigma**2))
    np.fill_diagonal(A, 0)
    
    D = np.diag(np.sum(A, axis=1))
    D_rev_sqrt = diag_rev_sqrt(D)
    
    L = D_rev_sqrt @ A @ D_rev_sqrt
    
    eigvals, eigvecs = np.linalg.eig(L)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    eigvecs = eigvecs[:, :k]
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
    
    return nkmeanspp(eigvecs, k, n=n, max_iter=max_iter)


if __name__ == "__main__":
    n = 100
    X1 = np.random.rand(n, 2)
    X2 = np.random.rand(n, 2) + 10
    X3 = np.random.rand(n, 2) + (0, 10)
    X4 = np.random.rand(n, 2) + (10, 0)
    # plt.scatter(X1[:, 0], X1[:, 1], c='r')
    # plt.scatter(X2[:, 0], X2[:, 1], c='b')
    # plt.scatter(X3[:, 0], X3[:, 1], c='g')
    # plt.scatter(X4[:, 0], X4[:, 1], c='y')
    # plt.show()
    k = 4
    X = np.concatenate([X1, X2, X3, X4], axis=0)
    centers,labels = spectral_clustering(X, k)
    print(labels)
    
