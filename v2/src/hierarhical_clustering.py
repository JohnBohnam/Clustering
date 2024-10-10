import numpy as np
import typing

from numerics import euclidean_distance, manhattan_distance

from plotting import plot_tree



def clust_dist_min(clusterA, clusterB, dists):
    return np.min(dists[clusterA][:, clusterB])

def clust_dist_max(clusterA, clusterB, dists):
    return np.max(dists[clusterA][:, clusterB])

def clust_dist_avg(clusterA, clusterB, dists):
    return np.mean(dists[clusterA][:, clusterB])

# not actually a centroid, since center of mass is only the centroid for the euclidean distance 
def clust_dist_centroid(clusterA, clusterB, dists):
    return np.linalg.norm(np.mean(dists[clusterA], axis=0) - np.mean(dists[clusterB], axis=0))

def hierarchical_naive(X : np.ndarray, k : int, dist_f = euclidean_distance, clust_dist_f = clust_dist_avg) -> typing.Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    dists = dist_f(X[:, None], X, axis=2)
    print(dists)
    clusters = [[i] for i in range(n)]
    while len(clusters) > k:
        min_dist = np.inf
        min_i = -1
        min_j = -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = clust_dist_f(clusters[i], clusters[j], dists)
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j
        
        print(f"merging {min_i} and {min_j} with distance {min_dist}")
        clusters[min_i] += clusters[min_j]
        clusters.pop(min_j)
        
    labels = np.zeros(n, dtype=int)
    for i, cluster in enumerate(clusters):
        labels[cluster] = i   
    return labels

# alpha_A, alpha_B, beta, gamma
def merging_avg_coefs(A_nel: int, B_nel: int):
    return A_nel/(A_nel + B_nel), B_nel/(A_nel + B_nel), 0, 0

def merging_single_coefs(A_nel: int, B_nel: int):
    return 0.5, 0.5, 0, -0.5

def merging_complete_coefs(A_nel: int, B_nel: int):
    return 0.5, 0.5, 0, 0.5

def merging_centroid_coefs(A_nel: int, B_nel: int):
    return A_nel/(A_nel + B_nel), B_nel/(A_nel + B_nel), -(A_nel * B_nel)/(A_nel + B_nel)**2, 0

def merging_ward_coefs(A_nel: int, B_nel: int):
    C_nel = A_nel + B_nel
    return (A_nel + C_nel)/(A_nel + B_nel + C_nel), (B_nel + C_nel)/(A_nel + B_nel + C_nel), -C_nel/(A_nel + B_nel + C_nel), 0


def hierarchical_lance_williams(X : np.ndarray, k: int, dist_f = euclidean_distance, merging_coefs = merging_avg_coefs) -> typing.Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    dists = dist_f(X[:, None], X, axis=2)
    clusters = [[i] for i in range(n)]
    padre = [-1 for i in range(n)]
    true_idx = np.arange(0, n)
    
    while len(clusters) > k:
        # print(dists)
        D_AB = np.inf
        A = -1
        B = -1
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                dist = dists[a, b]
                if dist < D_AB:
                    D_AB = dist
                    A = a
                    B = b
        
        # print(f"merging {A} and {B} with distance {D_AB}")
        # print(f"true idx: {true_idx[A]}, {true_idx[B]}")
        
        padre[true_idx[B]] = len(padre)
        padre[true_idx[A]] = len(padre)
        true_idx[A] = len(padre)
        padre.append(-1)
        
        for i in range(B, len(clusters)-1):
            true_idx[i] = true_idx[i+1]
        alpha_A, alpha_B, beta, gamma = merging_coefs(len(clusters[A]), len(clusters[B]))
        
        newdists = np.zeros(len(clusters))
        for K in range(len(clusters)):
            if K == B or K == A:
                continue
            newdists[K] = alpha_A * dists[A, K] + alpha_B * dists[B, K] + beta * dists[A, B] + gamma * np.abs(dists[A, K] - dists[B, K])
        
        # print(f"newdists: {newdists}")
        dists[A, :] = newdists
        dists[:, A] = newdists
        
        dists = np.delete(dists, B, axis=0)
        dists = np.delete(dists, B, axis=1)
        
        clusters[A] += clusters[B]
        clusters.pop(B)
                
    labels = np.zeros(n, dtype=int)
    for i, cluster in enumerate(clusters):
        labels[cluster] = i
    return labels, padre


if __name__ == '__main__':
    X1 = np.random.randn(400, 2)
    X2 = np.random.randn(400, 2) + 5
    X  = np.concatenate([X1, X2], axis=0)
    labels, padre = hierarchical_lance_williams(X, 2)
    print(labels)
    print(padre)
    plot_tree(padre)