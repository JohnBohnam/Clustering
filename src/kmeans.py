import numpy as np
import typing

from numerics import euclidean_distance, manhattan_distance, classification_impurity, normalize_labels




def kmeans(X : np.ndarray, k : int, dist_f = euclidean_distance, max_iter : int = 1000, centers=None,
           verbose=False) -> typing.Tuple[np.ndarray, np.ndarray]:
    if centers is None:
        centers = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iter):
        if verbose and _ % 10 == 0:
            print(f"Iteration {_}")
        labels = np.argmin(dist_f(X[:, None], centers, axis=2), axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    if verbose:
        print(f"Finished in {_} iterations")
    return centers, labels

def nkmeans(X : np.ndarray, k : int, dist_f = euclidean_distance, max_iter : int = 1000, n = 20, impurity_fun = classification_impurity) -> typing.Tuple[np.ndarray, np.ndarray]:
    best_centers = None
    best_labels = None
    best_impurity = np.inf
    for _ in range(n):
        try:
            centers, labels = kmeans(X, k, dist_f, max_iter)
        except:
            print("Failed")
            continue
        labels = normalize_labels(labels)
        impurity = impurity_fun(X, labels)
        if impurity < best_impurity:
            best_impurity = impurity
            best_centers = centers
            best_labels = labels
    return best_centers, best_labels

def kmeanspp(X : np.ndarray, k : int, dist_f = euclidean_distance, max_iter : int = 1000,
             verbose=False) -> typing.Tuple[np.ndarray, np.ndarray]:
    centers = []
    centers.append(X[np.random.choice(X.shape[0])])
    for _ in range(k - 1):
        dists = [dist_f(X, center) for center in centers]
        D = np.min(dists, axis=0)
        
        prob = D**2 / np.sum(D**2)
        centers.append(X[np.random.choice(X.shape[0], p=prob)])
    centers = np.array(centers)
    return kmeans(X, k, dist_f, max_iter, centers=centers, verbose=verbose)

def nkmeanspp(X : np.ndarray, k : int, dist_f = euclidean_distance, max_iter : int = 1000, n = 20, impurity_fun = classification_impurity,
              verbose=False) -> typing.Tuple[np.ndarray, np.ndarray]:
    best_centers = None
    best_labels = None
    best_impurity = np.inf
    for _ in range(n):
        if verbose:
            print(f"Attempt {_}")
        centers, labels = kmeanspp(X, k, dist_f, max_iter, verbose=verbose)
        impurity = impurity_fun(X, labels)
        if impurity < best_impurity:
            best_impurity = impurity
            best_centers = centers
            best_labels = labels
    return best_centers, best_labels


if __name__ == '__main__':
    X1 = np.random.randn(1000, 2)
    X2 = np.random.randn(1000, 2) + 5
    X = np.concatenate([X1, X2], axis=0)
    centers, labels = kmeans(X, 2)
    print(centers)
    print(labels)
    centers, labels = nkmeanspp(X, 2, n=10)
    print(centers)
    print(labels)
    # seems to work lol
