import networkx as nx
import numpy as np
import cv2

def centers_from_labels(labels, X):
    centers = []
    for i in range(len(np.unique(labels))):
        centers.append(np.mean(X[labels == i], axis=0))
    return np.array(centers)

def get_hue(color):
    r, g, b = color
    hue = np.arctan2(np.sqrt(3)*(g - b), 2*r - g - b)
    return hue

def sort_by_hue(colors):
    return colors[np.argsort([get_hue(color) for color in colors])]

def get_lightness(color):
    r, g, b = color
    return (max(r, g, b) + min(r, g, b))/2


def sort_by_lightness(colors):
    return colors[np.argsort([get_lightness(color) for color in colors])]


def diag_rev_sqrt(D: np.ndarray) -> np.ndarray:
    res = np.zeros_like(D)
    for i in range(D.shape[0]):
        if D[i, i] == 0:
            res[i, i] = 0
        res[i, i] = 1/np.sqrt(D[i, i])
    return res

def euclidean_distance(X : np.ndarray, Y : np.ndarray, axis=-1) -> np.ndarray:
    return np.linalg.norm(X - Y, axis=axis)

def manhattan_distance(X : np.ndarray, Y : np.ndarray, axis=-1) -> np.ndarray:
    return np.sum(np.abs(X - Y), axis=axis)

def normalize_labels(labels : np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    mapping = {label: i for i, label in enumerate(unique_labels)}
    return np.array([mapping[label] for label in labels])

def classification_impurity(X : np.ndarray, labels : np.ndarray) -> float:
    n = X.shape[0]
    unique_labels = np.unique(labels)
    clusters = [[] for _ in unique_labels]
    for i, label in enumerate(labels):
        clusters[label].append(i)

        
    centers = np.array([np.mean(X[cluster], axis=0) for cluster in clusters])
    impurity = 0
    for i, cluster in enumerate(clusters):
        impurity += np.sum(np.linalg.norm(X[cluster] - centers[i], axis=1))
    return impurity/n

def gen_G(padre):
    G = nx.DiGraph()
    for i, p in enumerate(padre):
        if p != -1:
            G.add_edge(p, i)
    return G

def resize_image(image, max_pixels=300):
    width, height = image.shape[:2]
    ratio = 1.0
    if width * height > max_pixels:
        ratio = np.sqrt(max_pixels / (width * height))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def rgb_to_cmyk_array(rgb_array):
    cmyk_array = np.zeros((rgb_array.shape[0], 4))
    
    for i, (r, g, b) in enumerate(rgb_array):
        k = 1 - max(r, g, b)
        
        if k < 1:
            c = (1 - r - k) / (1 - k)
            m = (1 - g - k) / (1 - k)
            y = (1 - b - k) / (1 - k)
        else:
            c = 0
            m = 0
            y = 0
        cmyk_array[i] = [c, m, y, k]
    return cmyk_array
