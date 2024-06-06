import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_tree(padre, h=10, w=10):
    G = nx.DiGraph()

    for i, p in enumerate(padre):
        if p != -1:
            G.add_edge(p, i)
        else:
            root = i
            
    ax = plt.figure(figsize=(w, h))
    
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=False, arrows=False, node_size=10, node_color='skyblue', font_size=10, font_weight='bold', ax=ax)
    plt.show()

def plot_2D_labeled_clusters(X, labels):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        X_label = X[labels == label]
        X_label = np.array(X_label)
        plt.scatter(X_label[:, 0], X_label[:, 1])
    plt.show()

def plot_multiple_2D_labeled_clusters(Xs, labelss):
    n = len(Xs)
    n_cols = (n + 1) // 2
    n_rows = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs = axs.flatten()

    for i in range(n):
        X = Xs[i]
        labels = labelss[i]
        unique_labels = np.unique(labels)
        for label in unique_labels:
            X_label = X[labels == label]
            X_label = np.array(X_label)
            axs[i].scatter(X_label[:, 0], X_label[:, 1])
        axs[i].set_title(f'Dataset {i+1}')

    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def plot_2D_labeled_clusters_with_centers(X, labels, centers):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        X_label = X[labels == label]
        X_label = np.array(X_label)
        plt.scatter(X_label[:, 0], X_label[:, 1])
    plt.scatter(centers[:, 0], centers[:, 1], c='red')
    plt.show()
    
def plot_multiple_2D_labeled_clusters_with_centers(Xs, labelss, centerss):
    n = len(Xs)
    n_cols = (n + 1) // 2
    n_rows = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs = axs.flatten()

    for i in range(n):
        X = Xs[i]
        labels = labelss[i]
        centers = centerss[i]
        unique_labels = np.unique(labels)
        for label in unique_labels:
            X_label = X[labels == label]
            X_label = np.array(X_label)
            axs[i].scatter(X_label[:, 0], X_label[:, 1], s=10, alpha=0.5)
        axs[i].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='x')
        axs[i].set_title(f'Dataset {i+1}')

    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()