from hierarhical_clustering import hierarchical_lance_williams
from sklearn.cluster import KMeans, SpectralClustering
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numerics import centers_from_labels, resize_image


max_pixels = 300


def reverse_rgb(rgb_arr):
    return np.array([rgb_arr[:, 2], rgb_arr[:, 1], rgb_arr[:, 0]]).T

def get_dominating_colors_hier(image, k):
    image = resize_image(image, max_pixels)
    image = image / 255
    # print("resized image shape:", image.shape,  " : ", image.shape[0] * image.shape[1])
    labels, padre = hierarchical_lance_williams(image.reshape(-1, 3), k)
    return centers_from_labels(labels, image.reshape(-1, 3))

def get_dominating_colors_kmeans(image, k):
    image = resize_image(image, max_pixels)
    image = image / 255
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(image.reshape(-1, 3))
    return kmeans.cluster_centers_

def get_dominating_colors_spectral(image, k):
    image = resize_image(image, max_pixels)
    image = image / 255
    spectral = SpectralClustering(n_clusters=k, random_state=0).fit(image.reshape(-1, 3))
    labels = spectral.labels_
    return centers_from_labels(labels, image.reshape(-1, 3))    


def plot_dominating_colors(colors):
    colors = reverse_rgb(colors)
    plt.imshow([colors])
    plt.axis('off')
    plt.show()
    

if __name__ == '__main__':
    image_path = 'data/afghan_girl.jpg'
    
    image = cv2.imread(image_path)
    colors = get_dominating_colors_hier(image, 5)
    print(colors)
    plot_dominating_colors(colors)
