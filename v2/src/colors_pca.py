import pickle
from PCA import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

def plot_transformed_map(transformed_map, image_folder):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for image_name, point in transformed_map.items():
        x, y = point
        ax.scatter(x, y, color='blue', s=0)
        # ax.text(x + 0.02, y + 0.02, omitt_prefix(image_name), fontsize=9, color='black')
        
        img_path = os.path.join(image_folder, image_name)
        # print(f"Image path: {img_path}")
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            imgbox = ax.inset_axes([x, y, 0.1, 0.1], transform=ax.transData)
            imgbox.imshow(img)
            imgbox.axis('off')
        else:
            print(f"Image {image_name} not found")

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    
    path_to_color_map = "./data/processed/movements_clr_map_k3.pkl"
    data_dir = "./data/movements"
    
    with open(path_to_color_map, 'rb') as f:
        img_colors_map = pickle.load(f)
    
    pca = PCA(2)
    X = np.array(list(img_colors_map.values()))
    
    pca.fit(X)
    
    n_show = 1000
    
    transformed_map = {}
    for image_name, color_vec in random.sample(list(img_colors_map.items()), n_show):
        transformed_map[image_name] = pca.transform(color_vec)
    
        
    plot_transformed_map(transformed_map, data_dir)