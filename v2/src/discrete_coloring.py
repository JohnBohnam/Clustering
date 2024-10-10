from clustering import plot_dominating_colors, get_dominating_colors_hier
import numpy as np


def assign_centers(image, centers):
    centers = np.array(centers)
    pixels = image.reshape(-1, 3)
    distances = np.sum(np.abs(pixels[:, np.newaxis] - centers), axis=2)
    nearest_centers = np.argmin(distances, axis=1)
    recolored_pixels = centers[nearest_centers]
    recolored_image = recolored_pixels.reshape(image.shape)
    return recolored_image.astype(np.uint8)

def discrete_coloring(image, k):
    centers = (get_dominating_colors_hier(image, k)*255).astype(np.int16)
    return assign_centers(image, centers), centers

    
if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    from clustering import get_dominating_colors_hier
    from hierarhical_clustering import hierarchical_lance_williams as hierarchical_lance_williams

    # image_path = "./data/starry_night.jpg"
    # image_path = "./data/winter_tree.png"
    image_path = "./data/afghan_girl.jpg"
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    
    k = 10
    
    centers = (get_dominating_colors_hier(image, k)*255).astype(np.int16)
    print(centers)
    plot_dominating_colors(centers)
    
    colored_image = assign_centers(image, centers)
    print(np.average(colored_image))
    print(np.std(colored_image))
    # print(colored_image.shape)
    # print(colored_image)
    
    plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
    plt.show()
    # plt.imshow(np.array([colored_image[:, :, 0], colored_image[:, :, 1], colored_image[:, :, 2]]).transpose(1, 2, 0))
    # plt.show()
    
    # plt.imshow(np.array([colored_image[:, :, 0], colored_image[:, :, 2], colored_image[:, :, 1]]).transpose(1, 2, 0))
    # plt.show()
    
    # plt.imshow(np.array([colored_image[:, :, 1], colored_image[:, :, 0], colored_image[:, :, 2]]).transpose(1, 2, 0))
    # plt.show()
    
    # plt.imshow(np.array([colored_image[:, :, 1], colored_image[:, :, 2], colored_image[:, :, 0]]).transpose(1, 2, 0))
    # plt.show()
    
    # plt.imshow(np.array([colored_image[:, :, 2], colored_image[:, :, 0], colored_image[:, :, 1]]).transpose(1, 2, 0))
    # plt.show()
    
    # plt.imshow(np.array([colored_image[:, :, 2], colored_image[:, :, 1], colored_image[:, :, 0]]).transpose(1, 2, 0))
    # plt.show()
    
    
    