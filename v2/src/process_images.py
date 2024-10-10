from clustering import get_dominating_colors_hier, plot_dominating_colors
from numerics import sort_by_hue, sort_by_lightness
import numpy as np
import cv2
import os
from PCA import PCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from tqdm import tqdm
import argparse

def img_to_color_vec(image, k):
    return np.array(sort_by_lightness(get_dominating_colors_hier(image, k))).reshape(-1)

def save_dominating_colors_map(data_dir, output_file, k):
    allowed_extensions = ['jpg', 'png', 'jpeg']
    
    def valid_extension(file_name):
        return file_name.split('.')[-1].lower() in allowed_extensions
    
    img_colors_map = {}
    
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if valid_extension(file):
                all_files.append(os.path.join(root, file))
    
    checkpoint_interval = 100
    checkpoint_file = output_file + '.checkpoint'
    
    print(f"Processing {len(all_files)} images from {data_dir}.")
    print(f"Extracting {k} dominant colors per image.")
    print(f"Saving to {output_file}")
    print(f"Checkpointing to {checkpoint_file}")
    
    processed_count = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            img_colors_map = pickle.load(f)
            processed_count = len(img_colors_map)
            print(f"Loaded checkpoint with {processed_count} images")
            all_files = all_files[processed_count:]
    
    for i, image_path in enumerate(tqdm(all_files, desc="Processing images", initial=processed_count)):
        try:
            image = cv2.imread(image_path)
            color_vec = img_to_color_vec(image, k)
            relative_image_path = os.path.relpath(image_path, data_dir)
            img_colors_map[relative_image_path] = color_vec
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        
        if (i + 1) % checkpoint_interval == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(img_colors_map, f)
            print(f"Saved checkpoint at {i + 1} images")

    with open(output_file, 'wb') as f:
        pickle.dump(img_colors_map, f)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save Dominating Colors Map for Images")

    parser.add_argument('data_dir', type=str, help="Directory containing the images to process", default='data/resized/movements')
    parser.add_argument('output_file', type=str, help="File to save the resulting color map", default='data/processed/colors_map_k2.pkl')
    parser.add_argument('k', type=int, help="Number of dominant colors to extract (k)", default=2)

    args = parser.parse_args()

    save_dominating_colors_map(args.data_dir, args.output_file, args.k)
    
    
    # k = 2    
    # data_dir = 'data/resized'
    
    # allowed_extensions = ['jpg', 'png', 'jpeg']
    
    # def valid_extension(file_name):
    #     return file_name.split('.')[-1] in allowed_extensions
    
    # img_colors_map = {}
    
    # limit = 100
    # count = 0
    # for image_name in os.listdir(data_dir):
    #     if valid_extension(image_name):
    #         count += 1
    #         if count > limit:
    #             break
    #         image = cv2.imread(os.path.join(data_dir, image_name))
    #         color_vec = img_to_color_vec(image, k)
    #         print(f"Color vector for {image_name}: {color_vec}")
    #         img_colors_map[image_name] = color_vec
            
    
            
    
    # pca = PCA(2)
    
    # X = np.array(list(img_colors_map.values()))
    
    
    # pca.fit(X)
    
    # def omitt_prefix(image_name):
    #     return '_'.join(image_name.split('_')[1:])
    
    # transformed_map = {}
    # for image_name, color_vec in img_colors_map.items():
    #     transformed_map[image_name] = pca.transform(color_vec)

    
    # with open('data/colors_map.pkl', 'wb') as f:
    #     pickle.dump(transformed_map, f)
    
    # transformed_map = None
        
    # with open('data/colors_map.pkl', 'rb') as f:
    #     transformed_map = pickle.load(f)
        
        
    # def plot_transformed_map(transformed_map, image_folder):
    #     fig, ax = plt.subplots(figsize=(8, 6))

    #     for image_name, point in transformed_map.items():
    #         x, y = point
    #         ax.scatter(x, y, color='blue', s=0)
    #         # ax.text(x + 0.02, y + 0.02, omitt_prefix(image_name), fontsize=9, color='black')
            
    #         img_path = os.path.join(image_folder, image_name)
    #         print(f"Image path: {img_path}")
    #         if os.path.exists(img_path):
    #             img = mpimg.imread(img_path)
    #             imgbox = ax.inset_axes([x, y, 0.1, 0.1], transform=ax.transData)
    #             imgbox.imshow(img)
    #             imgbox.axis('off')
    #         else:
    #             print(f"Image {image_name} not found")

    #     plt.axis('off')
    #     plt.show()

    
    # plot_transformed_map(transformed_map, data_dir)
        
    # mean = np.mean(X, axis=0).reshape(k, 3)    
    # def pca_vec_to_color_tuple(pca_vec):
    #     reconstructed = pca.reconstruct(pca_vec)
    #     reconstructed = reconstructed.reshape(k, 3)
    #     return reconstructed + mean
        
    # # print(f"Transformed map: {transformed_map}")
    
    # Z = pca.transform(X)
    
    # print(f"Z: {Z}")
    
    # component1 = np.array([0, 1])
    # component2 = np.array([1, 0])
    
    # colors1 = pca_vec_to_color_tuple(component1)
    # colors2 = pca_vec_to_color_tuple(component2)
    
    # print(f"Component 1: {component1} - {colors1}")
    # print(f"Component 2: {component2} - {colors2}")
    
    # print("eigen1:", pca.get_eigenvectors()[0])
    # print("eigen2:", pca.get_eigenvectors()[1])
    
    # plot_dominating_colors(colors1)
    # plot_dominating_colors(colors2)
    