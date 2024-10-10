from numerics import resize_image
import os
import cv2
from tqdm import tqdm

max_pixels = 40000

if __name__ == '__main__':    
    data_dir = 'data/movements'
    output_dir = 'data/resized/movements'

    allowed_extensions = ['jpg', 'png', 'jpeg']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def valid_extension(file_name):
        return file_name.split('.')[-1].lower() in allowed_extensions

    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if valid_extension(file):
                image_paths.append(os.path.join(root, file))

    for image_path in tqdm(image_paths, desc="Resizing images"):
        rel_dir = os.path.relpath(os.path.dirname(image_path), data_dir)
        rel_output_dir = os.path.join(output_dir, rel_dir)
        
        if not os.path.exists(rel_output_dir):
            os.makedirs(rel_output_dir)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}")
            continue
        resized_image = resize_image(image, max_pixels)
        
        new_image_name = f"{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(rel_output_dir, new_image_name), resized_image)
        
        # print(f"Resized {image_path} - {resized_image.shape} - {resized_image.shape[0] * resized_image.shape[1]} pixels")
