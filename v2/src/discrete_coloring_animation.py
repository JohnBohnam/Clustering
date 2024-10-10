import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import os

from discrete_coloring import discrete_coloring, assign_centers
from clustering import get_dominating_colors_kmeans, get_dominating_colors_spectral, reverse_rgb
from numerics import resize_image, sort_by_hue

def create_frame(image, counter):
    image_with_counter = image.copy()
    image_with_counter = cv2.cvtColor(image_with_counter, cv2.COLOR_BGR2RGB)
    if counter is None:
        return image_with_counter
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (20, 30) 
    font_scale = 1
    color = (255, 255, 255) 
    thickness = 2
    cv2.putText(image_with_counter, f"K = {counter}", position, font, font_scale, color, thickness)
    
    return image_with_counter


def create_frame_with_colors(image, counter, colors=None):
    image_with_counter = image.copy()
    
    image_with_counter = cv2.cvtColor(image_with_counter, cv2.COLOR_BGR2RGB)
    
    if counter is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (20, 30) 
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        cv2.putText(image_with_counter, f"K = {counter}", position, font, font_scale, color, thickness)

    if colors is not None:
        colors = reverse_rgb(colors)      
        colors = sort_by_hue(colors)
        num_colors = len(colors)
        
        color_strip_height = 50 
        color_strip = np.zeros((color_strip_height, image_with_counter.shape[1], 3), dtype=np.uint8)

        rect_width = image_with_counter.shape[1] // num_colors
        
        for i, color in enumerate(colors):
            start_x = i * rect_width
            end_x = (i + 1) * rect_width if i < num_colors - 1 else image_with_counter.shape[1]
            color_strip[:, start_x:end_x] = color 
            
        final_frame = np.vstack((image_with_counter, color_strip))
    else:
        final_frame = image_with_counter

    return final_frame


def animate_image_array(frames_array, output_path='output/animation', gif=True, show_labels=True):
    image_array, labels = zip(*frames_array)
    fig, ax = plt.subplots()
    
    im = ax.imshow(image_array[0])
    ax.axis('off')

    def update(frame_number):
        label = labels[frame_number] if show_labels else None
        im.set_data(create_frame(image_array[frame_number], label))
        return [im]

    anim = FuncAnimation(fig, update, frames=len(image_array), interval=1000, blit=True)
    
    anim.save(output_path+'.mp4', writer='ffmpeg', fps=1)
    if gif:
        anim.save(output_path+'.gif', writer='imagemagick', fps=1)
    
    plt.close(fig)  

default_ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 50, 100]
# default_ks = [1, 2, 3, 4, 5]

def get_frames_array_hier(image, ks=default_ks):
    frames_array = []
    for k in ks:
        # print(f"Processing k={k}")
        colored_image, centers = discrete_coloring(image, k)
        frames_array.append((colored_image, k, centers))
    return frames_array

def get_frames_array_kmeans(image, ks=default_ks):
    frames_array = []
    for k in ks:
        # print(f"Processing k={k}")
        centers = (get_dominating_colors_kmeans(image, k)*255).astype(np.int16)
        colored_image = assign_centers(image, centers)
        frames_array.append((colored_image, k, centers))
    return frames_array

def get_frames_array_spectral(image, ks=default_ks):
    frames_array = []
    for k in ks:
        # print(f"Processing k={k}")
        centers = (get_dominating_colors_spectral(image, k)*255).astype(np.int16)
        colored_image = assign_centers(image, centers)
        frames_array.append((colored_image, k, centers))
    return frames_array

def generate_animation(image, ks=default_ks, output_path='output/animation', gif=True):
    frames_array = get_frames_array_hier(image, ks)
    animate_image_array(frames_array, output_path, gif)


def interactive_plot(frames_array):
    image_array, labels = zip(*frames_array)
    initial_frame = 0

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25) 
    ax.axis('off') 
    image_display = ax.imshow(image_array[initial_frame])

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 0, len(image_array)-1, valinit=initial_frame, valstep=1)

    def update(val):
        frame_index = int(slider.val) 
        image_display.set_data(create_frame(image_array[frame_index], labels[frame_index]))
        fig.canvas.draw_idle()  
    slider.on_changed(update)

    plt.show()
    
def interactive_plots(frame_array1, frame_array2, frame_array3, title1='Hierarchical', title2='KMeans', title3='Spectral'):
    image_array1, labels1, centers_list1 = zip(*frame_array1)
    image_array2, labels2, centers_list2 = zip(*frame_array2)
    image_array3, labels3, centers_list3 = zip(*frame_array3)
    initial_frame = 0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    for ax in [ax1, ax2, ax3]:
        ax.axis('off')

    img_display1 = ax1.imshow(image_array1[initial_frame])
    img_display2 = ax2.imshow(image_array2[initial_frame])
    img_display3 = ax3.imshow(image_array3[initial_frame])
    
    ax1.set_title(title1)
    ax2.set_title(title2)
    ax3.set_title(title3)

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Frame', 0, len(image_array1)-1, valinit=initial_frame, valstep=1)

    def update(val):
        frame_index = int(slider.val) 
        img_display1.set_data(create_frame_with_colors(image_array1[frame_index], None, centers_list1[frame_index]))
        img_display2.set_data(create_frame_with_colors(image_array2[frame_index], None, centers_list2[frame_index]))
        img_display3.set_data(create_frame_with_colors(image_array3[frame_index], None, centers_list3[frame_index]))
        fig.canvas.draw_idle() 

    slider.on_changed(update)

    plt.show()
    

def save_animations(frame_array1, frame_array2, frame_array3, output_path='output/animation', title1='Hierarchical', title2='KMeans', title3='Spectral'):
    image_array1, labels1, centers1 = zip(*frame_array1)
    image_array2, labels2, centers2 = zip(*frame_array2)
    image_array3, labels3, centers3 = zip(*frame_array3)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    for ax in [ax1, ax2, ax3]:
        ax.axis('off')

    img_display1 = ax1.imshow(image_array1[0])
    img_display2 = ax2.imshow(image_array2[0])
    img_display3 = ax3.imshow(image_array3[0])

    ax1.set_title(title1 + ' K = ' + str(labels1[0]))
    ax2.set_title(title2 + ' K = ' + str(labels2[0]))
    ax3.set_title(title3 + ' K = ' + str(labels3[0]))

    def update(frame_index):
        img_display1.set_data(create_frame_with_colors(image_array1[frame_index], None, centers1[frame_index]))
        img_display2.set_data(create_frame_with_colors(image_array2[frame_index], None, centers2[frame_index]))
        img_display3.set_data(create_frame_with_colors(image_array3[frame_index], None, centers3[frame_index]))
        
        ax1.set_title(title1 + ' K = ' + str(labels1[frame_index]))
        ax2.set_title(title2 + ' K = ' + str(labels2[frame_index]))
        ax3.set_title(title3 + ' K = ' + str(labels3[frame_index]))
        
        return [img_display1, img_display2, img_display3]

    anim = FuncAnimation(fig, update, frames=len(image_array1), interval=1000, blit=True)

    anim.save(output_path+'.mp4', writer='ffmpeg', fps=1)
    anim.save(output_path+'.gif', writer='imagemagick', fps=1)
    plt.close(fig)


if __name__ == '__main__':
    
    ks = default_ks
    
    def save_all(image_path):
        print(f"Processing {image_path}")
        image = cv2.imread(image_path)
        max_pixels = 40000
        image = resize_image(image, max_pixels)
        
        frames_hier = get_frames_array_hier(image, ks)
        frames_kmeans = get_frames_array_kmeans(image, ks)
        frames_spectral = get_frames_array_spectral(image, ks)
        
        output_path = f'output/{(image_path.split("/")[-1]).split(".")[0]}' 
        save_animations(frames_hier, frames_kmeans, frames_spectral, output_path=output_path)
    
    # pahts = [
    #     "./data/london.jpg",
    #     "./data/afghan_girl.jpg",
    #     "./data/mondrian.jpg",
    #     "./data/starry_night.jpg",
    #     "./data/winter_tree.png",
    #     "./data/rycerz.jpeg",
    #     "./data/girl_with_earring.jpg",
    #     "./data/stanczyk.jpeg",
        
    # ]
    
    # all paths in the data dir (only jpg and png files)
    
    paths = os.listdir("./data")
    paths = [f"./data/{path}" for path in paths if path.endswith(".jpg") or path.endswith(".png") or path.endswith(".jpeg")]
    
    for path in paths:
        save_all(path)