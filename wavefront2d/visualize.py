import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

from .patch import Patch

def visualize_ior_fields(ior_fields: list[np.ndarray], titles: list[str]):
    plt.figure(figsize=(16, 4))
    for i, field in enumerate(ior_fields, 1):
        plt.subplot(1, 6, i)
        plt.imshow(field, cmap='Blues', vmin=1.0, vmax=1.5)
        plt.title(titles[i - 1])
    # Share the same colorbar
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    cbar_ax = plt.gcf().add_axes((0.85, 0.15, 0.02, 0.7))
    plt.colorbar(cax=cbar_ax)
    plt.show()

def visualize_gradients(grad_x, grad_y):        
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(grad_x, cmap='gray')
    plt.title('Gradient along x axis')
    plt.xlabel('x - axis[1]')
    plt.ylabel('y - axis[0]')

    plt.subplot(122)
    plt.imshow(grad_y, cmap='gray')
    plt.title('Gradient along y axis')

    plt.xlabel('x - axis[1]')
    plt.ylabel('y - axis[0]')

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    cbar_ax = plt.gcf().add_axes((0.85, 0.15, 0.02, 0.7))
    plt.colorbar(cax=cbar_ax)
    plt.show()

    
def visualize_wavefront_propagation_points(wavefront_pos_list: list[list[tuple]], wavefront_dir_list: list[list[tuple]], 
                                    cur_IOR: np.ndarray, num_steps: int, num_show_images=5):
    # show the wavefront propagation in num_show_images steps between 0 and num_steps
    num_show_images = [i for i in range(0, num_steps + 1, num_steps // num_show_images)]

    plt.figure(figsize=(5 * len(num_show_images), 5))
    for i in num_show_images:
        plt.subplot(1, len(num_show_images), num_show_images.index(i) + 1)
        plt.imshow(cur_IOR, cmap='Blues', vmin=1.0, vmax=1.5)
        plt.scatter([p[0] for p in wavefront_pos_list[i]], [p[1] for p in wavefront_pos_list[i]], color='red')
        for j, (x, y) in enumerate(wavefront_pos_list[i]):
            if j != 0 and j % 5 == 0:
                plt.arrow(x, y, wavefront_dir_list[i][j][0], wavefront_dir_list[i][j][1], color='blue', head_width=3)        
        plt.title(f'Step {i} (Remaining points Number: {len(wavefront_pos_list[i])})')
    plt.show()


def visualize_wavefront_propagation_patches(wavefront_patch_list: list[list[Patch]], 
                                    cur_IOR: np.ndarray, num_show_images=5):
    # show the wavefront propagation in num_show_images steps between 0 and num_steps
    num_steps = len(wavefront_patch_list) - 1
    num_show_images = [i for i in range(0, num_steps + 1, num_steps // num_show_images)]

    plt.figure(figsize=(5 * len(num_show_images), 5))
    for i in num_show_images:
        plt.subplot(1, len(num_show_images), num_show_images.index(i) + 1)
        plt.imshow(cur_IOR, cmap='Blues', vmin=1.0, vmax=1.5)
        
        for j, patch in enumerate(wavefront_patch_list[i]):
            start_pos = patch.start_pos
            end_pos = patch.end_pos
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='red')            
            if j % 20 == 0:
                mid_pos = patch.mid_pos()
                plt.arrow(mid_pos[0], mid_pos[1], patch.start_dir[0], patch.start_dir[1], color='blue', head_width=3)
        
        plt.title(f'Step {i} (Remaining patches Number: {len(wavefront_patch_list[i])})')
    plt.show()

def visualize_compare_irradiance(irradiance1: np.ndarray, irradiance2: np.ndarray, title: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(irradiance1, cmap='hot', interpolation='nearest')
    ax1.set_title('Point-based Irradiance')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    im = ax2.imshow(irradiance2, cmap='hot', interpolation='nearest')
    ax2.set_title('Patch-based Irradiance')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    fig.colorbar(im, ax=[ax1, ax2], label='Irradiance')
    fig.suptitle(title)
    plt.show()


