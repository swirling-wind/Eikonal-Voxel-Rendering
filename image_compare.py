import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_images(image1_path, image2_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    # 将BGR转换为RGB颜色空间
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return img1, img2

def calculate_numerical_difference(img1, img2):
    return np.abs(img1.astype(np.float32) - img2.astype(np.float32)).mean(axis=2)

def calculate_gradient_difference(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_diff_x = np.abs(sobelx1 - sobelx2)
    grad_diff_y = np.abs(sobely1 - sobely2)
    
    return np.sqrt(grad_diff_x**2 + grad_diff_y**2)

def calculate_divergence_difference(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    laplacian1 = cv2.Laplacian(gray1, cv2.CV_64F)
    laplacian2 = cv2.Laplacian(gray2, cv2.CV_64F)
    return np.abs(laplacian1 - laplacian2)

def apply_threshold(diff, threshold):
    return np.where(diff < threshold, 0, diff)

def visualize_differences(img1, img2, num_diff, grad_diff, div_diff, thresholds):
    num_diff_thresholded = apply_threshold(num_diff, thresholds['numerical'])
    grad_diff_thresholded = apply_threshold(grad_diff, thresholds['gradient'])
    div_diff_thresholded = apply_threshold(div_diff, thresholds['divergence'])
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    axs[0, 0].imshow(img1)
    axs[0, 0].set_title('Image 1')
    axs[0, 1].imshow(img2)
    axs[0, 1].set_title('Image 2')
    
    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(im, cax=cax)
    
    im2 = axs[0, 2].imshow(num_diff_thresholded, cmap='hot')
    axs[0, 2].set_title(f'Numerical Difference (threshold: {thresholds["numerical"]})')
    add_colorbar(im2, axs[0, 2])
    
    im3 = axs[1, 0].imshow(grad_diff_thresholded, cmap='hot')
    axs[1, 0].set_title(f'Gradient Difference (threshold: {thresholds["gradient"]})')
    add_colorbar(im3, axs[1, 0])
    
    im4 = axs[1, 1].imshow(div_diff_thresholded, cmap='hot')
    axs[1, 1].set_title(f'Divergence Difference (threshold: {thresholds["divergence"]})')
    add_colorbar(im4, axs[1, 1])
    
    combined_diff = num_diff_thresholded + grad_diff_thresholded + div_diff_thresholded
    im5 = axs[1, 2].imshow(combined_diff, cmap='hot')
    axs[1, 2].set_title('Combined Difference (thresholded)')
    add_colorbar(im5, axs[1, 2])
    
    plt.tight_layout()
    plt.show()

def compare_images(image1_path, image2_path, thresholds):
    img1, img2 = load_images(image1_path, image2_path)
    
    num_diff = calculate_numerical_difference(img1, img2)
    grad_diff = calculate_gradient_difference(img1, img2)
    div_diff = calculate_divergence_difference(img1, img2)
    
    visualize_differences(img1, img2, num_diff, grad_diff, div_diff, thresholds)
image1_path = os.path.join(os.getcwd(), "images", "Light_wooden_floor_room_4k.hdr", "bunny", "MLP_0.png")
image2_path = os.path.join(os.getcwd(), "images", "Light_wooden_floor_room_4k.hdr", "bunny", "Origin_0.png")
print(image1_path)
print(image2_path)
thresholds = {
    'numerical': 10,  # 0-255 范围内的阈值
    'gradient': 30,   # 根据图像特征调整
    'divergence': 50 # 根据图像特征调整
}
compare_images(image1_path, image2_path, thresholds)
# compare_images(image1_path, image2_path)