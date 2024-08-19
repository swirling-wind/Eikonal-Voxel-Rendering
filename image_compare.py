import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def load_images(img_path, origin_path):
    img = cv2.imread(img_path)
    origin_img = cv2.imread(origin_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    return img, origin_img

def calculate_numerical_difference(img1, img2, threshold):
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32)).mean(axis=2)
    return np.where(diff < threshold, 0, diff)

def calculate_gradient_difference(img1, img2, threshold):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_diff_x = np.abs(sobelx1 - sobelx2)
    grad_diff_y = np.abs(sobely1 - sobely2)
    
    grad_diff = np.sqrt(grad_diff_x**2 + grad_diff_y**2)
    return np.where(grad_diff < threshold, 0, grad_diff)

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(mse, max_pixel=255.0):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2)

def save_difference_image(diff, title, save_path, vmin, vmax):
    # plt.figure(figsize=(10, 8))
    plt.imshow(diff, cmap='hot', vmin=vmin, vmax=vmax)
    # plt.colorbar()
    # plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #         hspace = 0, wspace = 0)
    # plt.margins(0,0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def process_index(base_path, file_types, index, num_threshold, grad_threshold):
    metrics = {file_type: {} for file_type in file_types}
    num_diffs = {}
    grad_diffs = {}
    
    for file_type in file_types:
        img_path = os.path.join(base_path, f'{file_type}_{index}.png')
        origin_path = os.path.join(base_path, f'Origin_{index}.png')
        
        if not (os.path.exists(img_path) and os.path.exists(origin_path)):
            print(f"Files for {file_type}_{index} not found.")
            continue
        
        img, origin_img = load_images(img_path, origin_path)
        
        num_diff = calculate_numerical_difference(img, origin_img, num_threshold)
        grad_diff = calculate_gradient_difference(img, origin_img, grad_threshold)
        
        num_diffs[file_type] = num_diff
        grad_diffs[file_type] = grad_diff
        
        metrics[file_type]['num_mean'] = np.mean(num_diff)
        metrics[file_type]['num_max'] = np.max(num_diff)
        metrics[file_type]['grad_mean'] = np.mean(grad_diff)
        metrics[file_type]['grad_max'] = np.max(grad_diff)
        
        mse = calculate_mse(img, origin_img)
        metrics[file_type]['mse'] = mse
        metrics[file_type]['psnr'] = calculate_psnr(mse)
        metrics[file_type]['ssim'] = calculate_ssim(img, origin_img)
    
    if not num_diffs or not grad_diffs:
        return None
    
    num_min = min(np.min(diff) for diff in num_diffs.values())
    num_max = max(np.max(diff) for diff in num_diffs.values())
    grad_min = min(np.min(diff) for diff in grad_diffs.values())
    grad_max = max(np.max(diff) for diff in grad_diffs.values())
    
    for file_type in file_types:
        if file_type in num_diffs and file_type in grad_diffs:
            save_difference_image(num_diffs[file_type], 
                                  f'Numerical Difference ({file_type}, Index: {index}, Threshold: {num_threshold})', 
                                  os.path.join(base_path, f'num_diff_{file_type}_{index}.png'),
                                  num_min, num_max)
            save_difference_image(grad_diffs[file_type], 
                                  f'Gradient Difference ({file_type}, Index: {index}, Threshold: {grad_threshold})', 
                                  os.path.join(base_path, f'grad_diff_{file_type}_{index}.png'),
                                  grad_min, grad_max)
    
    print(f"Differences for index {index} have been calculated and saved.")
    return metrics

def process_all_images(base_path, file_types, num_threshold, grad_threshold):
    all_metrics = {}
    
    indices = set()
    for file_type in file_types:
        for filename in os.listdir(base_path):
            if filename.startswith(f'{file_type}_') and filename.endswith('.png'):
                index = filename[len(file_type)+1:-4]
                indices.add(index)
    
    for index in indices:
        metrics = process_index(base_path, file_types, index, num_threshold, grad_threshold)
        if metrics:
            all_metrics[index] = metrics
    
    return all_metrics

def print_csv(all_metrics):
    headers = ['Index', 'Type', 'Num Diff Mean', 'Num Diff Max', 'Grad Diff Mean', 'Grad Diff Max', 'MSE', 'PSNR', 'SSIM']
    print(','.join(headers))
    
    for index, metrics in all_metrics.items():
        for file_type, values in metrics.items():
            row = [
                index,
                file_type,
                f"{values['num_mean']:.4f}",
                f"{values['num_max']:.4f}",
                f"{values['grad_mean']:.4f}",
                f"{values['grad_max']:.4f}",
                f"{values['mse']:.4f}",
                f"{values['psnr']:.4f}",
                f"{values['ssim']:.4f}"
            ]
            print(','.join(row))

base_path = os.path.join(os.getcwd(), "images", "Light_wooden_frame_room_2k.hdr", "bunny")
file_types = ['MLP', 'Siren', 'Octree']
num_threshold = 15 
grad_threshold = 15

all_metrics = process_all_images(base_path, file_types, num_threshold, grad_threshold)
print_csv(all_metrics)
