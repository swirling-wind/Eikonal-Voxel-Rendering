import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from common.figure import save_colorbar, save_color_wheel

def load_images(img_path, origin_path):
    img = cv2.imread(img_path)
    origin_img = cv2.imread(origin_path)
    # 将BGR转换为RGB颜色空间
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
    
    grad_diff_x = sobelx1 - sobelx2
    grad_diff_y = sobely1 - sobely2
    
    grad_magnitude = np.sqrt(grad_diff_x**2 + grad_diff_y**2)
    grad_direction = np.arctan2(grad_diff_y, grad_diff_x)
    
    return np.where(grad_magnitude < threshold, 0, grad_magnitude), grad_direction

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(mse, max_pixel=255.0):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2)

def save_difference_image(diff, index, save_path, vmin, vmax):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(diff, cmap='hot', vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

    # Save separate colorbar image
    # if index == '2':
    #     save_colorbar(os.path.splitext(save_path)[0] + '_colorbar.png', vmin, vmax)

def save_gradient_difference_image(magnitude, direction, index, save_path, intensity=1.5):
    plt.figure(figsize=(10, 8))
    
    # 将方向从弧度转换为度数
    hue = (direction + np.pi) / (2 * np.pi)
    
    # 调整饱和度和明度以加粗梯度差异
    saturation = np.ones_like(magnitude)
    value = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX) # type: ignore
    
    # 应用强度因子
    value = np.clip(value * intensity, 0, 1)
    
    hsv_image = np.stack((hue, saturation, value), axis=2)
    rgb_image = cv2.cvtColor((hsv_image * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    plt.imshow(rgb_image)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def process_index(base_path, file_types, index, num_threshold, grad_threshold, intensity=1.5):
    metrics = {file_type: {} for file_type in file_types}
    num_diffs = {}
    grad_magnitudes = {}
    grad_directions = {}
    
    for file_type in file_types:
        img_path = os.path.join(base_path, f'{file_type}_{index}.png')
        origin_path = os.path.join(base_path, f'Origin_{index}.png')
        
        if not (os.path.exists(img_path) and os.path.exists(origin_path)):
            print(f"Files for {file_type}_{index} not found.")
            continue
        
        img, origin_img = load_images(img_path, origin_path)
        
        num_diff = calculate_numerical_difference(img, origin_img, num_threshold)
        grad_magnitude, grad_direction = calculate_gradient_difference(img, origin_img, grad_threshold)
        
        num_diffs[file_type] = num_diff
        grad_magnitudes[file_type] = grad_magnitude
        grad_directions[file_type] = grad_direction
        
        metrics[file_type]['num_mean'] = np.mean(num_diff)
        metrics[file_type]['num_max'] = np.max(num_diff)
        metrics[file_type]['grad_mean'] = np.mean(grad_magnitude)
        metrics[file_type]['grad_max'] = np.max(grad_magnitude)
        
        mse = calculate_mse(img, origin_img)
        metrics[file_type]['mse'] = mse
        metrics[file_type]['psnr'] = calculate_psnr(mse)
        metrics[file_type]['ssim'] = calculate_ssim(img, origin_img)
    
    if not num_diffs or not grad_magnitudes:
        return None
    
    num_min = min(np.min(diff) for diff in num_diffs.values())
    num_max = max(np.max(diff) for diff in num_diffs.values())
    
    for file_type in file_types:
        if file_type in num_diffs and file_type in grad_magnitudes:
            save_difference_image(num_diffs[file_type], 
                                  index, 
                                  os.path.join(base_path, f'num_diff_{file_type}_{index}.png'),
                                  num_min, num_max)
            save_gradient_difference_image(grad_magnitudes[file_type], grad_directions[file_type],
                                           index, 
                                           os.path.join(base_path, f'grad_diff_{file_type}_{index}.png'),
                                           intensity)
    
    print(f"Differences for index {index} have been calculated and saved.")
    return metrics

def process_all_images(base_path, file_types, num_threshold, grad_threshold, intensity=1.5):
    all_metrics = {}
    
    # 获取所有索引
    indices = set()
    for file_type in file_types:
        for filename in os.listdir(base_path):
            if filename.startswith(f'{file_type}_') and filename.endswith('.png'):
                index = filename[len(file_type)+1:-4]
                indices.add(index)
    
    # 对每个索引进行处理
    for index in indices:
        metrics = process_index(base_path, file_types, index, num_threshold, grad_threshold, intensity)
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


# base_path = os.path.join(os.getcwd(), "images", "Light_wooden_frame_room_2k.hdr", "geometry")
# file_types = ['MLP', 'Siren', 'Octree']
# num_threshold = 10
# grad_threshold = 15
# intensity = 4  # intensity for gradient difference visualization

# all_metrics = process_all_images(base_path, file_types, num_threshold, grad_threshold, intensity)
# print_csv(all_metrics)


## Save color wheel
# save_path = os.path.join(os.getcwd(), "images", "color_wheel.png")
# save_color_wheel(save_path)

## Save colorbar
# save_path = os.path.join(os.getcwd(), "images", "hot_colorbar.png")
# save_colorbar(save_path, vmin=0, vmax=105)