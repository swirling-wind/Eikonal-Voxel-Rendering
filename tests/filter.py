import numpy as np
from scipy.ndimage import gaussian_filter

GAUSSIAN_SIGMA = 2.0
GAUSSIAN_RADIUS = 2

# Create (7*7) array
arr = np.array([[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]], dtype=float)
                

# 使用不同的mode参数应用高斯滤波
filtered_constant = gaussian_filter(arr,  mode='constant', sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)
filtered_nearest = gaussian_filter(arr, mode='nearest', sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)
filtered_reflect = gaussian_filter(arr, mode='reflect', sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)

# 打印结果
print("Original array:")
print(arr)
print("\nFiltered array with mode='constant':")
print(filtered_constant)
print("\nFiltered array with mode='nearest':")
print(filtered_nearest)
print("\nFiltered array with mode='reflect':")
print(filtered_reflect)