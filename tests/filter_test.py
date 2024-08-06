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
                
filtered_no_radius = gaussian_filter(arr, sigma=GAUSSIAN_SIGMA)
filtered_with_radius = gaussian_filter(arr, sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)

filtered_nearest = gaussian_filter(arr, mode='nearest', sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)
filtered_reflect = gaussian_filter(arr, mode='reflect', sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)

print("Original array:")
print(arr)
print("\nFiltered array without radius:")
print(filtered_no_radius)
print("\nFiltered array with radius:")
print(filtered_with_radius)

print("\nFiltered array with mode='nearest':")
print(filtered_nearest)
print("\nFiltered array with mode='reflect':")
print(filtered_reflect)