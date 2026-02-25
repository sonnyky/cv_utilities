import numpy as np

# Pick a test point (center of image)
test_point = np.array([960, 540, 1])  # Center of 1920x1080

H_cv = np.array([
    [2.31, 0.279, -0.8902261],
    [-0.7466666, 1.9978, -0.186754],
    [2431.296, 2936.25, 1]
])

# Generate a grid of points (UV space)
w, h = 1920, 1080
uv_grid = np.array([
    [0, 0, 1],    # Bottom-left
    [1, 0, 1],    # Bottom-right
    [0, 1, 1],    # Top-left
    [1, 1, 1],    # Top-right
    [0.5, 0.5, 1] # Center
])
# Apply homography
transformed = H_cv @ test_point
transformed /= transformed[2]  # Normalize

print("Original Point:", test_point[:2])
print("Transformed Point:", transformed[:2])

for pt in uv_grid:
    transformed = H_cv @ pt
    transformed /= transformed[2]  # Normalize

    print(f"Original: {pt[:2]} -> Transformed: {transformed[:2]}")
