import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate a synthetic regression line in higher dimensions (e.g., 5D)
np.random.seed(42)
x = np.linspace(0, 10, 1000)  # 100 points along the regression line
regression_line = np.array([x, 2*x + 1, 0.5*x - 1, np.sin(x), np.log1p(x)]).T  # 5D data

# Step 2: Initialize UMAP for dimensionality reduction to 2D
reducer = umap.UMAP(n_components=2, random_state=42)

# Step 3: Fit and transform the data
regression_line_2d = reducer.fit_transform(regression_line)

# Step 4: Visualize the 2D projection
plt.figure(figsize=(8, 6))
plt.scatter(regression_line_2d[:, 0], regression_line_2d[:, 1], c=x, cmap='viridis', s=50)
plt.colorbar(label='Original X Values')
plt.title('2D Projection of Higher-Dimensional Regression Line')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()