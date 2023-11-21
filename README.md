## Matplot3DEx Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Example: 3D Surface Plot](#example-3d-surface-plot)
   - [Example: 3D Wireframe Plot](#example-3d-wireframe-plot)
   - [Example: 3D Bar Plot](#example-3d-bar-plot)
4. [Classes](#classes)
5. [Advanced Features](#advanced-features)
   - [Data Science Module](#data-science-module)
   - [Math Module](#math-module)
   - [Add-Ons Module](#add-ons-module)
6. [More Usage](#more-usage)
   - [Example: 3D Scatter Plot](#example-3d-scatter-plot-1)
   - [Example: 3D Surface Plot](#example-3d-surface-plot-1)
   - [Example: 2D Heatmap](#example-2d-heatmap)
   - [Example: Animated 3D Scatter Plot](#example-animated-3d-scatter-plot)
   - [Example: 2D Hexbin Plot](#example-2d-hexbin-plot)
   - [Example: 3D Quiver Plot](#example-3d-quiver-plot)

## Overview

Matplot3DEx is an extension of Matplotlib, providing a simplified API for creating a variety of 3D and 2D plots with enhanced customization options. This package aims to facilitate the creation of visually appealing and informative plots for data analysis and visualization tasks.

## Installation

To install Matplot3DEx, use the following command:

```bash
pip install Matplot3DEx
```

## Usage

### Example: 3D Surface Plot

```python
from Matplot3DEx import Matplot3DEx
import numpy as np

# Create a meshgrid for the surface plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = x**2 - y**2

# Create a 3D surface plot
surface_plot = Matplot3DEx.Surface3D(x, y, z, title='3D Surface Plot', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
surface_plot.show()
```
![ex1](https://github.com/Ishanoshada/Ishanoshada/blob/main/ss/mex1.png?raw=true)


### Example: 3D Wireframe Plot

```python
from Matplot3DEx import Matplot3DEx
import numpy as np

# Create a meshgrid for the wireframe plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = x**2 - y**2

# Create a 3D wireframe plot
wireframe_plot = Matplot3DEx.Wireframe3D(x, y, z, title='3D Wireframe Plot', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
wireframe_plot.show()
```
![ex2](https://github.com/Ishanoshada/Ishanoshada/blob/main/ss/Untitled.png?raw=true)

### Example: 3D Bar Plot

```python
from Matplot3DEx import Matplot3DEx

# Sample data for 3D bar plot
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
z = [3, 4, 5, 6, 7]

# Dimensions of the bars
dx = 0.8
dy = 0.8
dz = [1, 1, 1, 1, 1]

# Create a 3D bar plot
bar_plot = Matplot3DEx.Bar3D(x, y, z, dx, dy, dz, title='3D Bar Plot', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
bar_plot.show()
```
![ex3](https://github.com/Ishanoshada/Ishanoshada/blob/main/ss/mex3.png?raw=true)

## Classes

The Matplot3DEx package includes the following classes, each designed for specific types of plots:


| Class                             | Description                                             |
|-----------------------------------|---------------------------------------------------------|
| `Matplot3DEx.BasePlot`             | Base class for common plot settings (title, labels).    |
| `Matplot3DEx.Base3DPlot`           | Extension of `BasePlot` for 3D plots with Z-axis labels.|
| `Matplot3DEx.Scatter3D`            | 3D scatter plot with customizable color and markers.    |
| `Matplot3DEx.Surface3D`            | 3D surface plot with adjustable colormap and transparency. |
| `Matplot3DEx.Wireframe3D`          | 3D wireframe plot with customizable color and linewidth.|
| `Matplot3DEx.Bar3D`                | 3D bar plot with customizable dimensions and color.     |
| `Matplot3DEx.Quiver3D`             | 3D quiver plot for vector visualization.                 |
| `Matplot3DEx.Contour3D`            | 3D contour plot with adjustable colormap and levels.    |
| `Matplot3DEx.Heatmap2D`            | 2D heatmap plot with customizable colormap.             |
| `Matplot3DEx.AnimatedScatter3D`    | Animated 3D scatter plot for dynamic data visualization. |
| `Matplot3DEx.Boxplot2D`            | 2D box plot with customizable appearance.               |
| `Matplot3DEx.Hexbin2D`             | 2D hexbin plot with options for gridsize and colormap.  |
| `Matplot3DEx.TriangularMesh3D`     | 3D triangular mesh plot with adjustable colormap.      |
| `Matplot3DEx.Streamline3D`         | 3D streamline plot for visualizing vector fields.       |
| `Matplot3DEx.ColorSizeScatter2D`   | 2D scatter plot with color and size coding.              |
| `Matplot3DEx.GroupedBoxplot2D`     | Grouped 2D box plot for comparing categories.            |
| `Matplot3DEx.PairwiseScatterplotMatrix` | Pairwise scatterplot matrix for exploring relationships. |
| `Matplot3DEx.CorrelationHeatmap`   | Heatmap for visualizing correlation matrices.            |
| `Matplot3DEx.HistogramWithKDE`     | Histogram plot with Kernel Density Estimation (KDE).    |
| `Matplot3DEx.ViolinPlot`           | Violin plot for visualizing distribution and density.   |
| `Matplot3DEx.ParametricCurve`      | 2D plot of a parametric curve.                           |
| `Matplot3DEx.PolarRose`            | Polar plot representing a rose curve.                    |
| `Matplot3DEx.ParametricSurface3D`  | 3D parametric surface plot.                             |
| `Matplot3DEx.SaddleSurface`        | 3D plot of a saddle surface.                            |
| `Matplot3DEx.QuaternionRotation`   | 3D plot of a vector rotation using quaternions.         |
| `Matplot3DEx.TimeSeriesRollingAverage` | Time series plot with rolling average.               |
| `Matplot3DEx.HistogramOutlierDetection` | Histogram with outlier detection.                  |
| `Matplot3DEx.ScatterRegressionPlot` | Scatter plot with linear regression line.             |
| `Matplot3DEx.StackedAreaTimeSeries` | Stacked area plot for time series data.              |
| `Matplot3DEx.ConfusionMatrixHeatmap` | Heatmap for visualizing confusion matrices.         |


## Advanced Features

Matplot3DEx also provides advanced features for specific use cases:

- **Data Science Module**: Includes classes for statistical analysis and data exploration.
- **Math Module**: Incorporates mathematical plots and functions for mathematical visualizations.
- **Add-Ons Module**: Additional classes that extend the functionality of Matplot3DEx.

## More Usage

### Example: 3D Scatter Plot

```python
import numpy as np
from Matplot3DEx import Matplot3DEx

# Generate sample data
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

# Create a 3D scatter plot
scatter_plot = Matplot3DEx.Scatter3D(x, y, z, title='3D Scatter Plot', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
scatter_plot.show()
```

### Example: 3D Surface Plot

```python
import numpy as np
from Matplot3DEx import Matplot3DEx

# Generate sample data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = x**2 - y**2

# Create a 3D surface plot
surface_plot = Matplot3DEx.Surface3D(x, y, z, title='3D Surface Plot', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
surface_plot.show()
```

### Example: 2D Heatmap

```python
from Matplot3DEx import Matplot3DEx
import numpy as np

# Generate sample data
data = np.random.rand(10, 10)

# Create a 2D heatmap
heatmap_plot = Matplot3DEx.Heatmap2D(data, title='2D Heatmap', xlabel='X-axis', ylabel='Y-axis')
heatmap_plot.show()
```

### Example: Animated 3D Scatter Plot

```python
import numpy as np
from Matplot3DEx import Matplot3DEx

# Generate sample data
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

# Create an animated 3D scatter plot
animated_scatter = Matplot3DEx.AnimatedScatter3D(x, y, z, title='Animated 3D Scatter Plot', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
animated_scatter.animate(frames=50, interval=100)
```

### Example: 2D Hexbin Plot

```python
import numpy as np
from Matplot3DEx import Matplot3DEx

# Generate sample data
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)

# Create a 2D hexbin plot
hexbin_plot = Matplot3DEx.Hexbin2D(x, y, gridsize=30, cmap='viridis')
hexbin_plot.show()
```

### Example: 3D Quiver Plot

```python
import numpy as np
from Matplot3DEx import Matplot3DEx

# Generate sample data
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
z = np.linspace(-5, 5, 20)
x, y, z = np.meshgrid(x, y, z)
u = np.sin(x + y + z)
v = np.cos(x - y - z)
w = np.sin(2 * x)

# Create a 3D quiver plot
quiver_plot = Matplot3DEx.Quiver3D(x, y, z, u, v, w, title='3D Quiver Plot', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
quiver_plot.show()
```

## Contributing

If you find any issues or have suggestions for improvements, feel free to contribute! Visit the [GitHub repository](https://github.com/ishanoshada/matplot3dex) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ishanoshada/matplot3dex/blob/main/LICENSE) file for details.
