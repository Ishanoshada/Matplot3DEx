import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np

class Matplot3DEx:
    def __init__(self):
        pass 

    class BasePlot:
        def __init__(self, title=None, xlabel=None, ylabel=None):
            self.title = title
            self.xlabel = xlabel
            self.ylabel = ylabel

        def set_labels(self, ax):
            if self.xlabel:
                ax.set_xlabel(self.xlabel)
            if self.ylabel:
                ax.set_ylabel(self.ylabel)

        def set_title(self, ax):
            if self.title:
                ax.set_title(self.title)

    class Base3DPlot(BasePlot):
        def __init__(self, title=None, xlabel=None, ylabel=None, zlabel=None):
            super().__init__(title, xlabel, ylabel)
            self.zlabel = zlabel

        def set_zlabel(self, ax):
            if self.zlabel:
                ax.set_zlabel(self.zlabel)

    class Scatter3D(Base3DPlot):
        def __init__(self, x, y, z, title=None, xlabel=None, ylabel=None, zlabel=None,
                     color='blue', marker='o', size=20):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.x = x
            self.y = y
            self.z = z
            self.color = color
            self.marker = marker
            self.size = size

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D scatter plot
            ax.scatter(self.x, self.y, self.z, c=self.color, marker=self.marker, s=self.size)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Display the plot
            plt.show()

        def animate(self, frames=100, interval=100, repeat=True):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            def update(frame):
                ax.clear()
                ax.scatter(self.x[:frame], self.y[:frame], self.z[:frame], c=self.color,
                           marker=self.marker, s=self.size)

                self.set_labels(ax)
                self.set_title(ax)
                self.set_zlabel(ax)

            ani = animation.FuncAnimation(fig, update, frames=len(self.x), interval=interval, repeat=repeat)

            plt.show()

    class Surface3D(Base3DPlot):
        def __init__(self, x, y, z, title=None, xlabel=None, ylabel=None, zlabel=None,
                     cmap='viridis', alpha=1.0):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.x = x
            self.y = y
            self.z = z
            self.cmap = cmap
            self.alpha = alpha

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D surface plot
            surface = ax.plot_surface(self.x, self.y, self.z, cmap=self.cmap, alpha=self.alpha)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Add colorbar for better interpretation
            fig.colorbar(surface, ax=ax, pad=0.1)

            # Display the plot
            plt.show()

    class Wireframe3D(Base3DPlot):
        def __init__(self, x, y, z, title=None, xlabel=None, ylabel=None, zlabel=None,
                     color='red', linewidth=1.0):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.x = x
            self.y = y
            self.z = z
            self.color = color
            self.linewidth = linewidth

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D wireframe plot
            ax.plot_wireframe(self.x, self.y, self.z, color=self.color, linewidth=self.linewidth)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Display the plot
            plt.show()

    class Bar3D(Base3DPlot):
        def __init__(self, x, y, z, dx=1, dy=1, dz=1, title=None, xlabel=None, ylabel=None, zlabel=None,
                     color='purple', alpha=1.0):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.x = x
            self.y = y
            self.z = z
            self.dx = dx
            self.dy = dy
            self.dz = dz
            self.color = color
            self.alpha = alpha

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D bar plot
            ax.bar3d(self.x, self.y, self.z, self.dx, self.dy, self.dz, color=self.color, alpha=self.alpha)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Display the plot
            plt.show()

    class Quiver3D(Base3DPlot):
        def __init__(self, x, y, z, u, v, w, title=None, xlabel=None, ylabel=None, zlabel=None,
                     length=1.0, color='orange', arrow_length_ratio=0.05):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.x = x
            self.y = y
            self.z = z
            self.u = u
            self.v = v
            self.w = w
            self.length = length
            self.color = color
            self.arrow_length_ratio = arrow_length_ratio

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D quiver plot
            ax.quiver(self.x, self.y, self.z, self.u, self.v, self.w, length=self.length,
                      color=self.color, arrow_length_ratio=self.arrow_length_ratio)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Display the plot
            plt.show()

    class Contour3D(Base3DPlot):
        def __init__(self, x, y, z, title=None, xlabel=None, ylabel=None, zlabel=None,
                     cmap='viridis', levels=10, alpha=0.7):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.x = x
            self.y = y
            self.z = z
            self.cmap = cmap
            self.levels = levels
            self.alpha = alpha

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D contour plot
            contour = ax.contour3D(self.x, self.y, self.z, cmap=self.cmap, levels=self.levels, alpha=self.alpha)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Add colorbar for better interpretation
            fig.colorbar(contour, ax=ax, pad=0.1)

            # Display the plot
            plt.show()

    class Heatmap2D(BasePlot):
        def __init__(self, data, title=None, xlabel=None, ylabel=None,
                     cmap='viridis', alpha=0.7):
            super().__init__(title, xlabel, ylabel)
            self.data = data
            self.cmap = cmap
            self.alpha = alpha

        def show(self):
            fig, ax = plt.subplots()

            # Create 2D heatmap plot
            heatmap = ax.imshow(self.data, cmap=self.cmap, alpha=self.alpha)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Add colorbar for better interpretation
            fig.colorbar(heatmap, ax=ax, pad=0.1)

            # Display the plot
            plt.show()

    class AnimatedScatter3D(Scatter3D):
        def animate(self, frames=100, interval=100, repeat=True):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            def update(frame):
                ax.clear()
                ax.scatter(self.x[:frame], self.y[:frame], self.z[:frame], c=self.color,
                           marker=self.marker, s=self.size)

                self.set_labels(ax)
                self.set_title(ax)
                self.set_zlabel(ax)

            ani = animation.FuncAnimation(fig, update, frames=len(self.x), interval=interval, repeat=repeat)

            plt.show()

    class Boxplot2D(BasePlot):
        def __init__(self, data, title=None, xlabel=None, ylabel=None,
                     color='blue', notch=True, sym='gD', vert=True):
            super().__init__(title, xlabel, ylabel)
            self.data = data
            self.color = color
            self.notch = notch
            self.sym = sym
            self.vert = vert

        def show(self):
            fig, ax = plt.subplots()

            # Create 2D box plot
            boxplot = ax.boxplot(self.data, patch_artist=True, notch=self.notch, sym=self.sym, vert=self.vert)

            # Customize boxplot colors
            for patch in boxplot['boxes']:
                patch.set_facecolor(self.color)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Display the plot
            plt.show()

    class PolarPlot(BasePlot):
        def __init__(self, theta, r, title=None, xlabel=None, ylabel=None,
                     color='green', marker='o', linestyle='-', linewidth=2):
            super().__init__(title, xlabel, ylabel)
            self.theta = theta
            self.r = r
            self.color = color
            self.marker = marker
            self.linestyle = linestyle
            self.linewidth = linewidth

        def show(self):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

            # Create polar plot
            ax.plot(np.radians(self.theta), self.r, color=self.color, marker=self.marker,
                    linestyle=self.linestyle, linewidth=self.linewidth)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Display the plot
            plt.show()



    class Hexbin2D(BasePlot):
        def __init__(self, x, y, gridsize=30, cmap='viridis', mincnt=1, edgecolors='none'):
            super().__init__()
            self.x = x
            self.y = y
            self.gridsize = gridsize
            self.cmap = cmap
            self.mincnt = mincnt
            self.edgecolors = edgecolors

        def show(self):
            fig, ax = plt.subplots()

            # Create 2D hexbin plot
            hb = ax.hexbin(self.x, self.y, gridsize=self.gridsize, cmap=self.cmap, mincnt=self.mincnt, edgecolors=self.edgecolors)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Add colorbar for better interpretation
            fig.colorbar(hb, ax=ax, pad=0.1)

            # Display the plot
            plt.show()

    class TriangularMesh3D(Base3DPlot):
        def __init__(self, x, y, z, triangles, title=None, xlabel=None, ylabel=None, zlabel=None,
                     cmap='viridis', alpha=1.0):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.x = x
            self.y = y
            self.z = z
            self.triangles = triangles
            self.cmap = cmap
            self.alpha = alpha

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D triangular mesh plot
            mesh = ax.plot_trisurf(self.x, self.y, self.z, triangles=self.triangles, cmap=self.cmap, alpha=self.alpha)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Add colorbar for better interpretation
            fig.colorbar(mesh, ax=ax, pad=0.1)

            # Display the plot
            plt.show()


    class Streamline3D(Base3DPlot):
        def __init__(self, x, y, z, u, v, w, title=None, xlabel=None, ylabel=None, zlabel=None,
                     color='blue', linewidth=2, density=1):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.x = x
            self.y = y
            self.z = z
            self.u = u
            self.v = v
            self.w = w
            self.color = color
            self.linewidth = linewidth
            self.density = density

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D streamline plot
            ax.streamplot(self.x, self.y, self.z, self.u, self.v, self.w,
                          color=self.color, linewidth=self.linewidth, density=self.density)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Display the plot
            plt.show()

    class ColorSizeScatter2D(BasePlot):
        def __init__(self, x, y, colors, sizes, title=None, xlabel=None, ylabel=None, colormap='viridis', alpha=0.7):
            super().__init__(title, xlabel, ylabel)
            self.x = x
            self.y = y
            self.colors = colors
            self.sizes = sizes
            self.colormap = colormap
            self.alpha = alpha

        def show(self):
            fig, ax = plt.subplots()

            # Create 2D scatter plot with color and size coding
            scatter = ax.scatter(self.x, self.y, c=self.colors, s=self.sizes,
                                 cmap=self.colormap, alpha=self.alpha)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Add colorbar for better interpretation
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Color Legend')

            # Display the plot
            plt.show()


    class GroupedBoxplot2D(BasePlot):
        def __init__(self, data, categories, title=None, xlabel=None, ylabel=None, color='skyblue', notch=True, sym='gD', vert=True):
            super().__init__(title, xlabel, ylabel)
            self.data = data
            self.categories = categories
            self.color = color
            self.notch = notch
            self.sym = sym
            self.vert = vert

        def show(self):
            fig, ax = plt.subplots()

            # Create grouped boxplot
            boxplot = ax.boxplot(self.data, labels=self.categories, patch_artist=True, notch=self.notch, sym=self.sym, vert=self.vert)

            # Customize boxplot colors
            for patch in boxplot['boxes']:
                patch.set_facecolor(self.color)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Display the plot
            plt.show()

    class PairwiseScatterplotMatrix(BasePlot):
        def __init__(self, data, title=None, hue=None, palette='viridis'):
            super().__init__(title)
            self.data = data
            self.hue = hue
            self.palette = palette

        def show(self):
            # Create pairwise scatterplot matrix
            sns.pairplot(self.data, hue=self.hue, palette=self.palette)

            # Display the plot
            plt.show()

    class CorrelationHeatmap(BasePlot):
        def __init__(self, data, title=None, cmap='coolwarm'):
            super().__init__(title)
            self.data = data
            self.cmap = cmap

        def show(self):
            fig, ax = plt.subplots()

            # Create correlation heatmap
            sns.heatmap(self.data.corr(), annot=True, cmap=self.cmap)

            # Set title
            self.set_title(ax)

            # Display the plot
            plt.show()

    class HistogramWithKDE(BasePlot):
        def __init__(self, data, title=None, xlabel=None, ylabel=None, color='green', bins=30):
            super().__init__(title, xlabel, ylabel)
            self.data = data
            self.color = color
            self.bins = bins

        def show(self):
            fig, ax = plt.subplots()

            # Create histogram with KDE
            sns.histplot(self.data, kde=True, color=self.color, bins=self.bins)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Display the plot
            plt.show()

    class ViolinPlot(BasePlot):
        def __init__(self, x, y, data, title=None, xlabel=None, ylabel=None, palette='muted'):
            super().__init__(title, xlabel, ylabel)
            self.x = x
            self.y = y
            self.data = data
            self.palette = palette

        def show(self):
            fig, ax = plt.subplots()

            # Create violin plot
            sns.violinplot(x=self.x, y=self.y, data=self.data, palette=self.palette)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Display the plot
            plt.show()

    class ParametricCurve(BasePlot):
        def __init__(self, x_func, y_func, t_range, title=None, xlabel=None, ylabel=None, color='blue', linewidth=2):
            super().__init__(title, xlabel, ylabel)
            self.x_func = x_func
            self.y_func = y_func
            self.t_range = t_range
            self.color = color
            self.linewidth = linewidth

        def show(self):
            fig, ax = plt.subplots()

            # Create parametric curve plot
            t_values = np.linspace(*self.t_range, 1000)
            x_values = self.x_func(t_values)
            y_values = self.y_func(t_values)
            ax.plot(x_values, y_values, color=self.color, linewidth=self.linewidth)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Display the plot
            plt.show()

    class PolarRose(BasePlot):
        def __init__(self, n_petals, title=None, color='red', linewidth=2):
            super().__init__(title)
            self.n_petals = n_petals
            self.color = color
            self.linewidth = linewidth

        def show(self):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

            # Create polar rose plot
            theta = np.linspace(0, 2 * np.pi, 1000)
            r = np.cos(self.n_petals * theta)
            ax.plot(theta, r, color=self.color, linewidth=self.linewidth)

            # Set title
            self.set_title(ax)

            # Display the plot
            plt.show()

    class ParametricSurface3D(Base3DPlot):
        def __init__(self, x_func, y_func, z_func, t_range, title=None, xlabel=None, ylabel=None, zlabel=None, cmap='viridis', alpha=1.0):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.x_func = x_func
            self.y_func = y_func
            self.z_func = z_func
            self.t_range = t_range
            self.cmap = cmap
            self.alpha = alpha

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D parametric surface plot
            t_values = np.linspace(*self.t_range, 100)
            x_values = self.x_func(t_values)
            y_values = self.y_func(t_values)
            z_values = self.z_func(t_values)
            ax.plot_surface(x_values, y_values, z_values, cmap=self.cmap, alpha=self.alpha)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Add colorbar for better interpretation
            fig.colorbar(ax.plot_surface(x_values, y_values, z_values, cmap=self.cmap), ax=ax, pad=0.1)

            # Display the plot
            plt.show()

    class SaddleSurface(Base3DPlot):
        def __init__(self, title=None, xlabel=None, ylabel=None, zlabel=None, cmap='coolwarm', alpha=1.0):
            super().__init__(title, xlabel, ylabel, zlabel)
            self.cmap = cmap
            self.alpha = alpha

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create 3D saddle surface plot
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            x, y = np.meshgrid(x, y)
            z = x**2 - y**2
            ax.plot_surface(x, y, z, cmap=self.cmap, alpha=self.alpha)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)
            self.set_zlabel(ax)

            # Add colorbar for better interpretation
            fig.colorbar(ax.plot_surface(x, y, z, cmap=self.cmap), ax=ax, pad=0.1)

            # Display the plot
            plt.show()

    class QuaternionRotation(Base3DPlot):
        def __init__(self, quaternion, title=None):
            super().__init__(title)
            self.quaternion = quaternion

        def show(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Define initial and rotated vectors
            initial_vector = np.array([1, 0, 0])
            rotated_vector = self.quaternion.rotate(initial_vector)

            # Create a polygon representing the vectors
            vectors = np.array([[0, 0, 0], initial_vector, rotated_vector])
            ax.add_collection3d(Poly3DCollection([vectors], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))

            # Set labels and title
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title(self.title)

            # Display the plot
            plt.show()

    class TimeSeriesRollingAverage(BasePlot):
        def __init__(self, time, values, window_size, title=None, xlabel=None, ylabel=None, color='blue'):
            super().__init__(title, xlabel, ylabel)
            self.time = time
            self.values = values
            self.window_size = window_size
            self.color = color

        def show(self):
            fig, ax = plt.subplots()

            # Create time series plot with rolling average
            ax.plot(self.time, self.values, label='Original', color=self.color, alpha=0.7)
            ax.plot(self.time, self.values.rolling(window=self.window_size).mean(), label=f'Rolling Average ({self.window_size} periods)', color='red')

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Add legend for better interpretation
            ax.legend()

            # Display the plot
            plt.show()

    class HistogramOutlierDetection(BasePlot):
        def __init__(self, data, title=None, xlabel=None, ylabel=None, bins=30, color='green', outlier_color='red', threshold=2):
            super().__init__(title, xlabel, ylabel)
            self.data = data
            self.bins = bins
            self.color = color
            self.outlier_color = outlier_color
            self.threshold = threshold

        def show(self):
            fig, ax = plt.subplots()

            # Create histogram with outlier detection
            counts, edges, _ = ax.hist(self.data, bins=self.bins, color=self.color, alpha=0.7)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            outliers = bin_centers[counts > np.mean(counts) + self.threshold * np.std(counts)]

            # Highlight outliers
            for outlier in outliers:
                ax.axvline(outlier, color=self.outlier_color, linestyle='dashed', linewidth=2)

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Display the plot
            plt.show()

    class ScatterRegressionPlot(BasePlot):
        def __init__(self, x, y, title=None, xlabel=None, ylabel=None, color='blue'):
            super().__init__(title, xlabel, ylabel)
            self.x = x
            self.y = y
            self.color = color

        def show(self):
            fig, ax = plt.subplots()

            # Create scatter plot with regression line
            ax.scatter(self.x, self.y, color=self.color, alpha=0.7)
            model = LinearRegression().fit(self.x.values.reshape(-1, 1), self.y)
            ax.plot(self.x, model.predict(self.x.values.reshape(-1, 1)), color='red', label='Regression Line')

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Add legend for better interpretation
            ax.legend()

            # Display the plot
            plt.show()

    class StackedAreaTimeSeries(BasePlot):
        def __init__(self, time, values_list, labels, title=None, xlabel=None, ylabel=None, colormap='viridis'):
            super().__init__(title, xlabel, ylabel)
            self.time = time
            self.values_list = values_list
            self.labels = labels
            self.colormap = plt.get_cmap(colormap)

        def show(self):
            fig, ax = plt.subplots()

            # Create stacked area plot for time series
            ax.stackplot(self.time, self.values_list, labels=self.labels, colors=[self.colormap(i) for i in range(len(self.values_list))])

            # Set labels and title
            self.set_labels(ax)
            self.set_title(ax)

            # Add legend for better interpretation
            ax.legend()

            # Display the plot
            plt.show()

    class ConfusionMatrixHeatmap(BasePlot):
        def __init__(self, true_labels, predicted_labels, title=None, cmap='Blues'):
            super().__init__(title)
            self.true_labels = true_labels
            self.predicted_labels = predicted_labels
            self.cmap = cmap

        def show(self):
            fig, ax = plt.subplots()

            # Calculate confusion matrix
            cm = confusion_matrix(self.true_labels, self.predicted_labels)

            # Create confusion matrix heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap=self.cmap, cbar=False)

            # Set title
            self.set_title(ax)

            # Display the plot
            plt.show()
