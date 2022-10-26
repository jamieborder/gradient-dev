import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

pts = np.random.rand(10,2)

vor = Voronoi(pts)

fig = voronoi_plot_2d(vor)
plt.show(block=False)
