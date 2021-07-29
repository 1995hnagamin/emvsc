import numpy as np


class DistFuncPlot:
    def __init__(self, figure, axes):
        self.figure = figure
        self.axes = axes
        self.im = None

    def init_axes(self, fs, xmin, xmax, vmin, vmax):
        self.axes.set_title("distribution function")
        extent = [xmin, xmax, vmin, vmax]
        self.im = self.axes.imshow(fs, cmap="plasma", extent=extent, origin="lower")
        self.figure.colorbar(self.im, ax=self.axes)

    def plot(self, fs):
        self.im.set_data(fs)
        self.im.set_clim(vmin=np.min(fs), vmax=np.max(fs))
