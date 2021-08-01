import numpy as np
from matplotlib.colors import LogNorm


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


class LinePlot:
    def __init__(self, axes, xvalue):
        self.axes = axes
        self.x = xvalue

    def init_axes(self, g):
        self.axes.plot(self.x, g, color="black")

    def plot(self, g):
        for line in self.axes.get_lines():
            line.remove()
        self.axes.plot(self.x, g, color="black")


class VerocityDistPlot:
    def __init__(self, axes, vvalue, species):
        self.axes = axes
        self.v = vvalue
        self.species = species

    def init_axes(self, f):
        f_total = f.sum(axis=0)
        self.axes.plot(self.v, f_total.sum(axis=1), color="black", label="total")
        self.axes.set_prop_cycle(None)
        for s, species in enumerate(self.species):
            g = f[s].sum(axis=1)
            self.axes.plot(self.v, g, label=species.name, linewidth=0.3)
        self.axes.legend()

    def plot(self, f):
        f_total = f.sum(axis=0)
        for line in self.axes.get_lines():
            line.remove()
        self.axes.plot(self.v, f_total.sum(axis=1), color="black", label="total")
        self.axes.set_prop_cycle(None)
        for s, species in enumerate(self.species):
            g = f[s].sum(axis=1)
            self.axes.plot(self.v, g, label=species.name, linewidth=0.3)
        self.axes.legend()


class DispersionRelationPlot:
    def __init__(self, figure, axes, nx, nt):
        self.figure = figure
        self.axes = axes
        self.values = np.zeros((nt, nx))
        self.count = 0
        self.limit = nt
        self.extent = None

    def push_back(self, g):
        if self.count >= self.limit:
            return
        self.values[self.count, :] = g
        self.count += 1

    def init_axes(self, g, kmin, kmax, wmin, wmax):
        self.extent = [kmin, kmax, wmin, wmax]
        self.push_back(g)
        self.im = self.axes.imshow(self.values, cmap="rainbow", origin="lower")

    def plot(self, g):
        if self.count >= self.limit:
            return
        self.push_back(g)
        if self.count < self.limit:
            self.im.set_data(self.values)
        else:
            spec = np.absolute(np.fft.fftshift(np.fft.fft2(self.values)))
            self.im = self.axes.imshow(
                spec,
                cmap="rainbow",
                extent=self.extent,
                origin="lower",
                norm=LogNorm(vmin=np.min(spec), vmax=np.max(spec)),
            )
            self.figure.colorbar(self.im, ax=self.axes)
