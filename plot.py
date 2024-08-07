import numpy as np
from matplotlib.colors import LogNorm


class TotalDistFuncPlot:
    def __init__(self, figure, axes):
        self.figure = figure
        self.axes = axes
        self.im = None

    def init_axes(self, f, xmin, xmax, vmin, vmax):
        ftot = f.sum(axis=0)
        self.axes.set_title("distribution function")
        extent = [xmin, xmax, vmin, vmax]
        self.im = self.axes.imshow(ftot, cmap="plasma", extent=extent, origin="lower")
        self.figure.colorbar(self.im, ax=self.axes)

    def plot(self, f, *, show=True):
        if not show:
            return
        ftot = f.sum(axis=0)
        self.im.set_data(ftot)
        self.im.set_clim(vmin=np.min(ftot), vmax=np.max(ftot))


class LinePlot:
    def __init__(self, axes, xvalue):
        self.axes = axes
        self.x = xvalue

    def init_axes(self, g):
        self.axes.plot(self.x, g, color="black")

    def plot(self, g, *, show=True):
        if not show:
            return
        for line in self.axes.get_lines():
            line.remove()
        self.axes.plot(self.x, g, color="black")


class TimeSeriesPlot:
    def __init__(self, axes, tmax, nt, labels):
        self.axes = axes
        self.limit = nt
        self.labels = labels
        self.nlines = len(labels)
        self.t = np.linspace(0, tmax, nt)
        self.values = np.empty((nt, self.nlines))
        self.values[:, :] = np.nan
        self.count = 0

    def init_axes(self):
        return

    def push_back(self, v):
        if self.count >= self.limit:
            return
        self.values[self.count, :] = v
        self.count += 1

    def plot(self, v, *, show=True):
        if not show:
            self.push_back(v)
            return
        for line in self.axes.get_lines():
            line.remove()
        self.axes.set_prop_cycle(None)
        for i in range(self.nlines):
            self.axes.plot(self.t, self.values[:, i], label=self.labels[i])
        if self.nlines > 1:
            self.axes.legend()


class VelocityDistPlot:
    def __init__(self, axes, vvalue, species, dx):
        self.axes = axes
        self.v = vvalue
        self.species = species
        self.dx = dx

    def init_axes(self, f):
        self.axes.set_prop_cycle(None)
        for s, species in enumerate(self.species):
            g = f[s].sum(axis=1) * self.dx
            self.axes.plot(self.v, g, label=species.name, linewidth=0.3)
        if len(self.species) > 1:
            f_total = f.sum(axis=0)
            self.axes.plot(
                self.v,
                f_total.sum(axis=1) * self.dx,
                color="black",
                label="total",
            )
            self.axes.legend()

    def plot(self, f, *, show=True):
        if not show:
            return
        for line in self.axes.get_lines():
            line.remove()
        self.axes.set_prop_cycle(None)
        for s, species in enumerate(self.species):
            g = f[s].sum(axis=1) * self.dx
            self.axes.plot(self.v, g, label=species.name, linewidth=0.3)
        if len(self.species) > 1:
            f_total = f.sum(axis=0)
            self.axes.plot(
                self.v,
                f_total.sum(axis=1) * self.dx,
                color="black",
                label="total",
            )
            self.axes.legend()


class DispersionRelationPlot:
    colormap = "jet"

    def __init__(self, figure, axes, nx, nt):
        self.figure = figure
        self.axes = axes
        self.values = np.zeros((nt, nx))
        self.count = 0
        self.limit = nt
        self.extent = None
        self.klim = None
        self.wlim = None

    def push_back(self, g):
        if self.count >= self.limit:
            return
        self.values[self.count, :] = g
        self.count += 1

    def init_axes(self, g, dx, dt, klim, wlim):
        kmax = 1 / (2 * dx)
        wmax = 1 / (2 * dt)
        self.extent = [-kmax, kmax, -wmax, wmax]
        self.klim = klim
        self.wlim = wlim
        self.push_back(g)
        height, width = self.values.shape
        self.im = self.axes.imshow(
            self.values,
            cmap=self.colormap,
            origin="lower",
            aspect=width / height,
        )

    def plot(self, g, *, show=True):
        if not show:
            self.push_back(g)
            return
        self.im.set_data(self.values)
        if self.count >= self.limit:
            spec = np.absolute(np.fft.fftshift(np.fft.fft2(self.values)))
            # set the upper bound (P95%) and lower bound (P5%) of the colorbar
            upb = np.percentile(spec, 95)
            lwb = np.percentile(spec, 5)
            self.im = self.axes.imshow(
                spec,
                cmap=self.colormap,
                extent=self.extent,
                origin="lower",
                aspect=(2 * self.klim / self.wlim),
                norm=LogNorm(vmin=lwb, vmax=upb),
            )
            self.axes.set_xlabel("k")
            self.axes.set_ylabel("ω")
            self.axes.set_xlim([-self.klim, self.klim])
            self.axes.set_ylim([0, self.wlim])
            self.figure.colorbar(self.im, ax=self.axes)
