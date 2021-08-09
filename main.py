import wx
import numpy as np
import itertools
import os
import toml

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import matplotlib.animation as mplanim

import plot
import vlasov


def gaussian(x, x0, w):
    A = 1 / (2 * np.sqrt(2 * np.pi) * w)
    return A * np.exp(-((x - x0) ** 2) / (2 * w ** 2))


def uniform(x, lowb, upb):
    dx = upb - lowb
    return ((lowb <= x) & (x < upb)) / (2 * dx) + ((lowb < x) & (x <= upb)) / (2 * dx)


def velocity_distribution(species, type, vv):
    if type == "gaussian":
        ni = species["number_density"]
        v0 = species["drift_velocity"]
        w = species["standard_derivation"]
        return ni * gaussian(vv, v0, w)
    if type == "uniform":
        ni = species["number_density"]
        lowb = species["min_velocity"]
        upb = species["max_velocity"]
        return ni * uniform(vv, lowb, upb)
    raise


def load_vp2d_config(toml):
    general = toml["general"]
    xmax = general["system_length"]
    vmax = general["max_velocity"]
    nx = general["nx"]
    nv = general["nv"]
    x = np.linspace(0, xmax, nx, endpoint=False)
    v = np.linspace(-vmax, vmax, nv, endpoint=False)
    xx, vv = np.meshgrid(x, v, sparse=True)
    background_charge_density = general["background_charge_density"]
    dt = general["termination_time"] / general["nt"]

    species = []
    f_init = []
    for s in toml["species"]:
        name = s["name"]
        q = s["charge"]
        qm = s["charge_to_mass_ratio"]
        species.append(vlasov.Species(name, q, qm))

        type = s.get("distribution", "gaussian")
        amp = s["am_amplitude"]
        k = s["am_wavenumber"]
        fs = velocity_distribution(s, type, vv) * (1 + amp * np.cos(2 * np.pi * k * xx))
        f_init.append(fs)

    return vlasov.Vp2dConfig(
        species, np.array(f_init), background_charge_density, xmax, nx, vmax, nv, dt
    )


def create_charge_density_plot(ax, x):
    ax.set_title("charge density")
    ax.set_xlabel("x")
    ax.grid(True)
    return plot.LinePlot(ax, x)


def create_electric_field_plot(ax, x):
    ax.set_title("electric field Ex")
    ax.set_xlabel("x")
    ax.grid(True)
    return plot.LinePlot(ax, x)


def create_velocity_distribution_plot(ax, v, species, dx):
    ax.set_title("velocity distribution")
    ax.set_xlabel("v")
    ax.grid(True)
    return plot.VelocityDistPlot(ax, v, species, dx)


def create_time_series_plot(ax, tmax, nt, labels):
    ax.set_xlabel("time")
    ax.grid(True)
    return plot.TimeSeriesPlot(ax, tmax, nt, labels)


def plot_distribution_function(plot, show, f, rho, E):
    plot.plot(f, show=show)


def plot_charge_density(plot, show, f, rho, E):
    plot.plot(rho, show=show)


def plot_electric_field(plot, show, f, rho, E):
    plot.plot(E, show=show)


def plot_time_series_energy(vp2d: vlasov.Vp2dConfig):
    m = np.array([species.q / species.qm for species in vp2d.species])
    v = np.linspace(-vp2d.vmax, vp2d.vmax, vp2d.ngridv, endpoint=False)
    nspecies = len(vp2d.species)
    dx = vp2d.system_length / vp2d.ngridx
    dv = 2 * vp2d.vmax / vp2d.ngridv
    eps0 = 1.0

    def func(plot, show, f, rho, E):
        EE = eps0 / 2 * (E ** 2).sum() * dx

        KE = 0
        for s in range(nspecies):
            KE += (f[s].sum(axis=1) * (v ** 2)).sum() * m[s] / 2 * dv * dx

        Etot = EE + KE
        plot.plot([KE, EE, Etot], show=show)

    return func


def plot_time_series_courant(vp2d: vlasov.Vp2dConfig):
    dt = vp2d.dt
    vmax = vp2d.vmax
    qm = np.array([species.qm for species in vp2d.species])
    dx = vp2d.system_length / vp2d.ngridx
    dv = 2 * vp2d.vmax / vp2d.ngridv
    nspecies = len(vp2d.species)

    def func(plot, show, f, rho, E):
        k = 0
        for s in range(nspecies):
            k = max(k, np.max(np.sqrt((vmax / dx) ** 2 + (qm[s] * E / dv) ** 2)))
        plot.plot([k * dt], show=show)

    return func


def load_subplot_config(figure, config, vp2d, init):
    view = config["view"]
    n = len(view["subplot"])
    nrows = view["nrows"]
    ncols = view["ncols"]
    xmax = vp2d.system_length
    vmax = vp2d.vmax
    tmax = config["general"]["termination_time"]
    nt = config["general"]["nt"]
    x = np.linspace(0, xmax, vp2d.ngridx, endpoint=False)
    v = np.linspace(-vp2d.vmax, vp2d.vmax, vp2d.ngridv, endpoint=False)
    (f_init, rho, E) = init

    plots = []
    for i in range(n):
        ax = figure.add_subplot(nrows, ncols, i + 1)
        subplot = view["subplot"][i]
        type = subplot["type"]
        if type == "distribution function":
            ax.set_xlabel("x")
            ax.set_ylabel("v")
            p = plot.TotalDistFuncPlot(figure, ax)
            f = vp2d.initial_distribution
            p.init_axes(f, 0, xmax, -vmax, vmax)
            plots.append((p, plot_distribution_function))
        elif type == "charge density":
            ax.set_xlim(0, xmax)
            p = create_charge_density_plot(ax, x)
            p.init_axes(rho)
            plots.append((p, plot_charge_density))
        elif type == "electric field":
            ax.set_xlim(0, xmax)
            p = create_electric_field_plot(ax, x)
            p.init_axes(E)
            plots.append((p, plot_electric_field))
        elif type == "velocity distribution":
            dx = xmax / vp2d.ngridx
            ax.set_xlim(-vmax, vmax)
            p = create_velocity_distribution_plot(ax, v, vp2d.species, dx)
            p.init_axes(f_init)
            plots.append((p, plot_distribution_function))
        elif type == "Ex dispersion relation":
            dx = xmax / vp2d.ngridx
            dt = tmax / nt
            klim = subplot.get("max_wavenumber", 1 / (2 * dx))
            wlim = subplot.get("max_frequency", 1 / (2 * dt))
            ax.set_title("electric field")
            p = plot.DispersionRelationPlot(figure, ax, vp2d.ngridx, nt)
            p.init_axes(E, dx, dt, klim, wlim)
            plots.append((p, plot_electric_field))
        elif type == "energy":
            ax.set_xlim([0, tmax])
            ax.set_yscale("log")
            labels = ["KE", "EE", "total"]
            p = create_time_series_plot(ax, tmax, nt, labels)
            p.init_axes()
            plots.append((p, plot_time_series_energy(vp2d)))
        elif type == "courant number":
            ax.set_title("Courant number")
            ax.set_xlim([0, tmax])
            ax.set_yscale("log")
            ax.grid(True, which="both", axis="y")
            ax.axhline(0.1, color="0.1")
            ax.axhline(1, color="0.1")
            p = create_time_series_plot(ax, tmax, nt, ["C"])
            p.init_axes()
            plots.append((p, plot_time_series_courant(vp2d)))

    return plots


def call_set_data(gen, subplots):
    for val in gen:
        (i, (f, rho, E)) = val
        for (plot, func) in subplots:
            func(plot, False, f, rho, E)
        yield val


class PlotPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        self.figure = None
        self.animation = None
        self.is_running = False
        self.subplots = None
        self.dt = None

    def init_figure(self, config):
        self.figure = mpl.figure.Figure(figsize=(2, 2))
        canvas = FigureCanvasWxAgg(self, -1, self.figure)

        vp2d = load_vp2d_config(config)

        self.figure.clf()
        self.figure.subplots_adjust(hspace=0.5, wspace=0.3)
        problem = vlasov.vp2d(vp2d)
        init = next(problem)
        self.subplots = load_subplot_config(self.figure, config, vp2d, init)

        nt = config["general"]["nt"]
        self.dt = vp2d.dt
        values = zip(range(1, nt + 1), problem)
        tick = config["view"]["tick"]
        frames = itertools.islice(
            call_set_data(values, self.subplots), tick - 1, None, tick
        )
        self.animation = mplanim.FuncAnimation(
            self.figure, self.plot, frames=frames, interval=50, repeat=False
        )
        self.is_running = True

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def plot(self, val):
        (i, (f, rho, E)) = val
        time = i * self.dt
        self.figure.suptitle(f"T = {time:.3g}")

        for (plot, draw) in self.subplots:
            draw(plot, True, f, rho, E)

    def close_animation(self):
        self.animation.event_source.stop()

    def pause_resume(self):
        if self.is_running:
            self.animation.pause()
        else:
            self.animation.resume()
        self.is_running = not self.is_running


class PlotFrame(wx.Frame):
    def __init__(self, parent=None):
        super().__init__(None, title="plot", size=(900, 600))
        self.panel = PlotPanel(self)
        self.Bind(wx.EVT_CLOSE, self.onQuit)
        self.Bind(wx.EVT_CHAR_HOOK, self.onCharHook)

    def init_figure(self, config):
        self.panel.init_figure(config)

    def onQuit(self, event):
        self.panel.close_animation()
        self.Destroy()

    def onCharHook(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_SPACE:
            self.panel.pause_resume()
        event.Skip()


class MainFrame(wx.Frame):
    def __init__(self, parent=None):
        super().__init__(None, title="EmVSC")
        panel = wx.Panel(self)
        self.config = None

        loadBtn = wx.Button(panel, label="Load File")
        self.Bind(wx.EVT_BUTTON, self.onLoadBtn, loadBtn)

        self.runBtn = wx.Button(panel, label="Run")
        self.runBtn.Disable()
        self.Bind(wx.EVT_BUTTON, self.onRunBtn, self.runBtn)

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(mainSizer)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        mainSizer.Add(buttons)
        buttons.Add(loadBtn)
        buttons.Add(self.runBtn)

        self.configText = wx.StaticText(panel, wx.ID_ANY)
        mainSizer.Add(self.configText)

    def onLoadBtn(self, event):
        wildcard = "TOML files (*.toml)|*.toml"
        dialog = wx.FileDialog(
            self,
            "Open configuration files",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if dialog.ShowModal() == wx.ID_CANCEL:
            dialog.Destroy()
            return
        path = dialog.GetPath()
        if os.path.exists(path):
            t = toml.load(path)
            self.configText.SetLabel(toml.dumps(t))
            self.config = t
        dialog.Destroy()
        self.runBtn.Enable()

    def onRunBtn(self, _):
        if self.config is None:
            return
        plotFrame = PlotFrame(self)
        plotFrame.init_figure(self.config)
        plotFrame.Centre()
        plotFrame.Show()


if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    frame.Show()
    app.MainLoop()
