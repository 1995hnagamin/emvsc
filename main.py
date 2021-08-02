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
    dt = general["time_step"]

    species = []
    f_init = []
    for s in toml["species"]:
        name = s["name"]
        q = s["charge"]
        qm = s["charge_to_mass_ratio"]
        species.append(vlasov.Species(name, q, qm))

        ni = s["number_density"]
        v0 = s["drift_velocity"]
        amp = s["am_amplitude"]
        k = s["am_wavenumber"]
        w = s["standard_derivation"]
        fs = ni * gaussian(vv, v0, w) * (1 + amp * np.cos(2 * np.pi * k * xx))
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


def create_velocity_distribution_plot(ax, v, species):
    ax.set_title("velocity distribution")
    ax.set_xlabel("v")
    ax.grid(True)
    return plot.VerocityDistPlot(ax, v, species)


def plot_total_distribution_function(plot, show, f, rho, E):
    if show:
        f_total = f.sum(axis=0)
        plot.plot(f_total)


def plot_charge_density(plot, show, f, rho, E):
    if show:
        plot.plot(rho)
    else:
        plot.set_data(rho)


def plot_electric_field(plot, show, f, rho, E):
    if show:
        plot.plot(E)
    else:
        plot.set_data(E)


def plot_velocity_distribution(plot, show, f, rho, E):
    if show:
        plot.plot(f)
    else:
        plot.set_data(f)


def load_subplot_config(figure, view, vp2d):
    n = len(view["subplot"])
    nrows = view["nrows"]
    ncols = view["ncols"]
    xmax = vp2d.system_length
    vmax = vp2d.vmax
    x = np.linspace(0, xmax, vp2d.ngridx, endpoint=False)
    v = np.linspace(-vp2d.vmax, vp2d.vmax, vp2d.ngridv, endpoint=False)

    plots = []
    for i in range(n):
        ax = figure.add_subplot(nrows, ncols, i + 1)
        subplot = view["subplot"][i]
        type = subplot["type"]
        if type == "distribution function":
            p = plot.DistFuncPlot(figure, ax)
            f = vp2d.initial_distribution.sum(axis=0)
            p.init_axes(f, 0, xmax, -vmax, vmax)
            plots.append((p, plot_total_distribution_function))
        elif type == "charge density":
            ax.set_xlim(0, xmax)
            p = create_charge_density_plot(ax, x)
            plots.append((p, plot_charge_density))
        elif type == "electric field":
            ax.set_xlim(0, xmax)
            p = create_electric_field_plot(ax, x)
            plots.append((p, plot_electric_field))
        elif type == "velocity distribution":
            ax.set_xlim(-vmax, vmax)
            p = create_velocity_distribution_plot(ax, v, vp2d.species)
            plots.append((p, plot_velocity_distribution))

    return plots


class PlotPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        self.figure = None
        self.animation = None
        self.is_running = False
        self.subplots = None
        self.ndt = None
        self.tick = None

    def init_figure(self, config):
        self.figure = mpl.figure.Figure(figsize=(2, 2))
        canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.animation = mplanim.FuncAnimation(self.figure, self.plot, interval=50)
        self.is_running = True

        vp2d = load_vp2d_config(config)
        self.gen = vlasov.vp2d(vp2d)
        self.tick = config["view"]["tick"]
        self.ndt = self.tick * vp2d.dt

        self.figure.clf()
        self.figure.subplots_adjust(hspace=0.5, wspace=0.3)
        self.subplots = load_subplot_config(self.figure, config["view"], vp2d)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def set_data(self):
        (f, rho, E) = next(self.gen)

        for (plot, func) in self.subplots:
            func(plot, False, f, rho, E)

    def plot(self, i):
        for _ in range(self.tick - 1):
            self.set_data()

        (f, rho, E) = next(self.gen)
        time = i * self.ndt
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
