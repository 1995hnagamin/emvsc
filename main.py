import wx
import numpy as np
import itertools
import os
import toml

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import matplotlib.animation as mplanim

import vlasov


def gaussian(x, x0, w):
    A = 1 / (2 * np.sqrt(2 * np.pi) * w)
    return A * np.exp(-((x - x0) ** 2) / (2 * w ** 2))


def load_toml_file(toml):
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

    species = vlasov.Species()
    f_init = []
    for s in toml["species"]:
        name = s["name"]
        q = s["charge"]
        qm = s["charge_to_mass_ratio"]
        species.append(name, q, qm)

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


class Plot(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.is_running = False
        self.Bind(wx.EVT_KEY_DOWN, self.onKeyDown)

        self.figure = None
        self.animation = None
        self.axF = None
        self.axR = None
        self.axE = None
        self.axV = None
        self.x = None
        self.v = None
        self.im = None
        self.ndt = None
        self.config = None

    def init_figure(self, config: vlasov.Vp2dConfig):
        self.figure = mpl.figure.Figure(figsize=(2, 2))
        canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.animation = mplanim.FuncAnimation(self.figure, self.plot, interval=50)

        self.figure.clf()
        self.figure.subplots_adjust(hspace=0.5, wspace=0.3)
        self.axF = self.figure.add_subplot(221)
        self.axF.set_title("distibution function")

        self.axR = self.figure.add_subplot(222)
        self.axR.set_title("charge density")
        self.axR.set_xlabel("x")
        self.axR.set_xlim(0, config.system_length)
        self.axR.grid(True)

        self.axE = self.figure.add_subplot(223)
        self.axE.set_title("electric field Ex")
        self.axE.set_xlabel("x")
        self.axE.grid(True)
        self.axE.set_xlim(0, config.system_length)

        self.axV = self.figure.add_subplot(224)
        self.axV.set_title("velocity distribution")
        self.axV.set_xlabel("v")
        self.axV.grid(True)
        self.axV.set_xlim(-config.vmax, config.vmax)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

        f_init = config.initial_distribution
        self.x = np.linspace(0, config.system_length, config.ngridx, endpoint=False)
        self.v = np.linspace(-config.vmax, config.vmax, config.ngridv, endpoint=False)
        extent = [0, config.system_length, -config.vmax, config.vmax]
        self.im = self.axF.imshow(
            f_init.sum(axis=0), cmap="plasma", extent=extent, origin="lower"
        )
        self.figure.colorbar(self.im, ax=[self.axF])

        values = vlasov.vp2d(config)
        tick = 10
        self.ndt = tick * config.dt
        self.gen = itertools.islice(values, 0, None, tick)
        self.config = config

    def plot(self, i):
        (f, rho, E) = next(self.gen)
        time = i * self.ndt
        self.figure.suptitle(f"T = {time:.3g}")

        f_total = f.sum(axis=0)
        self.im.set_data(f_total)
        self.im.set_clim(vmin=np.min(f_total), vmax=np.max(f_total))

        for line in self.axR.get_lines():
            line.remove()
        self.axR.plot(self.x, rho, color="black")

        for line in self.axE.get_lines():
            line.remove()
        self.axE.plot(self.x, E, color="black")

        for line in self.axV.get_lines():
            line.remove()
        self.axV.plot(
            self.v, f_total.sum(axis=1), color="black", linewidth=0.3, label="total"
        )
        self.axV.set_prop_cycle(None)
        for s in range(self.config.species.n):
            name = self.config.species.name[s]
            g = f[s].sum(axis=1)
            self.axV.plot(self.v, g, label=name)
        self.axV.legend()

    def close_animation(self):
        self.animation.event_source.stop()

    def onKeyDown(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_SPACE:
            if self.is_running:
                self.animation.event_source.stop()
            else:
                self.animation.event_source.start()
            self.is_running = not self.is_running

        event.Skip()


class PlotFrame(wx.Frame):
    def __init__(self, parent=None):
        super().__init__(None, title="plot", size=(900, 600))
        self.panel = Plot(self)
        self.Bind(wx.EVT_CLOSE, self.onQuit)

    def init_figure(self, config):
        self.panel.init_figure(config)

    def onQuit(self, event):
        self.panel.close_animation()
        self.Destroy()


class MainFrame(wx.Frame):
    def __init__(self, parent=None):
        super().__init__(None, title="EmVSC")
        panel = wx.Panel(self)
        self.config = None

        loadBtn = wx.Button(panel, label="Load File")
        self.Bind(wx.EVT_BUTTON, self.onLoadBtn, loadBtn)

        runBtn = wx.Button(panel, label="Run")
        self.Bind(wx.EVT_BUTTON, self.onRunBtn, runBtn)

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(mainSizer)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        mainSizer.Add(buttons)
        buttons.Add(loadBtn)
        buttons.Add(runBtn)

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
            self.config = load_toml_file(t)
        dialog.Destroy()

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
