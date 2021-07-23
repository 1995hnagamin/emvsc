import wx
import numpy as np
import itertools

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import matplotlib.animation as mplanim

import vlasov

ni = 1.0
q = np.array([-1.0, -1.0])
eps0 = 1.0
q_ion = 1.0
qm = np.array([-1.0, -1.0])
xmax = 20.0
vmax = 10.0
nx = 100
nv = 100
dt = 0.01

k = 2 * np.pi / xmax
amp = 0.3
w = 1.0
v0 = 3.0


def gaussian(x, x0, w):
    A = 1 / (2 * np.sqrt(2 * np.pi) * w)
    return A * np.exp(-((x - x0) ** 2) / (2 * w ** 2))


class Plot(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.figure = mpl.figure.Figure(figsize=(2, 2))
        canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.animation = mplanim.FuncAnimation(self.figure, self.plot, interval=20)
        self.is_running = True
        self.Bind(wx.EVT_KEY_DOWN, self.onKeyDown)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

        x = np.linspace(0, xmax, nx, endpoint=False)
        v = np.linspace(-vmax, vmax, nv, endpoint=False)
        xx, vv = np.meshgrid(x, v, sparse=True)
        f0a = ni * gaussian(vv, +v0, w) * (1 + amp * np.cos(k * xx))
        f0b = ni * gaussian(vv, -v0, w) * (1 + amp * np.cos(k * xx))
        f_init = np.array([f0a, f0b])
        self.x = x
        self.v = v

        values = vlasov.vp2d(
            q=q,
            qm=qm,
            ion_density=q_ion * ni,
            system_length=xmax,
            vmax=vmax,
            init=f_init,
            ngridx=nx,
            ngridv=nv,
            dt=dt,
        )
        self.tick = 10
        self.gen = itertools.islice(values, 0, None, self.tick)

    def plot(self, i):
        (f, rho, E) = next(self.gen)
        self.figure.clf()
        self.figure.subplots_adjust(hspace=0.5, wspace=0.3)
        time = i * self.tick * dt
        self.figure.suptitle(f"T = {time:.3g}")

        axF = self.figure.add_subplot(221)
        axF.set_title("distibution function")
        im = axF.imshow(f.sum(axis=0), cmap="plasma", extent=[0, xmax, -vmax, vmax], origin="lower")
        self.figure.colorbar(im)

        axR = self.figure.add_subplot(222)
        axR.set_title("charge density")
        axR.set_xlabel("x")
        axR.plot(self.x, rho)
        axR.set_xlim(0, xmax)
        axR.set_ylim(-0.5, 0.5)
        axR.grid(True)

        axE = self.figure.add_subplot(223)
        axE.set_title("electric field Ex")
        axE.set_xlabel("x")
        axE.plot(self.x, E)
        axE.set_xlim(0, xmax)
        axE.set_ylim(-1.2, 1.2)
        axE.grid(True)

        axV = self.figure.add_subplot(224)
        axV.set_title("velocity distribution")
        axV.set_xlabel("v")
        for s in range(len(q)):
            g = f[s].sum(axis=1)
            axV.plot(self.v, g, label=f"species #{s}")
        axV.set_xlim(-vmax, vmax)
        axV.set_ylim(0, 30)
        axV.legend()
        axV.grid(True)

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

    def onQuit(self, event):
        self.panel.close_animation()
        self.Destroy()


class MainFrame(wx.Frame):
    def __init__(self, parent=None):
        super().__init__(None, title="EmVSC")
        panel = wx.Panel(self)
        runBtn = wx.Button(panel, label="Run")
        self.Bind(wx.EVT_BUTTON, self.onRunBtn, runBtn)

    def onRunBtn(self, _):
        plotFrame = PlotFrame(self)
        plotFrame.Centre()
        plotFrame.Show()


if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    frame.Show()
    app.MainLoop()
