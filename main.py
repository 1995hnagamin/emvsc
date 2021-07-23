import wx
import numpy as np
import itertools

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import matplotlib.animation as mplanim

import vlasov

ni = 1.0
q = 1.0
eps0 = 1.0
m = 1.0
xmax = 20.0
vmax = 10.0
nx = 100
nv = 100
dt = 0.001

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
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.animation = mplanim.FuncAnimation(self.figure, self.plot, interval=20)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

        x = np.linspace(0, xmax, nx, endpoint=False)
        v = np.linspace(-vmax, vmax, nv, endpoint=False)
        xx, vv = np.meshgrid(x, v, sparse=True)
        f0a = ni * gaussian(vv, +v0, w) * (1 + amp * np.cos(k * xx))
        f0b = ni * gaussian(vv, -v0, w) * (1 + amp * np.cos(k * xx))
        f_init = f0a + f0b
        self.x = x
        self.v = v

        values = vlasov.vp2d(
            q=q,
            qm=q / m,
            ion_density=ni,
            system_length=xmax,
            vmax=vmax,
            init=f_init,
            ngridx=nx,
            ngridv=nv,
            dt=dt,
        )
        self.gen = itertools.islice(values, 0, None, 1)

        (f, rho, E) = next(self.gen)
        self.figure.clf()
        self.figure.subplots_adjust(hspace=0.5, wspace=0.3)

        axF = self.figure.add_subplot(221)
        im = axF.imshow(f, cmap="plasma")
        self.figure.colorbar(im)

        axR = self.figure.add_subplot(222)
        axR.plot(self.x, rho)

        axE = self.figure.add_subplot(223)
        axE.plot(self.x, E)

        axV = self.figure.add_subplot(224)
        g = f.sum(axis=1)
        axV.plot(self.v, g)


if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame(None, -1, "sin x")
    plot = Plot(frame)
    frame.Show()
    app.MainLoop()
