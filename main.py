import wx
import numpy as np
import itertools

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import matplotlib.animation as mplanim

import vlasov

ni = 1.
q = 1.
eps0 = 1.
m = 1.
xmax = 20.
vmax = 10.
nx = 100
nv = 100
dt = 0.001

k = 2 * np.pi / xmax
amp = 0.3
w = 1.
v0 = 3.

class Plot(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.figure = mpl.figure.Figure(figsize=(2, 2))
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.animation = mplanim.FuncAnimation(self.figure, self.plot, interval=1000)
        self.ax = self.figure.add_subplot(111)
        self.colorbar = None

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

        x = np.linspace(0, xmax, nx, endpoint=False)
        v = np.linspace(-vmax, vmax, nv, endpoint=False)
        xx, vv = np.meshgrid(x, v, sparse=True)
        gamma = ni / (2*np.sqrt(2*np.pi)*w)
        f_init = gamma*np.exp(-(vv-v0)**2/(2*w**2))*(1+amp*np.cos(k*xx)) \
                +gamma*np.exp(-(vv+v0)**2/(2*w**2))*(1+amp*np.cos(k*xx))
        self.x = x
        self.v = v

        values = vlasov.vp2d(
            q=q,
            m=m,
            ion_density=ni,
            system_length=xmax,
            vmax=vmax,
            init=f_init,
            ngridx=nx,
            ngridv=nv,
            dt=dt)
        self.gen = itertools.islice(values, 0, None, 1)

    def plot(self, _):
        (f, _, _) = next(self.gen)
        self.ax.cla()

        self.ax.imshow(f, cmap='plasma')

if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame(None, -1, 'sin x')
    plot = Plot(frame)
    frame.Show()
    app.MainLoop()
