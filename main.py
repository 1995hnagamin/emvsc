import wx
import numpy as np
import itertools

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import matplotlib.animation as mplanim

import vlasov
class Plot(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.figure = mpl.figure.Figure(figsize=(2, 2))
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.animation = mplanim.FuncAnimation(self.figure, self.plot, interval=40)
        self.ax = self.figure.add_subplot(111)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

        resolution = 100
        self.xmax = 100.
        self.x = np.linspace(0, self.xmax, resolution)
        width = 5.
        center = self.xmax / 2
        f_init = np.exp(-((self.x-center)/width)**2)
        values = vlasov.adv1d(
            system_length=self.xmax,
            velocity=1.,
            init=f_init,
            ngrid=resolution,
            dt=.1)
        self.gen = itertools.islice(values, 0, None, 20)

    def plot(self, _):
        f = next(self.gen)

        self.ax.cla()
        self.ax.set_xlim(0, self.xmax)
        self.ax.set_ylim(-0.5, 1.5)
        self.ax.plot(self.x, f)

if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame(None, -1, 'sin x')
    plot = Plot(frame)
    frame.Show()
    app.MainLoop()
