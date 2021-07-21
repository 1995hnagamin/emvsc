import wx
import numpy as np

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

class Plot(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.figure = mpl.figure.Figure(figsize=(2, 2))
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)

        ax = self.figure.add_subplot(111)
        x = np.linspace(-np.pi, 3*np.pi, 500)
        ax.plot(x, np.sin(x))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)


if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame(None, -1, 'sin x')
    plot = Plot(frame)
    frame.Show()
    app.MainLoop()
