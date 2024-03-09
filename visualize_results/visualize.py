import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk

class SpatialForecastApp:
    def __init__(self, root, data_path:str):
        self.root = root
        # get the data
        self._load_data(data_path)
        # get the indecies to go over
        self.example_idx = 0
        self.step_idx = 0
        # frame to show the images
        self.canvas = tk.Canvas(self.root, width=600, height=400)
        self.canvas.pack()

        self.fig = Figure(figsize=(15,15), dpi=100)
        self.gs = self.fig.add_gridspec(2, 2, wspace=0.1, hspace=0.1)
        self.create_initial_plots()

        self.fig_canvas = FigureCanvasTkAgg(self.fig, self.canvas)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack()

        self.root.bind("<Left>", self.prev_forecast_step)
        self.root.bind("<Right>", self.next_forecast_step)
        self.root.bind("<Down>", self.prev_forecast)
        self.root.bind("<Up>", self.next_forecast)

    def _load_data(self, data_path):
        with np.load(data_path) as file_data:
            self.forecasts = file_data['forecasts']
            self.actuals = file_data['actuals']
            self.last_input = file_data['last_input']
            self.metadata = file_data['metadata']
        print(self.forecasts.shape)
        print(self.actuals.shape)
        print(self.last_input.shape)
        print(self.metadata.shape)
        print(self.forecasts.min(), self.forecasts.max())
        print(self.actuals.min(), self.actuals.max())


    def create_initial_plots(self):
        # set the axes titles
        self.axes = {
            'init': self.fig.add_subplot(self.gs[0, 0]),
            'err': self.fig.add_subplot(self.gs[0, 1]),
            'acts': self.fig.add_subplot(self.gs[1, 0]),
            'fcast': self.fig.add_subplot(self.gs[1, 1]),
            # 'err_curve': self.fig.add_subplot(self.gs[2, :]),
        }
        # name the axes
        self.axes['init'].set_title('Initial Conditions')
        self.axes['err'].set_title('Error')
        self.axes['acts'].set_title('Actuals')
        self.axes['fcast'].set_title('Forecasts')
        # remove the ticks
        self.axes['init'].axes.get_xaxis().set_ticks([])
        self.axes['err'].axes.get_xaxis().set_ticks([])
        self.axes['acts'].axes.get_xaxis().set_ticks([])
        self.axes['fcast'].axes.get_xaxis().set_ticks([])

        self.axes['init'].axes.get_yaxis().set_ticks([])
        self.axes['err'].axes.get_yaxis().set_ticks([])
        self.axes['acts'].axes.get_yaxis().set_ticks([])
        self.axes['fcast'].axes.get_yaxis().set_ticks([])
        # self.axes['err_curve'].set_title('Error Curve')
        # draw the axes
        self.imgs = {}
        self.imgs['init'] = self.axes['init'].imshow(self.last_input[self.example_idx, ...], vmin=0, vmax=1.0, cmap='RdBu_r')
        self.imgs['err'] = self.axes['err'].imshow(self.actuals[self.example_idx, ..., self.step_idx] - self.forecasts[self.example_idx, ..., self.step_idx], vmin=0, vmax=1.0, cmap='RdBu_r')
        self.imgs['acts'] = self.axes['acts'].imshow(self.actuals[self.example_idx, ..., self.step_idx], vmin=0, vmax=1.0, cmap='RdBu_r')
        self.imgs['fcast'] = self.axes['fcast'].imshow(self.forecasts[self.example_idx, ..., self.step_idx], vmin=0, vmax=1.0, cmap='RdBu_r')
        self.fig.colorbar(self.imgs['init'], ax=[x for x in self.axes.values()])
        # self.fig.colorbar(ax=[x for x in self.imgs.values()])
        # set the plot title
        # self.plot_error_graph()
        self.fig.suptitle(f"""Forecasts""")

    def plot_error_graph(self):
        self.axes['err_curve'].clear()
        starting = int(self.metadata[self.example_idx, -1])
        stopping = starting + self.actuals.shape[-1]
        time_steps = np.arange(starting, stopping)
        print(time_steps.shape)
        error = self.actuals[self.example_idx,...] - self.forecasts[self.example_idx,...]
        error = error.mean(axis=(0, 1))
        print(error.shape)
        self.axes['err_curve'].plot(time_steps, error)

    def display(self):
        self.fig.suptitle(f"""Forecasts {self.example_idx} {self.step_idx}""")
        # update the images
        self.imgs['init'].set_data(self.last_input[self.example_idx, ...])
        self.imgs['err'].set_data(self.actuals[self.example_idx, ..., self.step_idx] - self.forecasts[self.example_idx, ..., self.step_idx])
        self.imgs['acts'].set_data(self.actuals[self.example_idx, ..., self.step_idx])
        self.imgs['fcast'].set_data(self.forecasts[self.example_idx, ..., self.step_idx])
        # line update
        # self.plot_error_graph()
        # draw the canvas
        self.fig_canvas.draw()

    def prev_forecast_step(self, event):
        # Show the previous forecast
        if self.step_idx > 0:
            self.step_idx -= 1
            self.display()

    def next_forecast_step(self, event):
        # Show the next forecast
        if self.step_idx < self.actuals.shape[-1] - 1:
            self.step_idx += 1
            self.display()

    def prev_forecast(self, event):
        # Show the previous forecast
        if self.example_idx > 0:
            self.example_idx -= 1
            self.display()

    def next_forecast(self, event):
        # Show the next forecast
        if self.example_idx < self.actuals.shape[0] - 1:
            self.example_idx += 1
            self.display()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Spatial Forecast Visualization App")

    app = SpatialForecastApp(root, '/Users/kiernan/Documents/GitHub/Neural-Operator-Baselineing/results/celestial-bush-670-test-results.npz')
    root.mainloop()