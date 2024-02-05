import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk

def load_data(data_file='basic_heat_dataset.npy'):
    # base = os.path.dirname(os.getcwd())
    with open(os.path.join(os.getcwd(), 'data', data_file), mode='rb') as f:
        train_data = np.load(f)
        val_data = np.load(f)
        test_data = np.load(f)
    all_examples = np.concatenate((train_data, val_data, test_data), axis=0)
    return all_examples

class SpatialForecastApp:
    def __init__(self, root, data):
        self.root = root
        # get the actuals and the fixed things
        self.data = data
        self.shape = self.data.shape

        # setup the indecies we need
        self.example_idx = 0
        self.step_idx = 0

        # frame to show the images
        self.canvas = tk.Canvas(self.root, width=600, height=300)
        self.canvas.pack()

        self.fig = Figure(figsize=(12,8), dpi=100, tight_layout=True)
        self.create_initial_plots()

        self.fig_canvas = FigureCanvasTkAgg(self.fig, self.canvas)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack()

        self.root.bind("<Left>", self.prev_step)
        self.root.bind("<Right>", self.next_step)
        self.root.bind("<Down>", self.prev_example)
        self.root.bind("<Up>", self.next_example)

    def generate_title_string(self):
        return f"PDE example {self.example_idx} at step {self.step_idx+1} of {self.shape[1]} ({self.step_idx / (self.shape[1]-1) :.2f}%)"

    def create_initial_plots(self):
        # these are for the forecasts
        self.ax = self.fig.add_subplot(1, 1, 1)
        # set the axes titles
        self.ax.set_title(self.generate_title_string())
        # create the images
        self.im = self.ax.imshow(self.data[self.example_idx, self.step_idx, ...], cmap='RdBu_r')

    def display_forecast(self):
        # update images
        self.im.set_data(self.data[self.example_idx, self.step_idx, ...])
        # update the title
        self.ax.set_title(self.generate_title_string())
        # draw the canvas
        self.fig_canvas.draw()

    def prev_step(self, event):
        # Show the previous forecast
        if self.step_idx > 0:
            self.step_idx -= 1
            self.display_forecast()

    def next_step(self, event):
        # Show the next forecast
        if self.step_idx < self.shape[1] - 1:
            self.step_idx += 1
            self.display_forecast()

    def prev_example(self, event):
        # Show the previous forecast
        if self.example_idx > 0:
            self.example_idx -= 1
            self.display_forecast()

    def next_example(self, event):
        # Show the next forecast
        if self.example_idx < self.shape[0] - 1:
            self.example_idx += 1
            self.display_forecast()

def test():
    data = load_data()
    print(data.shape)

    root = tk.Tk()
    root.title("Spatial Forecast Visualization App")

    app = SpatialForecastApp(root, data)
    root.mainloop()

if __name__ == '__main__':
    test()