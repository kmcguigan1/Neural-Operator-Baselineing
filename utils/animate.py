import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


with np.load(r'logs/results/boysenberry-bun-792-test.npz') as data:
    # predictions = data['predictions'] - data['actuals']
    # predictions = data['predictions']
    predictions = data['actuals']
    # actuals = data['actuals']

fig, ax = plt.subplots(1,1)
plot = ax.imshow(predictions[0, :, :, 0])

def update(frame):
    plot.set_data(predictions[0, :, :, frame])

ani = FuncAnimation(fig, update, frames=predictions.shape[-1])
ani.save('acts.gif', fps=3)
plt.colorbar(plot)
plt.show()