import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

EX_IDX = 0


with np.load(r'results/comic-durian-847-test.npz') as data:
    predictions = data['predictions'][EX_IDX, ...]
    actuals = data['actuals'][EX_IDX, ...]

error = predictions - actuals

vmin = min(actuals.min(), predictions.min())
vmax = min(actuals.max(), predictions.max())

fig, axs = plt.subplots(1,3, figsize=(14,8))
fig.suptitle('FNO Model')
actuals_im = axs[0].imshow(actuals[..., 0], vmin=vmin, vmax=vmax, cmap='rainbow', origin="lower",)
preds_im = axs[1].imshow(predictions[..., 0], vmin=vmin, vmax=vmax, cmap='rainbow', origin="lower",)
error_im = axs[2].imshow(error[..., 0], vmin=vmin, vmax=vmax, cmap='rainbow', origin="lower",)

for ax, title in zip(axs, ['Actuals', 'Predictions', 'Error']):
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def update(frame):
    actuals_im.set_data(actuals[..., frame])
    preds_im.set_data(predictions[..., frame])
    error_im.set_data(error[..., frame])

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(actuals_im, cax=cbar_ax, fraction=0.046, pad=0.04)

ani = FuncAnimation(fig, update, frames=predictions.shape[-1])
ani.save('fno_ns_e-3.gif', fps=3)
# fig.colorbar()
plt.show()