import os
import numpy as np
import matplotlib.pyplot as plt

# RESULTS_PATH = r'C:\Users\Kiernan\Documents\GitHub\Neural-Operator-Baselineing\utils'
RESULTS_PATH = '.'
NAME1 = 'clone-republic-555-test.npz'
NAME2 = 'old-fleet-570-test.npz'
# NAME = 'curious-meadow-882-test.npz'
# NAME = 'lunar-glade-888-test.npz'
# NAME = 'spring-dragon-894-test.npz'
# NAME = 'pleasant-sponge-896-test.npz'

def load_data(name):
    with np.load(os.path.join(RESULTS_PATH, name), 'rb') as data:
        predictions = data['predictions']
        actuals = data['actuals']
    return predictions, actuals

# def main():
#     predictions, actuals = load_data()
#     error = np.abs(predictions - actuals)
#     error = error.mean(axis=(0,3))
#     print(error.shape)
#     im = plt.imshow(error, vmin=0, cmap='rainbow')
#     plt.colorbar(im)
#     plt.savefig(f"{NAME.replace('.npz', '')}.png")
#     plt.show()

def calculate_boundary_error(error):
    # generate the mask of error on the boundaries
    mask = np.zeros_like(error)

    mask[[0, -1], :] = 1
    mask[:, [0, -1]] = 1
    
    # mask[[0, 1, -2, -1], :] = 1
    # mask[:, [0, 1, -2, -1]] = 1

    # mask[[0, 1, 2, -3, -2, -1], :] = 1
    # mask[:, [0, 1, 2, -3, -2, -1]] = 1

    # mask[[0, 1, 2, 3, -4, -3, -2, -1], :] = 1
    # mask[:, [0, 1, 2, 3, -4, -3, -2, -1]] = 1

    mask = mask.astype(bool)
    # generate the internal error
    arr = np.ma.array(error, mask=mask)
    internal_error = arr.mean()
    # generate the boundary error
    arr = np.ma.array(error, mask=~mask)
    boundary_error = arr.mean()
    return internal_error, boundary_error

def plot_multiple():
    pred1, act1 = load_data(NAME1)
    error1 = np.abs(pred1 - act1).mean(axis=(0,3))
    internal, boundary = calculate_boundary_error(error1)
    print("FNO")
    print(f"\tInternal Error: {internal:.6f}")
    print(f"\tBoundary Error: {boundary:.6f}")
    print(f"\t  Error Rratio: {boundary / internal:.6f}")

    pred2, act2 = load_data(NAME2)
    error2 = np.abs(pred2 - act2).mean(axis=(0,3))
    internal, boundary = calculate_boundary_error(error2)
    print("GFNO")
    print(f"\tInternal Error: {internal:.6f}")
    print(f"\tBoundary Error: {boundary:.6f}")
    print(f"\t  Error Rratio: {boundary / internal:.6f}")

    vmin = min(error1.min(), error2.min())
    vmax = max(error1.max(), error2.max())
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(error1, vmin=vmin, vmax=vmax, cmap='Reds')
    im = axs[1].imshow(error2, vmin=vmin, vmax=vmax, cmap='Reds')
    fig.subplots_adjust(right=0.8)
    cbar_ax=fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig("compare.png")
    plt.show()





if __name__ == '__main__':
    plot_multiple()