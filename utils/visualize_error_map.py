import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = r'C:\Users\Kiernan\Documents\GitHub\Neural-Operator-Baselineing\results'
# NAME = 'dainty-leaf-861-test.npz'
# NAME = 'rich-violet-860-test.npz'
# NAME = 'curious-meadow-882-test.npz'
# NAME = 'lunar-glade-888-test.npz'
# NAME = 'spring-dragon-894-test.npz'
NAME = 'pleasant-sponge-896-test.npz'

def load_data():
    with np.load(os.path.join(RESULTS_PATH, NAME), 'rb') as data:
        predictions = data['predictions']
        actuals = data['actuals']
    print(predictions.shape)
    print(actuals.shape)
    return predictions, actuals

def main():
    predictions, actuals = load_data()
    error = np.abs(predictions - actuals)
    error = error.mean(axis=(0,3))
    print(error.shape)
    im = plt.imshow(error, vmin=0, cmap='rainbow')
    plt.colorbar(im)
    plt.savefig(f"{NAME.replace('.npz', '')}.png")
    plt.show()

if __name__ == '__main__':
    main()