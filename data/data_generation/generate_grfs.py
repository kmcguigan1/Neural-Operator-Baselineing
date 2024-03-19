# Main dependencies
import numpy as np
import torch
import math

from timeit import default_timer

np.random.seed(1999)
torch.manual_seed(1999)

class GaussianRF(object):

    def __init__(self, dim=2, size=64, alpha=1.5, tau=2, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real


def main():
    grf = GaussianRF()
    intial_conditions = grf.sample(5000)
    train_split = int(intial_conditions.shape[0] * 0.7)
    val_split = train_split + int(intial_conditions.shape[0] * 0.15)
    np.savez(
        f'grf_initial_conditions.npz',
        train_conditions=intial_conditions[:train_split,...],
        val_conditions=intial_conditions[train_split:val_split,...],
        test_conditions=intial_conditions[val_split:,...],
    )

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3,3)
    # for i, alpha in enumerate([1, 2, 3]):
    #     for j, tau in enumerate([1, 2, 3]):
    #         grf = GaussianRF(alpha=alpha, tau=tau)
    #         intial_conditions = grf.sample(1)[0,...] * 2
    #         axs[i,j].imshow(intial_conditions, cmap='rainbow')
    #         axs[i,j].set_title(f"{alpha} - {tau} - {intial_conditions.mean():.2f} {intial_conditions.std():.2f} {intial_conditions.min():.2f} {intial_conditions.max():.2f}")
    #         axs[i,j].imshow(intial_conditions, cmap='rainbow')
    #         axs[i,j].set_xticks([])
    #         axs[i,j].set_yticks([])

    # plt.show()
    
if __name__ == '__main__':
    main()