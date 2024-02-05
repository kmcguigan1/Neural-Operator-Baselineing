import torch
import torch.nn as nn 
import torch.nn.functional as F

class GausInstanceNormLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.means = None
        self.vars = None
    def transform(self, x):
        self.means = torch.mean(x, dim=(1,2,3), keepdim=True)
        self.vars = torch.std(x, dim=(1,2,3), keepdim=True)
        print(self.means.shape)
        print(self.vars.shape)
        return (x - self.means) / self.vars
    def inverse(self, x):
        return x * self.vars + self.means

def test():
    layer = GausInstanceNormLayer()
    x = torch.rand(3, 4, 2, 2)
    print(x.shape)
    x = layer.transform(x)
    print(x.shape)
    x = layer.inverse(x)
    print(x.shape)

if __name__ == '__main__':
    test()