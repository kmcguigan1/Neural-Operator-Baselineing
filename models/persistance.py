import torch
import torch.nn as nn
import torch.nn.functional as F

class PersistanceModel(nn.Module):
    def __init__(self, config:dict, img_size:tuple):
        super().__init__()
        self.time_steps_out = config["TIME_STEPS_OUT"]

    def forward(self, x, grid):
        B, H, W, C = x.shape
        assert C == 1, "Give only one past example to persistance model"
        x = torch.unsqueeze(x, dim=-1)
        x = x.repeat(1, 1, 1, self.time_steps_out, 1)
        x = x.squeeze(dim=-1)
        return x

def test():
    sample = torch.rand(3, 2, 2, 1)
    model = PersistanceModel({"TIME_STEPS_OUT":12}, (2,2))
    preds = model.forward(sample)
    print(preds.shape)
    print(sample[0, 0, 0, ...])
    print(preds[0, 0, 0, ...])

if __name__ == '__main__':
    test()