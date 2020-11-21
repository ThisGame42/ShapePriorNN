import torch
import torch.nn.functional as F


import ThinPlateSpline.TPS as tps


class DeformationNet(torch.nn.Module):
    def __init__(self, out_shape, ctrlpt_shape=(6, 6), base_filter=32):
        super().__init__()
        self.n_ctrlpt = ctrlpt_shape[0] * ctrlpt_shape[1]
        self.out_shape = out_shape
        self.n_param = (self.n_ctrlpt + 2)
        ctrl = tps.uniform_grid(ctrlpt_shape)
        self.register_buffer('ctrl', ctrl.view(-1,2))

        self.f = torch.nn.Sequential(
            torch.nn.Conv2d(1, base_filter, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(base_filter),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(base_filter, base_filter * 2, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(base_filter * 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(base_filter * 2, base_filter * 4, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(base_filter * 4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d((5,5))
        )

        self.loc = torch.nn.Sequential(
            torch.nn.Linear(25*128, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.n_param * 2),
            torch.nn.Tanh()
        )

        self.loc[-2].weight.data.normal_(0, 1e-3)
        self.loc[-2].bias.data.zero_()

    def forward(self, x, prior):
        feature = self.f(x)
        theta = self.loc(feature.view(x.shape[0], -1)).view(-1, self.n_param, 2)
        grid = tps.tps_grid(theta, self.ctrl, (x.shape[0], ) + self.out_shape)
        return F.grid_sample(prior, grid, align_corners=False), theta