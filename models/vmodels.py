import torch
import torchvision
import torch.nn as nn


class MResNet(torchvision.models.resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Identity()


class ResNet14(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = MResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.net(x)
        x = self.fc(x)
        return x

from utils.spec import SpecAugmentation
from loss_func.BarlowTwinloss import BarlowTwinLoss

class ResNet14_BT(ResNet14):
    def __init__(self, num_classes=10):
        super().__init__(num_classes)

        self.spec_aug = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                         freq_drop_width=8, freq_stripes_num=2)

        self.btl = BarlowTwinLoss(0.001)

        self.projector = nn.Linear(512, 2048)

        self.bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        aug_x = self.spec_aug(x)

        out1 = self.net(x)
        t1 = self.fc(out1)
        if not self.training:
            return t1
        out2 = self.net(aug_x)
        emb1 = self.projector(out1)
        emb2 = self.projector(out2)
        c = self.bn(emb1).T @ self.bn(emb2) / x.shape[0]
        btl = self.btl(c)
        t2 = self.fc(out2)
        return t1, t2, btl



if __name__ == "__main__":
    def print_nn(mm):
        def count_pars(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        num_pars = count_pars(mm)
        print(mm)
        print('# pars: {}'.format(num_pars))


    net = ResNet14()
    print_nn(net)
    # x = torch.randn(8, 1, 250, 64)
    # y, y2, c = net(x)
    # print(y.shape)