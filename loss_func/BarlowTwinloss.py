import torch


class BarlowTwinLoss(torch.nn.Module):
    """
    Barlow Twin Loss:
    from https://github.dev/facebookresearch/barlowtwins
    """

    def __init__(self, lambd=1.0):
        super(BarlowTwinLoss, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        on_diag = torch.diagonal(x).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(x).pow_(2).sum()
        off_diag_loss = self.lambd * off_diag
        loss = on_diag + off_diag_loss
        return loss, on_diag, off_diag_loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

