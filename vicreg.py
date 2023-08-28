import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets





class VICReg_w(nn.Module):
    def __init__(self, num_features,size,r=1,l=3):
        super().__init__()
        # self.args = args
        self.l = l
        self.num_features = int(num_features)*size*size
        self.projector = Projector(self.num_features,self.l)
        self.avg = nn.AdaptiveAvgPool2d((size,size))
        self.r = r
        self.conv1_x = nn.Conv2d(num_features,num_features,kernel_size=1,stride=1,padding=0,bias=True)
        self.conv1_y = nn.Conv2d(num_features,num_features,kernel_size=1,stride=1,padding=0,bias=True)
        self.conv1_z = nn.Conv2d(num_features,num_features,kernel_size=1,stride=1,padding=0,bias=True)
        self.conv1_k = nn.Conv2d(num_features,num_features,kernel_size=1,stride=1,padding=0,bias=True)

    def forward(self, x, y,z,k):
        x = self.conv1_x(x)
        x = self.avg(x)
        x = x.reshape(x.shape[0], -1)
        y = self.conv1_y(y)
        y = self.avg(y)
        y = y.reshape(y.shape[0], -1)
        x = self.projector(x)
        y = self.projector(y)

        z = self.conv1_z(z)
        z = self.avg(z)
        z = z.reshape(z.shape[0], -1)
        k = self.conv1_k(k)
        k = self.avg(k)
        k = k.reshape(k.shape[0], -1)
        z = self.projector(z)
        k = self.projector(k)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        z = z - z.mean(dim=0)
        k = k - k.mean(dim=0)


        w = torch.cat((x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1),k.unsqueeze(1)),1)
        std_w = torch.sqrt(w.var(dim=1) + 0.0001)
        w_loss = torch.mean(F.relu(self.r - std_w))

        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_y = (y.T @ y) / (x.size(0) - 1)
        cov_z = (z.T @ z) / (z.size(0) - 1)
        cov_k = (y.T @ k) / (k.size(0) - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)+\
        off_diagonal(cov_z).pow_(2).sum().div(self.num_features)+\
        off_diagonal(cov_k).pow_(2).sum().div(self.num_features)


        loss = (

            w_loss+0.02*cov_loss
        )
        return loss





def Projector(num_features,l):
    # num_features = num_features[0]
    # mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    # f = list(map(int, mlp_spec.split("-")))
    for i in range(l - 2):
        layers.append(nn.Linear(num_features, num_features))
        layers.append(nn.BatchNorm1d(num_features))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(num_features, num_features, bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]