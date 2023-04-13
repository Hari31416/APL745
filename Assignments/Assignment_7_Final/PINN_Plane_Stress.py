import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from scipy.io import loadmat

torch.manual_seed(123456)
np.random.seed(123456)

E = 1  # Young's Modulus
nu = 0.3  # Poisson Ratio
G = E / (2 * (1 + nu))  # Shear modulus


class Model(nn.Module):
    def __init__(self, X_interior, X_boundary, U_boundary):
        super(Model, self).__init__()
        self.X_interior = X_interior
        self.X_boundary = X_boundary
        self.U_boundary = U_boundary
        self.Nnet = nn.Sequential()
        self.Nnet.add_module("Linear_layer_1", nn.Linear(2, 30))
        self.Nnet.add_module("Tanh_layer_1", nn.Tanh())
        self.Nnet.add_module("Linear_layer_2", nn.Linear(30, 30))
        self.Nnet.add_module("Tanh_layer_2", nn.Tanh())
        self.Nnet.add_module("Linear_layer_3", nn.Linear(30, 30))
        self.Nnet.add_module("Tanh_layer_3", nn.Tanh())
        self.Nnet.add_module("Linear_layer_4", nn.Linear(30, 30))
        self.Nnet.add_module("Tanh_layer_4", nn.Tanh())
        self.Nnet.add_module("Linear_layer_5", nn.Linear(30, 30))
        self.Nnet.add_module("Tanh_layer_5", nn.Tanh())
        self.Nnet.add_module("Output_layer", nn.Linear(30, 2))
        print(self.Nnet)

        self.pde_losses = []
        self.boundary_losses = []
        self.total_losses = []

    # Forward Feed
    def forward(self, x):
        y = self.Nnet(x)
        return y

    def pde_loss(self, X_i):
        """Calculate the loss for the PDE.

        args:
            X_i: tensor of shape (N, 2)
        returns:
            loss_pde: the loss for the PDE
        """
        x, y = X_i[:, 0], X_i[:, 1]
        U_i = self.forward(X_i)
        u = U_i[:, 0]
        v = U_i[:, 1]

        dudx, dudy = torch.autograd.grad(
            u.sum(), X_i, create_graph=True, retain_graph=True
        )[0].T
        dvdx, dvdy = torch.autograd.grad(
            v.sum(), X_i, create_graph=True, retain_graph=True
        )[0].T

        du2dx2, du2dxdy = torch.autograd.grad(
            dudx.sum(), X_i, create_graph=True, retain_graph=True
        )[0].T
        du2dydx, du2dy2 = torch.autograd.grad(
            dudy.sum(), X_i, create_graph=True, retain_graph=True
        )[0].T

        dv2dx2, dv2dxdy = torch.autograd.grad(
            dvdx.sum(), X_i, create_graph=True, retain_graph=True
        )[0].T
        dv2dydx, dv2dy2 = torch.autograd.grad(
            dvdy.sum(), X_i, create_graph=True, retain_graph=True
        )[0].T

        l1 = (
            G * (du2dx2 + du2dy2)
            + G * ((1 + v) / (1 - v)) * (du2dx2 + dv2dydx)
            + torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)
        )

        l2 = (
            G * (dv2dx2 + dv2dy2)
            + G * ((1 + v) / (1 - v)) * (du2dydx + dv2dy2)
            + torch.sin(torch.pi * x)
            + torch.sin(2 * torch.pi * y)
        )

        loss_pde = torch.mean(l1**2) + torch.mean(l2**2)
        return loss_pde

    def loss(self):
        """Calculate the total loss."""
        l_pde = self.pde_loss(self.X_interior)
        U_pred_b = self.forward(self.X_boundary)
        l_b = 1000 * torch.mean((U_pred_b - self.U_boundary) ** 2)
        l_total = l_pde + l_b

        self.pde_losses.append(l_pde.item())
        self.boundary_losses.append(l_b.item())
        self.total_losses.append(l_total.item())
        return l_total
