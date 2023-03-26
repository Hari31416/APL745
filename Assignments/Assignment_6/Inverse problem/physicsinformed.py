import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class InversePhysicsInformedBarModel:
    """
    A class used for the definition of the data driven approach for Physics Informed Models for one dimensional bars.
    EA is estimated.
    """

    def __init__(self, x, L, dist_load):
        """Construct a InversePhysicsInformedBarModel model"""

        """
         Enter your code
         Task : initialize required variables for the class
        """
        self.x = x
        self.L = L
        self.dist_load = dist_load
        self.u_analytical = lambda x: torch.sin(2 * torch.pi * x)

    def predict(self, x):
        """Predict parameter EA of the differential equation."""

        """
        Params: 
            x - input spatial value
            u - input displacement value at x
            ea - model predicted value
        """

        """
        Enter your code
        """

        return self.model(x)

    def loss(self, inputs, verbose=True):
        e = self.model(inputs)
        e_x = torch.autograd.grad(
            e,
            inputs,
            grad_outputs=torch.ones_like(e),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        u = self.u_analytical(inputs)
        u_x = torch.autograd.grad(
            u,
            inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            inputs,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0]

        l = torch.mean((e * u_xx + e_x * u_x + self.dist_load(inputs)) ** 2)
        return l

    def train(self, epochs, optimizer, lr):
        """Train the model."""

        """
        This function is used for training the network. While updating the model params use "cost_function" 
        function for calculating loss
        Params:
            epochs - number of epochs
            optimizer - name of the optimizer
            **kwarhs - additional params

        This function doesn't have any return values. Just print the losses during training
        
        """

        """
            Enter your code        
        """
        model = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )
        self.model = model
        inputs = self.x

        optimizer = optimizer.lower()
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
            )
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=lr,
                max_iter=1000,
                max_eval=100,
                tolerance_grad=1e-05,
                tolerance_change=1e-06,
            )

        losses = []
        for epoch in range(epochs):
            u = self.model(inputs)
            loss_ = self.loss(inputs)
            optimizer.zero_grad()
            loss_.backward()

            def closure():
                optimizer.zero_grad()
                loss_ = self.loss(inputs)
                loss_.backward()
                return loss_

            optimizer.step(closure=closure)
            losses.append(loss_.item())

            print(f"Epoch {epoch+1:>4d}/{epochs} | loss={loss_.item():.4f}")

        return losses
