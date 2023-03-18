import torch
import numpy as np


class PhysicsInformedBarModel:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, x, E, A, L, u0, dist_load):
        """Construct a PhysicsInformedBar model
        Enter your code
        Task : initialize required variables for the class
        """
        self.x = x
        self.E = E
        self.A = A
        self.L = L
        self.u0 = u0
        self.dist_load = dist_load

    def costFunction(self, x, u_pred):
        """Compute the cost function.
        This function takes input x and model predicted output u_pred to compute loss

        Params:
            x - spatial value
            u_pred - NN predicted displacement value
            differential_equation_loss - calculated PDE residual loss
            boundary_condition_loss - calculated boundary loss
        """

        n = 2
        u_true = [0, 0]
        mse = torch.square(u_true - u_pred)
        mse = torch.sum(mse) / n

        torch.autograd.grad()

        return differential_equation_loss, boundary_condition_loss

    def get_displacements(self, x):
        """Get displacements.
        This function is used while inference (you can even use in your training phase if needed.
        It takes x as input and returns model predicted displacement)
        Params:
            x - input spatial value
            u - model predicted displacement
        """

        "Enter your code"

        return u

    def train(self, epochs, optimizer, **kwargs):
        """Train the model.
        This function is used for training the network. While updating the model params use "costFunction"
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
