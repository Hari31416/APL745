import torch
from torch import nn


class PhysicsInformedBarModel:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, x, E, A, L, u0, dist_load):
        """Construct a PhysicsInformedBar model"""
        self.x = x
        self.E = E
        self.A = A
        self.L = L
        self.u0 = u0
        self.dist_load = dist_load

    def _mseu(self, y_pred, y_true):
        """Mean squared error for initial condition"""
        # Mean squared error for u
        return torch.mean((y_pred - y_true) ** 2)

    def _msec(self, x, model):
        """Mean squared error for collocation points"""
        # calculates E*A*u_xx + p(x)
        u = model(x)
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0]
        sum_ = self.E * self.A * u_xx + self.dist_load(x)
        return torch.mean(sum_**2)

    def loss(self, inputs, model, verbose=True):
        """Compute the cost function.
        This function takes input and model and returns the loss

        Parameters:
        ----------
        inputs : torch.Tensor
            Input tensor
        model : torch.nn.Module
            Model to be trained
        verbose : bool
            If True, print the loss values

        Returns:
        -------
        total_loss : torch.Tensor
            Total loss
        """

        n = 2  # Number of boundary points
        x0 = inputs[:n]
        x = inputs[n:]
        mseu_loss = self._mseu(model(x0), self.u0)
        msec_loss = self._msec(x, model)
        total_loss = mseu_loss + msec_loss
        if verbose:
            print(
                f"MSEu: {mseu_loss:.4f} | MSEc: {msec_loss:.4f} | Total: {total_loss:.4f}"
            )
        return total_loss

    def predict(self, x):
        """Predict the output for given input x
        This function takes input x and returns the predicted output

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor

        Returns:
        -------
        y : torch.Tensor
            Predicted output
        """
        return self.model(x)

    def train(self, epochs, optimizer, lr, **kwargs):
        """Train the model.

        Parameters:
        ----------
        epochs : int
            Number of epochs
        optimizer : str
            Optimizer to be used
        lr : float
            Learning rate
        **kwargs : dict
            Additional arguments for the optimizer

        Returns:
        -------
        losses : list
            List of losses
        """
        model = nn.Sequential(
            nn.Linear(1, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1),
        )
        self.model = model

        optimizer = optimizer.lower()
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"])
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=kwargs["lr"])
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
        inputs = self.x
        for epoch in range(epochs):
            print(f"Epoch {epoch+1:>4d}/{epochs}")
            u = self.model(inputs)
            loss_ = self.loss(inputs, self.model)
            optimizer.zero_grad()
            loss_.backward()

            def closure():
                """Closure function for LBFGS optimizer"""
                optimizer.zero_grad()
                loss_ = self.loss(inputs, self.model, verbose=False)
                loss_.backward()
                return loss_

            optimizer.step(closure=closure)
            losses.append(loss_.item())

        return losses
