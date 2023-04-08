"""
Solution of 2D Forward Problem of Linear Elasticity
   for Plane Stress Boundary Value Problem using
      Physics-Informed Neural Networks (PINN)
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

torch.manual_seed(123456)
np.random.seed(123456)

E = 1                                       # Young's Modulus
nu = 0.3                                    # Poisson Ratio
G =                                         # Shear modulus

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define your model here (refer: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
        self.Nnet = nn.Sequential()
        self.Nnet.add_module('Linear_layer_1', nn.Linear(,))    # First linear layer
        self.Nnet.add_module('Tanh_layer_1', nn.Tanh())         # Add activation
        
        
        
        print(self.Nnet)                                        # Print model summary

    # Forward Feed
    def forward(self, x):
        y = self.Nnet(x)
        return y

    # PDE and BCs loss
    def loss(self, __ENTER VARIABLES HERE__):
        y =                                 # Interior Solution (output from from defined NN model)
        y_b =                               # Boundary Solution (output from from defined NN model)
        u_b, v_b =                          # u and v boundary
        u,v =                               # u and v interior

        # Calculate Gradients
        # Gradients of deformation in x-direction (first and second derivatives)
        u_g =                               # Gradient of u, Du = [u_x, u_y]
        u_x, u_y =                          # [u_x, u_y]
        u_xx =                              # Second derivative, u_xx
        u_xy =                              # Mixed partial derivative, u_xy
        u_yy =                              # Second derivative, u_yy

        # Gradients of deformation in y-direction (first and second derivatives)
        v_g =                               # Gradient of v, Dv = [v_x, v_y]
        v_x, v_y =                          # [v_x, v_y]
        v_xx =                              # Second derivative, v_xx
        v_xy =                              # Mixed partial derivative, v_xy
        v_yy =                              # Second derivative, v_yy

        f_1 =                               # Define body force for PDE-1
        f_2 =                               # Define body force for PDE-2
        
        loss_1 =                            # Define loss for PDE-1
        loss_2 =                            # Define loss for PDE-2

        loss_PDE =                          # MSE PDE loss
        loss_bc =                           # MSE BCs loss

        TotalLoss = 
        print(f'epoch {epoch}: loss_pde {loss_PDE:.8f}, loss_bc {loss_bc:.8f}')
        return TotalLoss

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),allow_unused=True, create_graph=True)

# Define model parameters here
device = ("cuda" if torch.cuda.is_available() else "cpu")



# Load the collocation point data




## Define data as PyTorch Tensor and send to device
xy_f_train = torch.tensor(ADD_HERE, requires_grad=True, dtype=torch.float32).to(device)
xy_b_train = torch.tensor(ADD_HERE, requires_grad=True, dtype=torch.float32).to(device)

# Define the boundary condition values
u_b_train = 
v_b_train = 

# Initialize model
model = Model().to(device)

# Loss and Optimizer
optimizer = 

# Training
def train(epoch):
    model.train()

    def closure():
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    print(f'epoch {epoch}: loss {loss:.8f} ')
    return loss

for epoch in range():
    train(epoch)

# %%
''' Testing PINN '''
# Generate collocation points and use trained model for testing

    
# %%
''' Plotting '''
