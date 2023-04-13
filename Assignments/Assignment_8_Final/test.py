import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy import integrate
from anti_derivative import PI_DeepONet
import os


DATA_DIR = "data"
PLOTS_DIR = "plots"

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)


data = np.load(
    os.path.join(DATA_DIR, "antiderivative_aligned_train.npz"), allow_pickle=True
)
X_temp = data["X"]
y_temp = data["y"]

x = X_temp[1]

# The U matrix
U = X_temp[0]

# The S matrix
S = y_temp

print(U.shape, S.shape, x.shape)

branch_layers = [100, 50, 50, 50, 50, 50]
trunk_layers = [1, 50, 50, 50, 50, 50]

model = PI_DeepONet(branch_layers, trunk_layers, U, x, S)
print(model.summary())
model.train(1000)
