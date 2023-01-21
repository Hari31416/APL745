import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from GD import BatchGradientDescent, SteepestDescent, Style
import os
import time

DATA_DIR = "data"
SAVE_DIR = "plots"


data = np.loadtxt(os.path.join(DATA_DIR, "prob1data.txt"), delimiter=",")
data = data.T

t = data[:, 0]
y = data[:, 1]
t = t.reshape(-1, 1)
X = np.concatenate((t, t**2), axis=1)


def train_for_tol(tol):
    bgd = BatchGradientDescent(fit_intercept=True, tol=tol)

    sd = SteepestDescent(fit_intercept=True, tol=tol)

    np.random.seed(42)
    tic = time.time()
    bgd.fit(X, y, epochs=100000, verbose=0)
    print(bgd.weights)
    toc = time.time()
    print(Style.OKBLUE + f"Time taken for BGD {toc - tic}", Style.END)
    print(Style.MAGENTA + "----" * 10 + Style.END)
    tic = time.time()
    sd.fit(X, y, epochs=10000, verbose=1)
    print(sd.weights)
    toc = time.time()
    print(Style.OKBLUE + f"Time taken for SD {toc - tic}", Style.END)


print(Style.HEADER + f"For tol = {1e-3}" + Style.END)
train_for_tol(1e-3)
print(Style.MAGENTA + "----" * 10 + Style.END)
print(Style.MAGENTA + "----" * 10 + Style.END)

print(Style.HEADER + f"For tol = {1e-4}" + Style.END)
train_for_tol(1e-4)
print(Style.MAGENTA + "----" * 10 + Style.END)
print(Style.MAGENTA + "----" * 10 + Style.END)

print(Style.HEADER + f"For tol = {1e-5}" + Style.END)
train_for_tol(1e-5)
print(Style.MAGENTA + "----" * 10 + Style.END)
print(Style.MAGENTA + "----" * 10 + Style.END)

print(Style.HEADER + f"For tol = {1e-6}" + Style.END)
train_for_tol(1e-6)
print(Style.MAGENTA + "----" * 10 + Style.END)
print(Style.MAGENTA + "----" * 10 + Style.END)
