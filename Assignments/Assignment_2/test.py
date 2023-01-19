import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from GD import GradientDescent, BatchGradientDescent, LogisticGradientDescent


X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from GD import GradientDescent, BatchGradientDescent, LogisticGradientDescent

gd_errors = []


def callback(model, w, epoch):
    if epoch % 10 == 0:
        y_pred = model.predict(X)
        error = model._get_loss(y_pred, y)
        gd_errors.append(error)


gd = GradientDescent()
gd.fit(X, y, learning_rate=0.1, epochs=10000, verbose=0, callback=callback)

# print(gd.weights)
