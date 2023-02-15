from model import Sequential
from layers import Input, Dense, Dropout, BatchNormalization
import numpy as np
from sklearn.datasets import make_regression
import pandas as pd
from utils import one_hot
import matplotlib.pyplot as plt

fashion = pd.read_csv(
    "/media/hari31416/Hari_SSD/Users/harik/Desktop/APL745/Assignments/Assignment_4/data/fashion-mnist_train.csv"
)


def temp():

    fashion = pd.read_csv(
        "/media/hari31416/Hari_SSD/Users/harik/Desktop/APL745/Assignments/Assignment_4/data/fashion-mnist_train.csv"
    )
    # nx, m = 10, 1000
    # ny = 1
    # X = np.random.randn(nx, m)
    # X = X / np.linalg.norm(X, axis=0)
    # y_label = np.random.randint(0, ny + 1, (1, m))

    # X, y = load_breast_cancer(return_X_y=True)
    # X = X.T
    # X = X / np.linalg.norm(X, axis=0)
    # y = y.reshape(1, -1)
    # np.random.shuffle(X)
    # X = X[:, :50]
    # y = y[:, :50]

    # print(y.shape, X.shape, X.max(), X.min())

    # def one_hot(y_label, classes):
    #     y = np.zeros((classes, y_label.shape[1]))
    #     for i in range(y_label.shape[1]):
    #         y[y_label[0, i], i] = 1
    #     return y

    # Y = one_hot(y_label, ny)

    X = fashion.iloc[:, 1:].values
    y = fashion.iloc[:, 0].values
    y_oh = one_hot(y, 10)

    X = X.T
    X = X / X.max()
    # y = y.reshape(1, y.shape[0])
    # y = (y == 0).astype(int)
    # print(X.shape, y.shape, X.min(), X.max())
    nx, m = X.shape
    ny, m = y_oh.shape
    # ny, m = y.shape
    print(m, ny, nx)
    model = Sequential("my_model")
    inp = Input((nx,), name="First Input")
    dense = Dense(20, activation="relu", name="Dense 1")
    model.add(inp)
    model.add(dense)
    model.add(Dense(10, name="Dense A", activation="relu"))
    model.add(Dense(8, name="Dense C", activation="relu"))
    # model.add(Dense(ny, name="Dense Z", activation="sigmoid"))
    model.add(Dense(ny, name="Dense Z", activation="softmax"))

    model.summary()

    # model.compile(loss="binary_cross_entropy", metrics=["accuracy"])
    model.compile(
        loss="categorical_cross_entropy",
        metrics=["accuracy", "precision"],
        initializer="glorot",
    )

    # y_new = m[3].forward(m[2].forward(m[1].forward(X)))

    # y_hat = model.fit(X, y)
    # print(y_hat.max(), y_hat.min())
    # assert y_hat.shape == y.shape
    # assert np.allclose(y_hat, y_new)

    # model.fit(X, y, epochs=100, verbose=1, lr=0.01, batch_size=32)
    model.fit(X, y_oh, epochs=30, lr=0.01, batch_size=32)
    y_pred = model.predict(X)
    print(np.mean((y_oh == np.argmax(y_pred, axis=0))))


def classification(classes=(0, 1)):
    X = fashion.iloc[:, 1:].values
    y = fashion.iloc[:, 0].values

    y_1ids = np.where(y == classes[0])[0]
    y_2ids = np.where(y == classes[1])[0]

    y_final = np.concatenate((y_1ids, y_2ids))
    X_final = X[y_final]
    y_final = y[y_final]
    y_final = (y_final == classes[0]).astype(int)
    y = np.reshape(y_final, (1, -1))

    X = X_final.T
    X = X / X.max()
    nx, _ = X.shape
    ny = 1

    model = Sequential("my_model")
    inp = Input((nx,), name="First Input")
    model.add(inp)
    # model.add(
    #     Dense(
    #         30,
    #         activation="tanh",
    #         name="Dense 1",
    #         l1=0.001,
    #         l2=0.001,
    #     )
    # )
    # model.add(
    #     Dense(
    #         10,
    #         name="Dense A",
    #         activation="tanh",
    #         l1=0.001,
    #         l2=0.001,
    #     ),
    # )
    # model.add(Dense(8, name="Dense C", activation="tanh"))
    # model.add(Dense(ny, name="Dense Z", activation="sigmoid"))

    model.add(Dense(30, activation="relu", name="Dense 1"))
    # model.add(Dropout(0.5, "dropout 1"))
    model.add(Dense(10, name="Dense 2", activation="relu"))
    # model.add(Dropout(0.5, "dropout 2"))
    # model.add(BatchNormalization("batch norm 1"))
    model.add(Dense(8, name="Dense 3", activation="relu"))
    # model.add(Dropout(0.5, "dropout 3"))
    model.add(Dense(ny, name="Dense 4", activation="sigmoid"))
    model.summary()
    # print(model.info[0])
    # print(model.info["b"])

    model.compile(
        loss="binary_cross_entropy",
        metrics=["accuracy", "precision"],
        initializer="glorot",
    )

    history = model.fit(X, y, epochs=100, lr=0.01, batch_size=32, verbose=2)
    # history_df = pd.DataFrame(history)
    # width = 6
    # height = 6 * len(history_df.columns)
    # history_df.plot(subplots=True, figsize=(width, height))
    # plt.tight_layout()
    # plt.show()


def multi_class():

    X = fashion.iloc[:, 1:].values
    y = fashion.iloc[:, 0].values
    y_oh = one_hot(y, 10)

    X = X.T
    X = X / X.max()
    nx, m = X.shape
    ny, m = y_oh.shape
    print(m, ny, nx)

    model = Sequential("my_model")
    inp = Input((nx,), name="First Input")
    dense = Dense(20, activation="relu", name="Dense 1")
    model.add(inp)
    model.add(dense)
    model.add(Dense(10, name="Dense A", activation="relu"))
    # model.add(Dropout(0.2, "dropout 1"))
    model.add(Dense(8, name="Dense C", activation="relu"))
    model.add(Dense(ny, name="Dense Z", activation="softmax"))

    model.summary()

    model.compile(
        loss="categorical_cross_entropy",
        metrics=["accuracy", "precision", "recall", "f1"],
        initializer="glorot",
    )

    history = model.fit(X, y_oh, epochs=100, lr=0.05, batch_size=100, verbose=2)
    history_df = pd.DataFrame(history)
    width = 6
    height = 6 * len(history_df.columns)
    history_df.plot(subplots=True, figsize=(width, height))
    plt.tight_layout()
    plt.show()


def regression():
    m = 1000
    nx = 10
    ny = 1
    X, y, coef = make_regression(
        n_samples=m,
        n_features=nx,
        n_informative=8,
        n_targets=ny,
        bias=0.0,
        effective_rank=None,
        tail_strength=0.5,
        noise=0.02,
        shuffle=True,
        random_state=42,
        coef=True,
    )
    X = X.T
    y = y.reshape(1, -1)
    assert y.shape == (1, m)

    model = Sequential("my_model")
    inp = Input((nx,), name="First Input")
    # dense = Dense(10, activation="relu", name="Dense 1")
    model.add(inp)
    # model.add(dense)
    # model.add(Dense(10, name="Dense A", activation="tanh"))
    # model.add(Dense(5, name="Dense B", activation="relu"))
    # model.add(Dense(5, name="Dense D", activation="relu"))
    model.add(
        Dense(
            1,
            name="Dense C",
            activation="linear",
        )
    )

    model.summary()

    model.compile(
        loss="mse",
        metrics=["mse", "mae"],
        initializer="glorot",
    )

    history = model.fit(X, y, epochs=100, lr=0.01, batch_size=32, verbose=2)
    # history_df = pd.DataFrame(history)
    # width = 6
    # height = 6 * len(history_df.columns)
    # history_df.plot(subplots=True, figsize=(width, height))
    # plt.tight_layout()
    for i, j in zip(coef, model[-1].weight[0]):
        print(i, j)
    # plt.show()
    # print("Learned Coefficients")
    # print(model[-1].weight)
    # print("True Coefficients")
    # print(coef)


if __name__ == "__main__":
    # multi_class()
    # classification(classes=(0, 1))
    regression()
