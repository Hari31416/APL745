import numpy as np
from scipy import optimize
from functools import partial
import time


class Style:
    """
    A class which contains the color codes for printing in color
    """

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    END = "\033[0m"
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class SoftmaxClassifier:
    def __init__(self, fit_intercept):
        self.fit_intercept = fit_intercept
        self.W = None
        self._fitted = False
        self.losses = []
        self.scores = []

    def _preprocess(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.append(ones, X, axis=1)
        return X

    def second_to_natural(self, t):
        """
        Convert seconds to natural time
        """
        time_string = ""
        hours = int(t / 3600)
        minutes = int((t - hours * 3600) / 60)
        seconds = int(t - hours * 3600 - minutes * 60)
        if hours > 0:
            time_string += f"{hours:02d}h "
        time_string += f"{minutes:02d}m "
        time_string += f"{seconds:02d}s"
        return time_string

    def _epoch_progress(self, epoch, epochs, verbose, time):
        """
        Print the progress of the epoch
        The formate is 1000/1000 [====================] 100.0% 00h 00m 00s
        """
        if verbose < 0:
            return
        arrow_length = 20
        spaces = arrow_length - int((epoch / epochs) * arrow_length)
        filled = arrow_length - spaces
        progress = f"[{'=' * filled}{' ' * spaces}]"
        percentage = f"{((epoch+1) / epochs) * 100:.1f}%"
        if epoch == 0:
            self.time_text = "Estimating..."
        if (epoch + 1) % 10 == 0:
            time_remaining = (epochs - epoch) * time
            self.time_text = self.second_to_natural(time_remaining)

        epoch = f"{epoch+1:4d}/{epochs:4d}"
        string = (
            f"{Style.CYAN}{epoch} {progress} {percentage} {self.time_text} {Style.END}"
        )
        print(string, end="\r", flush=False)

    def _loss_progress(self, X, y, epoch, verbose):
        """
        Print the loss progress of the epoch
        The formate is Epoch: 1000 => Loss: 0.000000
        """
        if verbose < 1:
            return
        elif verbose == 1:
            i = 1000
        elif verbose == 2:
            i = 100
        elif verbose == 3:
            i = 10
        else:
            i = 1
        if (epoch + 1) % i == 0:
            loss = self.loss(X, y)
            print(
                f"{Style.BLUE}\nEpoch: {epoch+1:5} => Loss: {loss:10.6f}{Style.END}",
                flush=False,
            )

    def softmax(self, z):
        """
        Compute the softmax function for each row of the input z.

        Parameters
        ----------
        z : array_like
            A numpy array of shape (m, n) where m is the number of examples and n is
            the number of classes.

        Returns
        -------

        s : array_like
            A numpy array equal to the softmax of z of shape (m, n)
        """
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def _predict_proba(self, X):
        return self.softmax(np.dot(X, self.W))

    def loss(self, X, y):
        m = y.size
        h = self._predict_proba(X)
        J = (1 / (2 * m)) * np.sum(-y * np.log(h + 1e-11)) + (
            self.lambda_ / (2 * m)
        ) * np.sum(self.W[1:, :] ** 2)
        return J

    def _grad(self, X, y):
        m = y.size
        h = self._predict_proba(X)
        grad = (1 / m) * np.dot(X.T, (h - y)) + (self.lambda_ / m) * self.W
        grad[0, :] = 1 / m * np.dot(X.T, (h - y))[0, :]
        return grad

    def one_hot(y, n_classes):
        """
        Converts a vector of labels into a one-hot matrix.

        Parameters
        ----------
        y : array_like
            An array of shape (m, ) that contains labels for X. Each value in y
            should be an integer in the range [0, n_classes).

        n_classes : int
            The number of classes.

        Returns
        -------
        one_hot : array_like
            An array of shape (m, n_classes) where each row is a one-hot vector.
        """
        m = y.shape[0]
        one_hot = np.zeros((m, n_classes))
        for i in range(m):
            one_hot[i, y[i]] = 1
        return one_hot

    def fit(self, X, y, epochs=100, lr=0.1, lambda_=0.1, verbose=0, tol=None):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array_like
            An array of shape (m, n) that contains the training data.
        y : array_like
            An array of shape (m, ) that contains the training labels.
        epochs : int
            The number of epochs to train for.
        lr : float
            The learning rate.
        lambda_ : float
            The regularization parameter.
        verbose : int
            The verbosity level. 0 is silent, 1 is per epoch, 2 is per 100
            epochs, 3 is per 10 epochs, and 4 is per epoch.
        tol : float
            The tolerance for the loss. If the loss is less than the tolerance,
            the training will stop early.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._preprocess(X)
        self.lambda_ = lambda_
        _, n = X.shape
        self.W = np.random.randn(n, y.shape[1])
        loss = 1e-6
        toc = None
        for i in range(epochs):
            tic = time.time()
            if toc is not None:
                t = tic - toc
            else:
                t = 0
            # print(t, type(t))
            self._epoch_progress(i, epochs, verbose=verbose, time=t)
            toc = tic
            self._loss_progress(X, y, i, verbose)
            self.W -= lr * self._grad(X, y)
            loss_new = self.loss(X, y)
            self.losses.append(loss_new)

            y_hat = np.argmax(self._predict_proba(X), axis=1)
            y_true = np.argmax(y, axis=1)
            score = np.mean(y_hat == y_true)
            self.scores.append(score)

            if tol is not None:
                if (abs(loss_new - loss)) / loss < tol:
                    print(f"{Style.GREEN}Converged at epoch {i+1}{Style.END}")
                    print(f"Loss: {loss_new}")
                    return
                loss = loss_new

        self._fitted = True
        return self

    def predict(self, X):
        """
        Predict the labels for the data.

        Parameters
        ----------
        X : array_like
            An array of shape (m, n) that contains the data.

        Returns
        -------
        y : array_like
            An array of shape (m, ) that contains the predicted labels.
        """
        X = self._preprocess(X)
        return np.argmax(self._predict_proba(X), axis=1)

    def score(self, X, y):
        """
        Compute the accuracy of the model.

        Parameters
        ----------
        X : array_like
            An array of shape (m, n) that contains the data.
        y : array_like
            An array of shape (m, ) that contains the labels.

        Returns
        -------
        score : float
            The accuracy of the model.
        """
        return np.mean(self.predict(X) == y)
