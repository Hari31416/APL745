import numpy as np


class GradientDescent:
    """
    A class which implements gradient descent for multiliear regression
    """

    def __init__(self, fit_intercept=True, tol=None):
        self._fit_intercept = fit_intercept
        self._tol = tol

    def _delJdelW(self, y_hat, y_true, X):
        m = len(y_hat)
        res = np.matmul((y_hat - y_true), X)
        return res / m

    def _get_weights(self, n):
        w = np.random.random((n))
        return w

    def _get_yhat(self, X, w):
        return np.dot(X, w.T)

    def _get_loss(self, y_hat, y_true):
        m = len(y_hat)
        return np.sum((y_hat - y_true) ** 2) / (2 * m)

    def _preprocess(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self._fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.append(ones, X, axis=1)
        return X

    def _epoch_progress(self, epoch, epochs):

        arrow_length = 20
        spaces = arrow_length - int((epoch / epochs) * arrow_length)
        filled = arrow_length - spaces
        progress = f"[{'=' * filled}{' ' * spaces}]"
        percentage = f"{((epoch+1) / epochs) * 100:.1f}%"
        epoch = f"{epoch+1}/{epochs}"
        string = f"{epoch} {progress} {percentage}"
        print(string, end="\r", flush=True)

    def _progress(self, X, y, epoch, verbose):
        if not verbose:
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
            y_hat = self._get_yhat(X, self.weights)
            loss = self._get_loss(y_hat, y)
            print(f"\nEpoch: {epoch+1:5} => Loss: {loss:10.6f}", flush=False)

    def fit(self, X, y, learning_rate=0.01, epochs=100, verbose=1, callback=None):
        """
        Fit the model to the data

        Parameters
        ----------
        X : array-like
            The input data
        y : array-like
            The target data
        learning_rate : float, optional
            The learning rate, by default 0.01
        epochs : int, optional
            The number of epochs, by default 100
        verbose : int, optional
            The verbosity level, by default 1
        callback : function, optional
            A callback function, by default None

        Returns
        -------
        None
        """

        X = self._preprocess(X)
        self.weights = self._get_weights(n=X.shape[1])
        error = 1e10
        for i in range(epochs):
            self._epoch_progress(i, epochs)
            self._progress(X, y, i, verbose)
            y_hat = self._get_yhat(X, self.weights)
            self.weights -= learning_rate * self._delJdelW(y_hat, y, X)

            if callback is not None:
                callback(self, self.weights, i)
            if self._tol is not None:
                error_new = self._get_loss(y_hat, y)
                if (abs(error_new - error))/error < self._tol:
                    print(f"Converged at epoch {i+1}")
                    print(f"Error: {error_new}")
                    return
                error = error_new

    def predict(self, X):
        X = self._preprocess(X)
        return self._get_yhat(X, self.weights)


class BatchGradientDescent(GradientDescent):
    def __init__(
        self,
        fit_intercept=True,
        tol=None,
    ):
        super().__init__(fit_intercept, tol)

    def __create_batch(self, X, y, batch_size):
        m = X.shape[0]
        for i in range(0, m, batch_size):
            yield X[i : i + batch_size], y[i : i + batch_size]

    def fit(
        self,
        X,
        y,
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        verbose=0,
        callback=None,
    ):
        X = self._preprocess(X)
        self.__batch_size = batch_size
        self.weights = self._get_weights(n=X.shape[1])
        error = 1e10

        for i in range(epochs):
            self._epoch_progress(i, epochs)
            self._progress(X, y, i, verbose)
            j = 1
            for X_batch, y_batch in self.__create_batch(X, y, self.__batch_size):
                y_hat = self._get_yhat(X_batch, self.weights)
                self.weights -= learning_rate * self._delJdelW(y_hat, y_batch, X_batch)
                j += 1
            if self._tol is not None:
                y_hat = self._get_yhat(X, self.weights)
                error_new = self._get_loss(y_hat, y)
                if (abs(error_new - error))/error < self._tol:
                    print(f"Converged at epoch {i+1}")
                    print(f"Error: {error_new}")
                    return
                error = error_new
            if callback is not None:
                callback(self, self.weights, i)


class LogisticGradientDescent(GradientDescent):
    def __init__(self, fit_intercept=True, tol=None):
        super().__init__(fit_intercept, tol)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _get_loss(self, y_hat, y_true):
        m = len(y_hat)
        return np.sum(-y_true * np.log(y_hat) - (1 - y_true) * (np.log(1 - y_hat))) / m

    def _get_yhat(self, X, w):
        return self.sigmoid(np.dot(X, w.T))
