import numpy as np


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


class BatchGradientDescent:
    """
    A class which implements gradient descent for multiliear regression. Other classes may inherit from this class
    """

    def __init__(self, fit_intercept=True, tol=None):
        self._fit_intercept = fit_intercept
        self._tol = tol
        self._losses = []

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

    def _epoch_progress(self, epoch, epochs, verbose):
        """
        Print the progress of the epoch
        The formate is 1000/1000 [====================] 100.0%
        """
        if verbose < 0:
            return
        arrow_length = 20
        spaces = arrow_length - int((epoch / epochs) * arrow_length)
        filled = arrow_length - spaces
        progress = f"[{'=' * filled}{' ' * spaces}]"
        percentage = f"{((epoch+1) / epochs) * 100:.1f}%"
        epoch = f"{epoch+1}/{epochs}"
        string = f"{Style.CYAN}{epoch} {progress} {percentage}{Style.END}"
        print(string, end="\r", flush=True)

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
            y_hat = self._get_yhat(X, self.weights)
            loss = self._get_loss(y_hat, y)
            print(
                f"{Style.BLUE}\nEpoch: {epoch+1:5} => Loss: {loss:10.6f}{Style.END}",
                flush=False,
            )

    def score(self, X, y):
        """
        Creates the R2 score of the model
        """
        try:
            weights = self.weights
        except AttributeError:
            raise AttributeError("Model has not been fitted yet")
        y_hat = self.predict(X)
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)
        return 1 - (ss_res / ss_tot)

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
        loss = 1e-6
        for i in range(epochs):
            self._epoch_progress(i, epochs, verbose=verbose)
            self._loss_progress(X, y, i, verbose)
            y_hat = self._get_yhat(X, self.weights)
            self.weights -= learning_rate * self._delJdelW(y_hat, y, X)

            if callback is not None:
                callback(self, self.weights, i)
            loss_new = self._get_loss(y_hat, y)
            self._losses.append(loss_new)
            if self._tol is not None:
                if (abs(loss_new - loss)) / loss < self._tol:
                    print(f"{Style.GREEN}Converged at epoch {i+1}{Style.END}")
                    print(f"Loss: {loss_new}")
                    return
                loss = loss_new

    def predict(self, X):
        X = self._preprocess(X)
        return self._get_yhat(X, self.weights)


class MiniBatchGradientDescent(BatchGradientDescent):
    def __init__(
        self,
        fit_intercept=True,
        tol=None,
    ):
        super().__init__(fit_intercept, tol)

    def __create_batch(self, X, y, batch_size):
        m = X.shape[0]
        if batch_size > 1:
            for i in range(0, m, batch_size):
                yield X[i : i + batch_size], y[i : i + batch_size]
        else:
            for i in range(m):
                ids = np.random.choice(m, 1, replace=False)
                yield X[ids], y[ids]

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
        batch_size : int, optional
            The batch size, by default 32
        verbose : int, optional
            The verbosity level, by default 1
        callback : function, optional
            A callback function, by default None

        Returns
        -------
        None

        """
        X = self._preprocess(X)
        self.__batch_size = batch_size
        self.weights = self._get_weights(n=X.shape[1])
        loss = 1e-6

        for i in range(epochs):
            self._epoch_progress(i, epochs, verbose=verbose)
            self._loss_progress(X, y, i, verbose)
            j = 1
            for X_batch, y_batch in self.__create_batch(X, y, self.__batch_size):
                y_hat = self._get_yhat(X_batch, self.weights)
                self.weights -= learning_rate * self._delJdelW(y_hat, y_batch, X_batch)
                j += 1

            if callback is not None:
                callback(self, self.weights, i)

            y_hat = self._get_yhat(X, self.weights)
            loss_new = self._get_loss(y_hat, y)
            self._losses.append(loss_new)

            if self._tol is not None:
                if (abs(loss_new - loss)) / loss < self._tol:
                    print(f"{Style.GREEN}Converged at epoch {i+1}{Style.END}")
                    print(f"Loss: {loss_new}")
                    return
                loss = loss_new


class LogisticGradientDescent(BatchGradientDescent):
    def __init__(self, fit_intercept=True, tol=None):
        super().__init__(fit_intercept, tol)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _get_loss(self, y_hat, y_true):
        m = len(y_hat)
        return np.sum(-y_true * np.log(y_hat) - (1 - y_true) * (np.log(1 - y_hat))) / m

    def _get_yhat(self, X, w):
        return self.sigmoid(np.dot(X, w.T))

    def score(self, X, y):
        """
        Calculates the accuracy of the model
        """
        try:
            self.weights
        except AttributeError:
            raise AttributeError("Model has not been fitted yet")
        y_hat = self.predict(X)
        y_hat = np.round(y_hat)
        return np.mean(y_hat == y)


class SteepestDescent(BatchGradientDescent):
    def __init__(self, fit_intercept=True, tol=None):
        super().__init__(fit_intercept, tol)

    def phi(self, alpha, weights, X, y):
        """
        Calculates the phi as a function of alpha

        phi(alpha) = J(w - alpha * delw)
        """
        y_hat = self._get_yhat(X, weights)
        delf = self._delJdelW(y_hat, y, X)
        w_new = weights - alpha * delf
        y_hat_new = self._get_yhat(X, w_new)
        return self._get_loss(y_hat_new, y)

    def dphi(self, alpha, weights, X, y):
        """
        Calculates the derivative of the phi function. Needed for the secant method
        """
        eps = 1e-5
        phi_plus = self.phi(alpha + eps, weights, X, y)
        phi_minus = self.phi(alpha - eps, weights, X, y)
        return (phi_plus - phi_minus) / (2 * eps)

    def secant(self, f, x0, x1, tol=1e-5, max_iter=1000, **kwargs):
        """
        Determines the minimum of a function using the secant method
        """
        x = x0
        x_prev = x1
        for i in range(max_iter):
            x_new = x - f(x, **kwargs) * (x - x_prev) / (
                f(x, **kwargs) - f(x_prev, **kwargs) + 1e-10
            )
            if abs(x_new - x) < tol:
                return x_new
            x_prev = x
            x = x_new
        return x_new

    def _get_lr(self, X, y, weights):
        """
        Determines the learning rate using the secant method

        Parameters
        ----------
        X : array-like
            The input data
        y : array-like
            The target data
        weights : array-like
            The weights

        Returns
        -------
        float
            The learning rate
        """
        raw_lr = self.secant(
            self.phi, 0.001, 0.999, weights=weights, X=X, y=y, tol=1e-5, max_iter=1000
        )

        if raw_lr < 1e-3:
            raw_lr = 1e-3
        elif raw_lr > 0.7:
            raw_lr = 0.7
        return raw_lr

    def fit(self, X, y, epochs=100, verbose=0, callback=None):
        """
        Fit the model to the data

        Parameters
        ----------
        X : array-like
            The input data
        y : array-like
            The target data
        epochs : int, optional
            The number of epochs, by default 100
        verbose : int, optional
            The verbosity level, by default 0
        callback : function, optional
            A callback function, by default None

        Returns
        -------
        None
        """

        X = self._preprocess(X)
        self.weights = self._get_weights(n=X.shape[1])
        loss = 1e-6
        for i in range(epochs):
            self._loss_progress(X, y, i, verbose)
            self._epoch_progress(i, epochs, verbose=verbose)
            y_hat = self._get_yhat(X, self.weights)

            weights = self.weights
            learning_rate = self._get_lr(X, y, weights)

            self.weights -= learning_rate * self._delJdelW(y_hat, y, X)

            loss_new = self._get_loss(y_hat, y)
            self._losses.append(loss_new)

            if callback is not None:
                callback(self, self.weights, i, learning_rate)
            if self._tol is not None:
                if (abs(loss_new - loss)) / loss < self._tol:
                    print(f"{Style.GREEN}Converged at epoch {i+1}{Style.END}")
                    print(f"Loss: {loss_new}")
                    return
                loss = loss_new
