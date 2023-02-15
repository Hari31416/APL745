from nn.layers import *
from nn.losses import *
from nn.scores import *
from nn.utils import *
from nn.initializers import *
from tqdm import tqdm
import numpy as np
from tabulate import tabulate

NON_TRAIANBLE_LAYERS = [Dropout, BatchNormalization]
PLACEHOLDER = "--" * 3


class Sequential:
    """
    The Sequential class is a container for layers.

    It provides a forward and backward pass for the layers in the container.

    Parameters
    ----------
    name : str, optional
        The name of the model, by default "Sequential"
    """

    def __init__(self, name="Sequential") -> None:
        """
        The Sequential class is a container for layers.

        It provides a forward and backward pass for the layers in the container.

        Parameters
        ----------
        name : str, optional
            The name of the model, by default "Sequential"

        Returns
        -------
        Sequential
        """
        self.input = None
        self.name = name
        self.layers = []
        self.trainable_layers = []
        self.info = {}
        self.lr = 0.01
        self.__names = []
        self.__j = 1

    def __repr__(self) -> str:
        return f"Sequential({self.name})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def add(self, layer: Layer):
        """
        Add a layer to the model

        Parameters
        ----------
        layer : Layer
            The layer to add

        Raises
        ------
        AttributeError
            If the first layer is not of type `Input`
        ValueError
            If two layers have the same name

        Returns
        -------
        None
        """
        if len(self) == 0 and not isinstance(layer, Input):
            raise AttributeError("First layer must be of type `Input`")

        name = layer.name
        if name is not None:
            if name in self.__names:
                raise ValueError("Two layers can not have the same name.")
            else:
                self.__names.append(layer.name)
                self.layers.append(layer)
        else:
            self.__names.append(f"{layer.__class__.__name__}_{self.__j}")
            self.layers.append(layer)
            self.__j += 1

    def __shape_list(self) -> list[int]:
        """
        Calculate the number of neurons in each layer
        """
        neurons = []
        neurons.append(self[0].input_shape[0])
        for i in range(1, len(self)):
            for layer in NON_TRAIANBLE_LAYERS:
                if isinstance(self[i], layer):
                    self[i].neurons = self[i - 1].neurons
        for i in range(1, len(self)):
            neurons.append(self[i].neurons)
        self.__neurons = neurons
        return neurons

    def __cal_W_b_shapes(self):
        """
        Calculate the shapes of the weights and biases
        """
        shape_list = self.__shape_list()
        Ws = [PLACEHOLDER]
        bs = [PLACEHOLDER]
        for i in range(1, len(self)):
            if isinstance(self[i], Dense):
                Ws.append((shape_list[i], shape_list[i - 1]))
                bs.append((shape_list[i], 1))
            else:
                Ws.append(PLACEHOLDER)
                bs.append(PLACEHOLDER)
        return Ws, bs

    def __parameters(self):
        """
        Calculate the number of parameters in each layer
        """
        Ws, bs = self.__cal_W_b_shapes()
        parameters = [0]
        for i in range(1, len(self)):
            if Ws[i] == PLACEHOLDER:
                parameters.append(0)
                continue
            a, b = Ws[i]
            c, _ = bs[i]
            p = a * b + c
            parameters.append(p)
        return parameters

    def __info(self, print_too=True):
        """
        Creates a summary of the model

        Parameters
        ----------
        print_too : bool, optional
            Whether to print the summary or not, by default True

        Returns
        -------
        list[dict]
            A list of dictionaries containing the summary of the model
        """
        Ws, bs = self.__cal_W_b_shapes()
        neurons = self.__neurons
        names = self.__names
        parameters = self.__parameters()
        total_parameters = sum(parameters)

        table = []

        for i in range(len(Ws)):
            temp_dict = {
                "name": names[i],
                "neurons": neurons[i],
                "W": Ws[i],
                "b": bs[i],
                "params": parameters[i],
                "output": (neurons[i],),
            }
            table.append(temp_dict)
        self.info = table
        if print_too:
            headers = {
                "name": "Name",
                "neurons": "# Neurons",
                "W": "Weight Shapes",
                "b": "Bias Shapes",
                "params": "# Parameters",
                "output": "Output Shapes",
            }
            tabulate_res = tabulate(table, headers=headers)
            table_row_len = len(tabulate_res.split("\n")[1])
            table_res = "Model: " + self.name + "\n"
            table_res += "_" * table_row_len + "\n"
            table_res += tabulate_res
            table_res += "\n"
            table_res += "=" * table_row_len
            table_res += f"\nTotal Parameters: {total_parameters}\n"
            table_res += "_" * table_row_len
            print(table_res)
        return self.info

    def __create_weights(self, shape, initializer):
        """
        Create weights for the layers

        Parameters
        ----------
        shape : tuple[int]
            The shape of the weights
        initializer : str
            The initializer to use

        Returns
        -------
        np.ndarray
            The weights
        """
        initializer = initializer.lower()
        init = Initializer()
        return init(shape, initializer)

    def summary(self):
        """
        Print a summary of the model

        Raises
        ------
        AttributeError
            If the first layer is not of type `Input`
        ValueError
            If the model has no layers

        Returns
        -------
        None
        """
        if len(self) == 0:
            raise ValueError("Model has no layers.")
        if not isinstance(self.layers[0], Input):
            raise AttributeError("First layer must be of type `Input`")
        self.__info()

    def compile(self, loss: str, metrics: list[str] = [], initializer="glorot"):
        """
        Compile the model

        Parameters
        ----------
        loss : str
            The loss function to use
        metrics : list[str], optional
            The metrics to use, by default []
        initializer : str, optional
            The initializer to use, by default "glorot"

        Returns
        -------
        None
        """
        self._loss_function = parse_loss(loss)
        if self._loss_function is MeanSquaredLoss:
            self.__task = "regression"
        else:
            self.__task = "classification"
        self.loss = self._loss_function()
        if self.info != {}:
            self.__info(print_too=False)

        if len(metrics):
            self.metrics = [parse_metric(metric) for metric in metrics]
            self.history = {str(self.loss): []}
            for metric in self.metrics:
                self.history[str(metric)] = []

        for i in range(1, len(self)):
            w_shape = self.info[i]["W"]
            if w_shape == PLACEHOLDER:
                continue
            b_shape = self.info[i]["b"]
            self[i].weight = self.__create_weights(w_shape, initializer=initializer)
            self[i].bias = self.__create_weights(b_shape, "zeros")

    def _forward(self, X):
        """
        Do a forward pass

        Parameters
        ----------
        X : np.ndarray
            The input data

        Returns
        -------
        np.ndarray
            The output of the model
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def __cal_delta_L(self, y_true):
        """
        Calculate the delta of the last layer

        Parameters
        ----------
        y_true : np.ndarray
            The true labels

        Returns
        -------
        np.ndarray
            The delta of the last layer
        """
        output_layer = self[-1]
        y_hat = output_layer.output
        loss_derivative = self.loss.derivative(y_true, y_hat)
        dAdZ = output_layer.activation.derivative(output_layer.Z)
        delta_L = loss_derivative * dAdZ
        return delta_L

    def _backward(self, y_true):
        """
        Do a backward pass

        Parameters
        ----------
        y_true : np.ndarray
            The true labels

        Returns
        -------
        None
        """
        delta_L = self.__cal_delta_L(y_true)
        reverse_layers_nums = range(len(self) - 1, 0, -1)
        for i in reverse_layers_nums:
            delta_L = self[i].backward(delta_L)

    def _update_parameters(self):
        """
        Update the parameters of the model

        Returns
        -------
        None
        """
        for layer in self.layers[1:]:
            layer.update(self.lr)

    def __create_batch(self, X, y, batch_size):
        """
        Create a batch

        Parameters
        ----------
        X : np.ndarray
            The input data
        y : np.ndarray
            The true labels
        batch_size : int
            The size of the batch

        Yields
        -------
        tuple[np.ndarray, np.ndarray]
            The batch
        """
        m = X.shape[1]
        if batch_size > 1:
            for i in range(0, m, batch_size):
                yield X[:, i : i + batch_size], y[:, i : i + batch_size]
        else:
            for i in range(m):
                ids = np.random.choice(m, 1, replace=False)
                yield X[:, ids], y[:, ids]

    def __pregress_bar(self, epoch):
        """
        Return a progress bar

        Parameters
        ----------
        epoch : int
            The current epoch

        Returns
        -------
        str
            The progress bar
        """
        arrow_length = 20
        spaces = arrow_length - int((epoch / self.epochs) * arrow_length)
        filled = arrow_length - spaces
        progress = f"[{'=' * filled}{' ' * spaces}]"
        percentage = f"{((epoch+1) / self.epochs) * 100:.1f}%"
        cur = f"{epoch+1}"
        all_epochs = f"{self.epochs}"
        epoch = f"Epoch {cur.zfill(4)}/{all_epochs.zfill(4)}"
        string = f"{epoch} {progress} {percentage}"
        return string

    def __progress(self, X, y, epoch, verbose):
        """
        Print the progress of the model

        Parameters
        ----------
        X : np.ndarray
            The input data
        y : np.ndarray
            The true labels
        epoch : int
            The current epoch
        verbose : int
            The verbosity level
            - If 0, onlt a progress bar will be printed
            - If 1, the loss and metrics will be printed after 100 epoch
            - If 2, the loss and metrics will be printed after 10 epoch
            - If 3, the loss and metrics will be printed after 1 epoch

        Raises
        ------
        ValueError
            If the loss is exploding

        Returns
        -------
        None
        """
        y_hat = self._forward(X)
        loss = self.loss.loss(y, y_hat)
        if np.isnan(loss):
            raise ValueError("Loss is exploding", loss)
        self.history[str(self.loss)].append(loss)

        if len(self.metrics):
            y_pred, y_true = self.__get_y_pred(y_hat, y)

            metrics_value = []
            for metric in self.metrics:
                val = metric.score(y_true, y_pred)
                metrics_value.append(val)
                self.history[str(metric)].append(val)
            metrics = zip(self.metrics, metrics_value)
            metric_text = ""
            for metric in metrics:
                metric_text += f"{metric[0]}: {metric[1]:.5f} | "

        if verbose < 0:
            return
        elif verbose == 0:
            string = self.__pregress_bar(epoch)
            string += f" - Loss: {loss:.4f}"
            print(string, end="\r")
            return
        elif verbose == 1:
            i = 100
        elif verbose == 2:
            i = 10
        else:
            i = 1

        if (epoch + 1) % i == 0 or epoch == 0:
            cur = f"{epoch+1}"
            all_epochs = f"{self.epochs}"
            text = f"Epoch {cur.zfill(4)}/{all_epochs.zfill(4)} | Loss: {loss:.5f} | "
            text += metric_text
            print(text)

    def __get_y_pred(self, y_hat, y_true):
        """
        Convert the predictions to the correct format (1, m) for progess

        Parameters
        ----------
        y_hat : np.ndarray
            The predictions
        y_true : np.ndarray
            The true labels

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The predictions and the true labels

        """
        if self.__task == "regression":
            y_hat = y_hat
            y_true = y_true
        else:
            if y_hat.shape[0] == 1:
                y_hat = np.where(y_hat > 0.5, 1, 0)
                y_true = np.where(y_true > 0.5, 1, 0)
            else:
                y_hat = np.argmax(y_hat, axis=0).reshape(1, -1)
                y_true = np.argmax(y_true, axis=0).reshape(1, -1)

        return y_hat, y_true

    def fit(self, X, y, epochs=100, verbose=1, lr=0.01, batch_size=32):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : numpy.ndarray
            The input data. Shape (n_features, n_samples)
        y : numpy.ndarray
            The target data. Shape (n_classes, n_samples)
        epochs : int, optional
            The number of epochs to train the model, by default 100
        verbose : int, optional
            The verbosity level, by default 1
            - If 0, onlt a progress bar will be printed
            - If 1, the loss and metrics will be printed after 100 epoch
            - If 2, the loss and metrics will be printed after 10 epoch
            - If 3, the loss and metrics will be printed after 1 epoch
        lr : float, optional
            The learning rate, by default 0.01
        batch_size : int, optional
            The batch size, by default 32

        Returns
        -------
        dict
            A dictionary containing the loss and metrics for each epoch
        """
        self.lr = lr
        self.epochs = epochs
        self.input = X
        for i in range(epochs):
            for X_batch, y_batch in self.__create_batch(X, y, batch_size=batch_size):
                self._forward(X_batch)
                self._backward(y_batch)
                self._update_parameters()
                if np.nan in (self[-1].output):
                    raise ValueError("NAN in output")

            self.__progress(X, y, i, verbose)

        return self.history

    def predict(self, X):
        """
        Predict the labels for the input data

        Parameters
        ----------
        X : np.ndarray
            The input data

        Returns
        -------
        np.ndarray
            The predictions
        """
        self._forward(X)
        return self[-1].output
