from nnet.layers import *
from nnet.losses import *
from nnet.scores import *
from nnet.utils import *
from nnet.initializers import *
from tqdm import tqdm
import numpy as np
from tabulate import tabulate

NON_TRAIANBLE_LAYERS = [
    Dropout,
    MaxPool2D,
    AveragePool2D,
    Flatten,
    Input,
]
PLACEHOLDER = "--" * 3


class Sequential:
    def __init__(self, name="Sequential") -> None:
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

    def _calc_output_shapes(self):
        output_shapes = [self[0].input_shape]
        for i in range(1, len(self)):
            out = self[i]._output_shape(output_shapes[i - 1])
            output_shapes.append(out)
        return output_shapes

    def _calc_input_shapes(self, output_shapes=None):
        input_shapes = [self[0].input_shape]
        for i in range(1, len(self)):
            in_ = output_shapes[i - 1]
            input_shapes.append(in_)
        return input_shapes

    def _calc_W_b_shapes(self):
        output_shapes = self._calc_output_shapes()
        input_shapes = self._calc_input_shapes(output_shapes=output_shapes)
        Ws = [None]
        bs = [None]
        for i in range(1, len(self)):
            W, b = self[i]._calc_W_b_shape(input_shapes[i], output_shapes[i])
            Ws.append(W)
            bs.append(b)
        return Ws, bs

    def _calc_num_params(self):
        Ws, bs = self._calc_W_b_shapes()
        num_params = [0]
        for i in range(1, len(self)):
            num_params.append(self[i]._num_params(Ws[i], bs[i]))
        return num_params

    def __info(self, print_too=True):
        Ws, bs = self._calc_W_b_shapes()
        output_shapes = self._calc_output_shapes()
        input_shapes = self._calc_input_shapes(output_shapes=output_shapes)
        names = self.__names
        parameters = self._calc_num_params()
        total_parameters = sum(parameters)

        table = []

        for i in range(len(Ws)):
            temp_dict = {
                "name": names[i],
                "input": input_shapes[i],
                "output": output_shapes[i],
                "W": Ws[i],
                "b": bs[i],
                "params": parameters[i],
            }
            table.append(temp_dict)
        self.info = table
        if print_too:
            headers = {
                "name": "Name",
                "input": "Input Shapes",
                "output": "Output Shapes",
                "W": "Weight Shapes",
                "b": "Bias Shapes",
                "params": "# Parameters",
            }
            tabulate_res = tabulate(
                table,
                headers=headers,
                tablefmt="fancy_grid",
                missingval="None",
            )
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
        initializer = initializer.lower()
        init = Initializer()
        return init(shape, initializer)

    def summary(self):
        if len(self) == 0:
            raise ValueError("Model has no layers.")
        if not isinstance(self.layers[0], Input):
            raise AttributeError("First layer must be of type `Input`")
        self.__info()

    def compile(self, loss: Loss, metrics: list[str] = [], initializer="glorot"):
        self._loss_function = parse_loss(loss)
        if self._loss_function is MeanSquaredLoss:  # TODO: Fix this
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
            if w_shape is None:
                continue
            b_shape = self.info[i]["b"]
            self[i].weight = self.__create_weights(w_shape, initializer=initializer)
            self[i].bias = self.__create_weights(b_shape, "zeros")

    def _forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def __cal_delta_L(self, y_true):
        output_layer = self[-1]
        y_hat = output_layer.output
        loss_derivative = self.loss.derivative(y_true, y_hat)
        dAdZ = output_layer.activation.derivative(output_layer.Z)
        delta_L = loss_derivative * dAdZ
        return delta_L

    def _backward(self, y_true):
        delta_L = self.__cal_delta_L(y_true)
        reverse_layers_nums = range(len(self) - 1, 0, -1)
        for i in reverse_layers_nums:
            delta_L = self[i].backward(delta_L)

    def _update_parameters(self):
        for layer in self.layers[1:]:
            layer.update(self.lr)

    def __create_batch(self, X, y, batch_size):
        m = X.shape[-1]
        if batch_size > 1:
            for i in range(0, m, batch_size):
                yield X[..., i : i + batch_size], y[..., i : i + batch_size]
        else:
            for i in range(m):
                ids = np.random.choice(m, 1, replace=False)
                yield X[..., ids], y[..., ids]

    def __pregress_bar(self, epoch, batches=None):
        if batches:
            length = batches
        else:
            length = self.epochs
        arrow_length = 20
        spaces = arrow_length - int((epoch / length) * arrow_length)
        filled = arrow_length - spaces
        progress = f"[{'=' * filled}{' ' * spaces}]"
        percentage = f"{((epoch+1) / length) * 100:.1f}%"
        cur = f"{epoch+1}"
        all_epochs = f"{length}"
        if batches:
            epoch = f"Batch {cur.zfill(4)}/{all_epochs.zfill(4)}"
        else:
            epoch = f"Epoch {cur.zfill(4)}/{all_epochs.zfill(4)}"
        string = f"{epoch} {progress} {percentage}  "
        return string

    def __get_y_pred(self, y_hat, y_true):
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

    def __progress(self, X, y, epoch, verbose, batches=None):

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
            if batches is not None:
                string = self.__pregress_bar(epoch, batches=batches)
                print(string, end="\r")
                return
            else:
                cur = f"{epoch+1}"
                all_epochs = f"{self.epochs}"
                text = (
                    f"Epoch {cur.zfill(4)}/{all_epochs.zfill(4)} | Loss: {loss:.5f} | "
                )
                text += metric_text
                print(text)

    def fit(self, X, y, epochs=100, verbose=1, lr=0.01, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.input = X
        for i in range(epochs):
            num_batches = int(np.ceil(X.shape[-1] / batch_size))
            cur_batch = 0
            for X_batch, y_batch in self.__create_batch(X, y, batch_size=batch_size):
                self._forward(X_batch)
                self._backward(y_batch)
                self._update_parameters()
                if np.nan in (self[-1].output):
                    raise ValueError("NAN in output")
                self.__progress(
                    X_batch, y_batch, cur_batch, verbose=verbose, batches=num_batches
                )
                cur_batch += 1

            self.__progress(X, y, i, verbose)

        return self.history

    def predict(self, X):
        self._forward(X)
        return self[-1].output

    def evaluate(self, X, y, metric="Accuracy"):
        y_pred = self.predict(X)
        y_pred, y_true = self.__get_y_pred(y_pred, y)
        metric = parse_metric(metric)
        return metric.score(y_true, y_pred)
