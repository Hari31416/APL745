import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")

if device == "cuda":
    print(torch.cuda.get_device_name())


def dnn(layers):
    """Creates a deep neural network with given layers

    Arguments:
        layers {list} -- list of layers

    Returns:
        nn.Sequential -- deep neural network
    """
    net = nn.Sequential()
    for i in range(len(layers) - 1):
        net.add_module("linear{}".format(i), nn.Linear(layers[i], layers[i + 1]))
        net.add_module("relu{}".format(i), nn.ReLU())
    return net


class DeepONet(nn.Module):
    """DeepONet model"""

    def __init__(self, branch_layers, trunk_layers):
        """Initializes DeepONet model

        Arguments:
            branch_layers {list} -- list of layers for branch
            trunk_layers {list} -- list of layers for trunk

        """
        super(DeepONet, self).__init__()
        # A bias parameter is important for the model to perform well
        self.bias = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.branch = dnn(branch_layers)
        self.trunk = dnn(trunk_layers)

    def forward(self, u, y):
        """Forward pass for DeepONet"""
        B = self.branch(u)
        T = self.trunk(y)
        output = torch.einsum("bi,bi->b", B, T)
        output = torch.unsqueeze(output, 1)
        output = output + self.bias
        return output

    def predict(self, u_star, y_star):
        """Predicts output for given input"""
        with torch.no_grad():
            s = self.forward(u_star, y_star)
            s = s.detach().cpu().numpy()
        return s

    def summary(self):
        print(self)


class TrainDeepONet:
    """Trains DeepONet model"""

    def __init__(self, model):
        """Initializes TrainDeepONet

        Arguments:
            model {DeepONet} -- DeepONet model
        """
        super(TrainDeepONet, self).__init__()
        self.model = model

    def batch_dataset(self, batch_size, us, ys, ss):
        """Batches dataset for DeepONet"""
        us = torch.tensor(us, dtype=torch.float32).to(device)
        ys = torch.tensor(ys, dtype=torch.float32).to(device)
        ss = torch.tensor(ss, dtype=torch.float32).to(device)
        train_dataset = torch.utils.data.TensorDataset(us, ys, ss)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        return train_dataloader

    def loss(self, y_pred, y):
        """Loss function for DeepONet"""
        return torch.mean((y_pred - y) ** 2)

    def load_dataset(self, path):
        """Loafds the dataset from given path"""
        data = np.load(path, allow_pickle=True)
        y = data["X"][1].astype(np.float32)  # output locations (100,1)
        u = data["X"][0].astype(np.float32)  # input functions (150,100)
        s = data["y"].astype(np.float32)  # output functions (150,100)
        return u, y, s

    def create_dataset(self, u, y, s):
        """Create dataset for DeepONet

        Arguments:
            u {np.ndarray} -- input functions (m,n)
            y {np.ndarray} -- output locations (n,1)
            s {np.ndarray} -- output functions (m,n)

        Returns:
            np.ndarray -- input functions (m*n,m)
            np.ndarray -- output locations (m*n,1)
            np.ndarray -- output functions (m*n,1)

        """
        u = u.T
        s = s.T
        y = y.reshape(-1, 1)
        m, n = u.shape
        us = np.zeros((m * n, m))
        ys = np.zeros((m * n, 1))
        ss = np.zeros((m * n, 1))
        for i in range(n):
            for j in range(m):
                us[i * m + j, :] = u[:, i]
                ys[i * m + j, :] = y[j]
                ss[i * m + j, :] = s[j, i]
        return us, ys, ss

    def preprocess_dataset(self, path, batch_size):
        """Preprocesses the dataset to be used by DeepONet"""
        u, y, s = self.load_dataset(path)
        us, ys, ss = self.create_dataset(u, y, s)
        batched_dataset = self.batch_dataset(
            batch_size=batch_size,
            us=us,
            ys=ys,
            ss=ss,
        )
        return batched_dataset

    def train(self, epochs, batch_size, train_data_path, optimizer):
        """Trains the DeepONet model

        Arguments:
            epochs {int} -- number of epochs
            batch_size {int} -- batch size
            train_data_path {str} -- path to the training dataset
            optimizer {torch.optim} -- optimizer

        Returns:
            list -- list of losses
        """
        losses = []
        data = self.preprocess_dataset(train_data_path, batch_size)
        verbose_freq = epochs // 10
        verbose_freq = max(verbose_freq, 1)

        for epoch in range(epochs):
            l_total = 0
            for u, y, s in data:
                self.model.train()
                optimizer.zero_grad()
                y_pred = self.model(u, y)
                l = self.loss(y_pred, s)
                l_total += l.item()
                l.backward()
                optimizer.step()
            l_total = l_total / len(data)
            losses.append(l_total)

            if epoch % verbose_freq == 0:
                print("Epoch: {0:>3d}, Loss: {1:6f}".format(epoch, l_total))
            l_total = 0
        return losses
