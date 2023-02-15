import numpy as np

ALL = [
    "glorot",
    "he",
    "xavier",
    "random",
    "uniform",
    "zeros",
]


class Initializer:
    def __init__(self) -> None:
        pass

    def glorot(self):
        sigma = np.sqrt(2 / (self.n_cur + self.n_prev))
        return np.random.normal(0, sigma, self.shape)

    def he(self):
        sigma = np.sqrt(2 / self.n_prev)
        return np.random.normal(0, sigma, self.shape)

    def xavier(self):
        sigma = np.sqrt(1 / self.n_prev)
        return np.random.normal(0, sigma, self.shape)

    def random(self):
        return np.random.random(self.shape)

    def uniform(self, min, max):
        return np.random.uniform(min, max, self.shape)

    def zeros(self):
        return np.zeros(self.shape)

    def __call__(self, shape, initializer="glorot", **kwargs):
        self.shape = shape
        self.n_cur = self.shape[0]
        self.n_prev = self.shape[1]
        if initializer == "glorot":
            weight = self.glorot()
        elif initializer == "he":
            weight = self.he()
        elif initializer == "xavier":
            weight = self.xavier()
        elif initializer == "random":
            weight = self.random()
        elif initializer == "uniform":
            if kwargs == {}:
                kwargs = {"min": -1, "max": 1}
            weight = self.uniform(**kwargs)
        elif initializer == "zeros":
            weight = self.zeros()
        else:
            raise ValueError(f"Invalid initializer. Please use one of {ALL}")
        weight = weight.astype(np.float32)
        return weight
