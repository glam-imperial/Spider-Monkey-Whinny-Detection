import numpy as np


class Mixup():
    def __init__(self, batch_size, mixup_alpha):
        self.batch_size = batch_size

        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(0)

        self.mixup_lambdas = None

    def _make_lambda(self):
        if self.mixup_lambdas is None:
            self.mixup_lambdas = np.zeros((self.batch_size // 2, 2), dtype=np.float32)

            for i in range(self.batch_size, 2):
                lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
                self.mixup_lambdas[i, 0] = lam
                self.mixup_lambdas[i, 1] = 1.0 - lam

    def mixup_data(self, data):
        self._make_lambda()

        effective_batch = data.shape[0] // 2
        remainder_batch = data.shape[0] % 2

        if len(data.shape) == 1:
            mixup_lambdas_0 = self.mixup_lambdas[:effective_batch, 0]
            mixup_lambdas_1 = self.mixup_lambdas[:effective_batch, 1]
            mixed_data = data[0:effective_batch*2:2] * mixup_lambdas_0 + \
                         data[1:(effective_batch*2)+1:2] * mixup_lambdas_1
            if remainder_batch > 0:
                mixed_data = np.concatenate([mixed_data, data[-1]])
        elif len(data.shape) == 2:
            mixup_lambdas_0 = self.mixup_lambdas[:effective_batch, 0].reshape((effective_batch, 1))
            mixup_lambdas_1 = self.mixup_lambdas[:effective_batch, 1].reshape((effective_batch, 1))
            mixed_data = data[0:effective_batch*2:2, :] * mixup_lambdas_0 + \
                         data[1:(effective_batch*2)+1:2, :] * mixup_lambdas_1
            if remainder_batch > 0:
                mixed_data = np.vstack([mixed_data, data[-1].reshape(([1,] + list(data.shape)[1:]))])
        elif len(data.shape) == 3:
            mixup_lambdas_0 = self.mixup_lambdas[:effective_batch, 0].reshape((effective_batch, 1, 1))
            mixup_lambdas_1 = self.mixup_lambdas[:effective_batch, 1].reshape((effective_batch, 1, 1))
            mixed_data = data[0:effective_batch*2:2, :, :] * mixup_lambdas_0 + \
                         data[1:(effective_batch*2)+1:2, :, :] * mixup_lambdas_1
            if remainder_batch > 0:
                mixed_data = np.vstack([mixed_data, data[-1].reshape(([1,] + list(data.shape)[1:]))])
        elif len(data.shape) == 3:
            mixup_lambdas_0 = self.mixup_lambdas[:effective_batch, 0].reshape((effective_batch, 1, 1, 1))
            mixup_lambdas_1 = self.mixup_lambdas[:effective_batch, 1].reshape((effective_batch, 1, 1, 1))
            mixed_data = data[0:effective_batch*2:2, :, :, :] * mixup_lambdas_0 + \
                         data[1:(effective_batch*2)+1:2, :, :, :] * mixup_lambdas_1
            if remainder_batch > 0:
                mixed_data = np.vstack([mixed_data, data[-1].reshape(([1,] + list(data.shape)[1:]))])
        else:
            raise ValueError("Invalid data shape.")

        return mixed_data

    def mixup_x(self, x):
        self._make_lambda()
        mixed_x = np.multiply(x[0::2, :, :], self.mixup_lambdas[0, :]) +\
                  np.multiply(x[1::2, :, :], self.mixup_lambdas[1, :])

        return mixed_x

    def mixup_y(self, y):
        self._make_lambda()
        mixed_y = np.multiply(y[0::2, :], self.mixup_lambdas[0, :]) + \
                  np.multiply(y[1::2, :], self.mixup_lambdas[1, :])

        return mixed_y
