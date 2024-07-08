import numpy as np

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def update(self, params, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(grads[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = np.random.randn(out_channels, in_channels * kernel_size * kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels)

    def im2col(self, x, kernel_height, kernel_width, stride, padding):
        if x.ndim != 4:
            raise ValueError("Expected input to be a 4D tensor")
        if any(dim <= 0 for dim in x.shape):
            raise ValueError("Input dimensions must be positive numbers")

        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_indices(x.shape, kernel_height, kernel_width, stride)

        cols = x_padded[:, k, i, j]
        cols = cols.transpose(1, 2, 0).reshape(kernel_height * kernel_width * x.shape[1], -1)
        return cols

    def get_indices(self, shape, height, width, stride):
        N, C, H, W = shape
        out_height = (H + 2 * self.padding - height) // stride + 1
        out_width = (W + 2 * self.padding - width) // stride + 1

        i0 = np.repeat(np.arange(height), width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)

        j0 = np.tile(np.arange(width), height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(C), height * width).reshape(-1, 1)

        return k, i, j

    def forward(self, x):
        self.x = x
        self.cols = self.im2col(x, self.kernel_size, self.kernel_size, self.stride, self.padding)
        res = self.weight.dot(self.cols) + self.bias[:, np.newaxis]
        output_height = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        expected_size = self.out_channels * output_height * output_width * x.shape[0]
        actual_size = res.size
        out = res.reshape(self.out_channels, output_height, output_width, x.shape[0])
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dy, lr):
        dy_reshaped = dy.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        self.weight_grad = dy_reshaped.dot(self.cols.T).reshape(self.weight.shape)
        self.bias_grad = dy_reshaped.sum(axis=1)
        dx_cols = self.weight.T.dot(dy_reshaped)
        dx = self.col2im(dx_cols, self.x.shape, self.kernel_size, self.kernel_size, self.padding, self.stride)
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad
        return dx   
    
    def col2im(self, cols, x_shape, field_height, field_width, padding, stride):
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.get_indices(x_shape, field_height, field_width, stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding != 0:
            return x_padded[:, :, padding:-padding, padding:-padding]
        return x_padded

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dy):
        dx = dy * (self.x > 0)
        return dx   

class BatchNorm:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)
        self.training = True

    def forward(self, x):
        self.is_four_dim = len(x.shape) == 4  # Use an instance variable to store if the input is 4D

        if self.is_four_dim:
            N, C, H, W = x.shape
            # print(f"Input shape (4D): {x.shape}")
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            N, C = x.shape

        if self.training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        self.x_centered = x - mean
        self.stddev_inv = 1. / np.sqrt(var + self.epsilon)
        x_norm = self.x_centered * self.stddev_inv
        out = self.gamma * x_norm + self.beta

        if self.is_four_dim:
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            # print(f"Output shape (4D): {out.shape}")

        return out

    def backward(self, dy):
        if self.is_four_dim:
            N, C, H, W = dy.shape
            dy = dy.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            N, C = dy.shape

        dbeta = np.sum(dy, axis=0)
        dgamma = np.sum(self.x_centered * self.stddev_inv * dy, axis=0)
        dx_norm = self.gamma * dy
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * np.power(self.stddev_inv, 3), axis=0)
        dmean = np.sum(dx_norm * -self.stddev_inv, axis=0) + dvar * np.mean(-2. * self.x_centered, axis=0)
        dx = dx_norm * self.stddev_inv + dvar * 2 * self.x_centered / N + dmean / N

        if self.is_four_dim:
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return dx

class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, dy):
        return dy * (1 - np.square(np.tanh(dy)))

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, dy):
        sig = self.forward(dy)
        return dy * sig * (1 - sig)

class MaxPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        new_height = 1 + (H - self.kernel_size) // self.stride
        new_width = 1 + (W - self.kernel_size) // self.stride
        x_reshaped = x.reshape(N, C, H // self.kernel_size, self.kernel_size, W // self.kernel_size, self.kernel_size)
        out = x_reshaped.max(axis=3).max(axis=4)
        return out

    def backward(self, dy):
        N, C, H, W = self.x.shape
        dX = np.zeros_like(self.x)
        
        for n in range(N):
            for c in range(C):
                for i in range(dy.shape[2]):
                    for j in range(dy.shape[3]):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = min(h_start + self.kernel_size, H)
                        w_end = min(w_start + self.kernel_size, W)

                        x_slice = self.x[n, c, h_start:h_end, w_start:w_end]
                        mask = x_slice == np.max(x_slice)
                        dX[n, c, h_start:h_end, w_start:w_end] += dy[n, c, i, j] * mask

        return dX

class AvgPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        new_height = 1 + (H - self.kernel_size) // self.stride
        new_width = 1 + (W - self.kernel_size) // self.stride
        x_reshaped = x.reshape(N, C, H // self.kernel_size, self.kernel_size, W // self.kernel_size, self.kernel_size)
        out = x_reshaped.mean(axis=3).mean(axis=4)
        return out

    def backward(self, dy):
        N, C, H, W = self.x.shape
        dX = np.zeros_like(self.x)
        
        for n in range(N):
            for c in range(C):
                for i in range(dy.shape[2]):
                    for j in range(dy.shape[3]):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = min(h_start + self.kernel_size, H)
                        w_end = min(w_start + self.kernel_size, W)

                        area = (h_end - h_start) * (w_end - w_start)
                        dX[n, c, h_start:h_end, w_start:w_end] += dy[n, c, i, j] / area

        return dX

class Linear:
    def __init__(self, in_features: int, out_features: int, bias=True):
        self.weight = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)
        self.bias = np.zeros(out_features) if bias else None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight.T) + self.bias

    def backward(self, dy, lr):
        self.weight_grad = np.dot(dy.T, self.x)
        self.bias_grad = dy.sum(axis=0)
        dx = np.dot(dy, self.weight)
        self.weight -= lr * self.weight_grad
        if self.bias is not None:
            self.bias -= lr * self.bias_grad
        return dx        

class CrossEntropyLoss:
    def __call__(self, x, label):
        m = x.shape[0]
        p = np.exp(x - np.max(x, axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        self.p = p
        self.label = label
        log_likelihood = -np.log(p[range(m), label])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self):
        m = self.p.shape[0]
        grad = self.p.copy()
        grad[range(m), self.label] -= 1
        grad /= m
        return grad

