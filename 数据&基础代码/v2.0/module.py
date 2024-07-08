import numpy as np
import cupy as cp

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = cp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.bias = cp.zeros(out_channels)
        self.weight_grad = cp.zeros_like(self.weight)
        self.bias_grad = cp.zeros_like(self.bias)

    def forward(self, x):
        self.input = x
        batch_size, in_channels, height, width = x.shape
        out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        out = cp.zeros((batch_size, self.out_channels, out_height, out_width))

        x_padded = cp.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(0, height - self.kernel_size + 1, self.stride):
            for j in range(0, width - self.kernel_size + 1, self.stride):
                x_slice = x_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                out[:, :, i // self.stride, j // self.stride] = cp.tensordot(x_slice, self.weight, axes=([1, 2, 3], [1, 2, 3])) + self.bias
        self.output = out
        return out

    def backward(self, d_out, lr):
        batch_size, _, out_height, out_width = d_out.shape
        _, _, in_height, in_width = self.input.shape
        d_input = cp.zeros_like(self.input)

        x_padded = cp.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        d_x_padded = cp.zeros_like(x_padded)

        for i in range(out_height):
            for j in range(out_width):
                x_slice = x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                for k in range(self.out_channels):
                    self.weight_grad[k] += cp.tensordot(d_out[:, k, i, j], x_slice, axes=([0], [0]))
                d_x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += cp.tensordot(d_out[:, :, i, j], self.weight, axes=([1], [0]))
        self.bias_grad = cp.sum(d_out, axis=(0, 2, 3))
        
        if self.padding != 0:
            d_input = d_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_x_padded

        # Update weights
        self.weight -= lr * self.weight_grad / batch_size
        self.bias -= lr * self.bias_grad / batch_size

        return d_input
    
class MaxPool2d:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def forward(self, x):
        self.input = x
        batch_size, channels, height, width = x.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        out = cp.zeros((batch_size, channels, out_height, out_width))

        for i in range(0, height - self.kernel_size + 1, self.stride):
            for j in range(0, width - self.kernel_size + 1, self.stride):
                x_slice = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                out[:, :, i // self.stride, j // self.stride] = cp.max(x_slice, axis=(2, 3))
        self.output = out
        return out

    def backward(self, d_out, lr):
        batch_size, channels, out_height, out_width = d_out.shape
        d_input = cp.zeros_like(self.input)

        for i in range(0, self.input.shape[2] - self.kernel_size + 1, self.stride):
            for j in range(0, self.input.shape[3] - self.kernel_size + 1, self.stride):
                x_slice = self.input[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                max_x_slice = cp.max(x_slice, axis=(2, 3), keepdims=True)
                temp_binary_mask = (x_slice == max_x_slice)
                d_input[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += temp_binary_mask * (d_out[:, :, i // self.stride, j // self.stride])[:, :, None, None]
        return d_input

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = cp.random.randn(out_features, in_features) * np.sqrt(2. / in_features)
        self.bias = cp.zeros(out_features)
        self.weight_grad = cp.zeros_like(self.weight)
        self.bias_grad = cp.zeros_like(self.bias)

    def forward(self, x):
        self.input = x
        return cp.dot(x, self.weight.T) + self.bias

    def backward(self, d_out, lr):
        batch_size = self.input.shape[0]
        self.weight_grad = cp.dot(d_out.T, self.input)
        self.bias_grad = cp.sum(d_out, axis=0)
        d_input = cp.dot(d_out, self.weight)

        # Update weights
        self.weight -= lr * self.weight_grad / batch_size
        self.bias -= lr * self.bias_grad / batch_size

        return d_input

class ReLU:
    def forward(self, x):
        self.input = x
        return cp.maximum(0, x)

    def backward(self, d_out, lr):
        return d_out * (self.input > 0)

class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + cp.exp(-x))
        return self.output

    def backward(self, d_out, lr):
        return d_out * self.output * (1 - self.output)

class CrossEntropyLoss:
    def forward(self, logits, labels):
        m = labels.shape[0]
        self.labels = labels
        self.logits = logits
        exp_logits = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)
        log_likelihood = -cp.log(self.probs[cp.arange(m), labels])
        loss = cp.sum(log_likelihood) / m
        return loss

    def backward(self, lr):
        m = self.labels.shape[0]
        grad = self.probs.copy()
        grad[cp.arange(m), self.labels] -= 1
        grad /= m
        return grad

if __name__ == "__main__":
    print("Modules defined.")
