# import numpy as np

# class Conv2d:
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
#                  stride: int = 1, padding: int = 0, dtype = None):
#         # ------------------------------请完成此部分内容------------------------------
    
    
#         # ---------------------------------------------------------------------------

#     def forward(self, x):
#         """
#         x - shape (N, C, H, W)
#         return the result of Conv2d with shape (N, O, H', W')
#         """
#        # ------------------------------请完成此部分内容------------------------------
    
    
#         # ---------------------------------------------------------------------------

#     def backward(self, dy, lr):
#         """
#         dy - the gradient of last layer with shape (N, O, H', W')
#         lr - learning rate
#         calculate self.w_grad to update self.weight,
#         calculate self.b_grad to update self.bias,
#         return the result of gradient dx with shape (N, C, H, W)
#         """
#        # ------------------------------请完成此部分内容------------------------------
    

    
#         # ---------------------------------------------------------------------------
    

     

# class ReLU:
#     def forward(self, x):
      
#     def backward(self, dy):
       

# class Tanh:
#     def forward(self, x):
       

#     def backward(self, dy):
        
# class Sigmoid:
#     def forward(self, x):
       
#     def backward(self, dy):
       
# class MaxPool2d:
#     def __init__(self, kernel_size: int, stride = None, padding = 0):
      

#     def forward(self, x):
#         """
#         x - shape (N, C, H, W)
#         return the result of MaxPool2d with shape (N, C, H', W')
#         """
       

#     def backward(self, dy):
#         """
#         dy - shape (N, C, H', W')
#         return the result of gradient dx with shape (N, C, H, W)
#         """
     
# class AvgPool2d:
#     def __init__(self, kernel_size: int, stride = None, padding = 0):
       
#     def forward(self, x):
#         """
#         x - shape (N, C, H, W)
#         return the result of AvgPool2d with shape (N, C, H', W')
#         """
       

#     def backward(self, dy):
#         """
#         dy - shape (N, C, H', W')
#         return the result of gradient dx with shape (N, C, H, W)
#         """
       
        

# class Linear:
#     def __init__(self, in_features: int, out_features: int, bias: bool = True):
        
        

#     def forward(self, x):
#         """
#         x - shape (N, C)
#         return the result of Linear layer with shape (N, O)
#         """
       


#     def backward(self, dy, lr):
#         """
#         dy - shape (N, O)
#         return the result of gradient dx with shape (N, C)
#         """
        

# class CrossEntropyLoss:
#     def __call__(self, x, label):
       
import cupy as cp

class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dtype=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化权重和偏置
        self.weight = cp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * cp.sqrt(2. / in_channels)
        self.bias = cp.zeros(out_channels)

    def im2col(self, x, HH, WW, stride):
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - HH) // stride + 1
        out_w = (W + 2 * self.padding - WW) // stride + 1
        col = cp.zeros((N, C, HH, WW, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                i_start = i * stride
                i_end = i_start + HH
                j_start = j * stride
                j_end = j_start + WW
                 # 添加检查，确保切片的大小总是匹配的
                if i_end <= H + 2 * self.padding and j_end <= W + 2 * self.padding:
                    col[:, :, :, :, i, j] = x[:, :, i_start:i_end, j_start:j_end]


        col = col.transpose(1, 2, 3, 0, 4, 5).reshape(C * HH * WW, -1)
        return col
    
    def col2im(self, cols, x_shape, HH, WW, stride, padding):
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = cp.zeros((N, C, H_padded, W_padded))

        out_h = (H + 2 * padding - HH) // stride + 1
        out_w = (W + 2 * padding - WW) // stride + 1

        cols_reshaped = cols.reshape(C, HH, WW, N, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)

        for i in range(out_h):
            for j in range(out_w):
                i_start = i * stride
                i_end = i_start + HH
                j_start = j * stride
                j_end = j_start + WW
                # 添加检查，确保切片的大小总是匹配的
                if i_end <= H + 2 * padding and j_end <= W + 2 * padding:
                    x_padded[:, :, i_start:i_end, j_start:j_end] += cols_reshaped[:, :, :, :, i, j]

        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]


    def forward(self, x):
        """
        x - shape (N, C, H, W)
        return the result of Conv2d with shape (N, O, H', W')
        """
        self.x = x  # 记录输入张量
        N, C, H, W = x.shape
        F, _, HH, WW = self.weight.shape
        H_out = 1 + (H + 2 * self.padding - HH) // self.stride
        W_out = 1 + (W + 2 * self.padding - WW) // self.stride

        # 添加padding
        # x_padded = cp.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # out = cp.zeros((N, F, H_out, W_out))

        # for i in range(H_out):
        #     for j in range(W_out):
        #         x_slice = x_padded[:, :, i * self.stride:i * self.stride + HH, j * self.stride:j * self.stride + WW]
        #         for k in range(F):
        #             out[:, k, i, j] = cp.sum(x_slice * self.weight[k, :, :, :], axis=(1, 2, 3))

        # out += self.bias[None, :, None, None]
        # return out

        # 使用im2col展开

        # 添加padding
        x_padded = cp.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # 使用im2col展开
        x_col = self.im2col(x_padded, HH, WW, self.stride)
        w_col = self.weight.reshape(F, -1)
        out = w_col @ x_col + self.bias[:, None]
        out = out.reshape(F, H_out, W_out, N)
        out = out.transpose(3, 0, 1, 2)  # 转换回(N, F, H_out, W_out)的形状

        return out
    
        
    def backward(self, dy, lr):
        """
        dy - the gradient of last layer with shape (N, O, H', W')
        lr - learning rate
        calculate self.w_grad to update self.weight,
        calculate self.b_grad to update self.bias,
        return the result of gradient dx with shape (N, C, H, W)
        """
        N, F, H_out, W_out = dy.shape
        _, C, H, W = self.x.shape # 使用前向传播中记录的输入形状

        # x_padded = cp.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        # dx_padded = cp.zeros_like(x_padded)
        # dw = cp.zeros_like(self.weight)
        # db = cp.zeros_like(self.bias)

        # for i in range(H_out):
        #     for j in range(W_out):
        #         x_slice = x_padded[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
        #         for k in range(F):
        #             dw[k] += cp.sum(x_slice * dy[:, k, i, j][:, None, None, None], axis=0)
        #         for n in range(N):
        #             dx_padded[n, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size] += \
        #                 cp.sum(self.weight[:, :, :, :] * dy[n, :, i, j][:, None, None, None], axis=0)
        # db = cp.sum(dy, axis=(0, 2, 3))

        # # 去掉padding
        # dx = dx_padded[:, :, self.padding:H + self.padding, self.padding:W + self.padding]
        x_padded = cp.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        x_col = self.im2col(x_padded, self.kernel_size, self.kernel_size, self.stride)
        dy_col = dy.transpose(1, 2, 3, 0).reshape(F, -1)

        dw = dy_col @ x_col.T
        dw = dw.reshape(self.weight.shape)
        db = cp.sum(dy, axis=(0, 2, 3))

        dx_col = self.weight.reshape(F, -1).T @ dy_col
        dx = self.col2im(dx_col, (N, C, H, W), self.kernel_size, self.kernel_size, self.stride, self.padding)

        # 更新权重和偏置
        self.weight -= lr * dw
        self.bias -= lr * db

        return dx

class ReLU:
    def forward(self, x):
        self.x = x
        return cp.maximum(0, x)

    def backward(self, dy):
        dx = dy.copy()
        dx[self.x <= 0] = 0
        return dx

class Tanh:
    def forward(self, x):
        self.y = cp.tanh(x)
        return self.y

    def backward(self, dy):
        return dy * (1 - self.y ** 2)

class Sigmoid:
    def forward(self, x):
        self.y = 1 / (1 + cp.exp(-x))
        return self.y

    def backward(self, dy):
        return dy * self.y * (1 - self.y)

class MaxPool2d:
    def __init__(self, kernel_size: int, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        H_out = 1 + (H - self.kernel_size) // self.stride
        W_out = 1 + (W - self.kernel_size) // self.stride

        out = cp.zeros((N, C, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                x_slice = x[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                out[:, :, i, j] = cp.max(x_slice, axis=(2, 3))

        return out

    def backward(self, dy):
        N, C, H, W = self.x.shape
        dx = cp.zeros_like(self.x)

        H_out, W_out = dy.shape[2], dy.shape[3]

        for i in range(H_out):
            for j in range(W_out):
                x_slice = self.x[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                max_x_slice = cp.max(x_slice, axis=(2, 3), keepdims=True)
                mask = (x_slice == max_x_slice)
                dx[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size] += dy[:, :, i, j][:, :, None, None] * mask

        return dx

class AvgPool2d:
    def __init__(self, kernel_size: int, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        H_out = 1 + (H - self.kernel_size) // self.stride
        W_out = 1 + (W - self.kernel_size) // self.stride

        out = cp.zeros((N, C, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                x_slice = x[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                out[:, :, i, j] = cp.mean(x_slice, axis=(2, 3))

        return out

    def backward(self, dy):
        N, C, H, W = self.x.shape
        dx = cp.zeros_like(self.x)

        H_out, W_out = dy.shape[2], dy.shape[3]

        for i in range(H_out):
            for j in range(W_out):
                dx[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size] += dy[:, :, i, j][:, :, None, None] / (self.kernel_size * self.kernel_size)

        return dx

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = cp.random.randn(out_features, in_features) * cp.sqrt(2. / in_features)
        self.bias = cp.zeros(out_features) if bias else None

    def forward(self, x):
        self.x = x
        if self.bias is not None:
            return cp.dot(x, self.weight.T) + self.bias
        else:
            return cp.dot(x, self.weight.T)

    def backward(self, dy, lr):
        self.dw = cp.dot(dy.T, self.x)
        if self.bias is not None:
            self.db = cp.sum(dy, axis=0)

        dx = cp.dot(dy, self.weight)
        self.weight -= lr * self.dw
        if self.bias is not None:
            self.bias -= lr * self.db

        return dx

class CrossEntropyLoss:
    def __call__(self, x, label):
        """
        x - shape (N, C) : logits
        label - shape (N, ) : ground truth labels
        return loss and gradient with shape (N, C)
        """
        N = x.shape[0]
        exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
        softmax = exp_x / cp.sum(exp_x, axis=1, keepdims=True)
        self.loss = -cp.sum(cp.log(softmax[cp.arange(N), label])) / N

        self.dx = softmax.copy()
        self.dx[cp.arange(N), label] -= 1
        self.dx /= N

        return self.loss, self.dx

    def backward(self):
        return self.dx


