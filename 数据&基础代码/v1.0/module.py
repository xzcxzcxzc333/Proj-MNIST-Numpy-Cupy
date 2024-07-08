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
    #dtype (optional): 可以指定数据类型，例如 np.float32。如果未指定，则使用默认数据类型，dtype 参数用于指定数组或其他数据结构的期望数据类型。在这个特定的上下文中（构造函数的定义里），dtype 被设置为默认值 None，这意味着如果不显式提供一个数据类型，那么数据类型将由使用的库（如 CuPy 或 NumPy）在内部决定。在许多情况下，库会基于输入数据自动选择一个合适的数据类型。
    #在 Python 中，self 是类中方法的第一个参数，它是一个对当前对象实例的引用。当你创建一个类的实例时，Python 会自动传递这个引用给每一个方法的第一个参数，这就是为什么你会在方法定义中看到 self 参数，但在调用这些方法时不需要传递它。
        self.in_channels = in_channels #输入图像的通道数
        self.out_channels = out_channels  #卷积产生的通道数
        self.kernel_size = kernel_size  #卷积核的大小
        self.stride = stride #卷积的步长
        self.padding = padding #输入的两侧加上的零填充

        # 初始化权重和偏置
        self.weight = cp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * cp.sqrt(2. / in_channels)
        #权重使用一种随输入通道数缩放的方法初始化，这有助于维持稳定的梯度。使用 CuPy 生成一个形状为 (out_channels, in_channels, kernel_size, kernel_size) 的数组，数组中的元素是从标准正态分布（均值为0，方差为1的高斯分布）中抽取的。
        #这里的两个kernel_size表示卷积核的高度和宽度
        #cp.sqrt(2. / in_channels)表示，He 初始化方法的一部分，用于帮助在训练开始时保持激活的标准化。具体来说，它帮助在使用 ReLU 激活函数的深度神经网络中防止梯度消失问题。
        self.bias = cp.zeros(out_channels)

        print("Conv2d initialized with in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}".format(
            in_channels, out_channels, kernel_size, stride, padding))

    
    def get_weights(self):
        return {'weight': self.weight, 'bias': self.bias}
    #返回一个包含当前卷积层权重和偏置的字典

    def set_weights(self, weights):
        self.weight = weights['weight']
        self.bias = weights['bias']
    #将层的权重和偏置设置为提供的值
    #weights 是一个字典，该字典包含了两个键：'weight' 和 'bias'，它们分别存储了某个神经网络层的权重和偏置

    def im2col(self, x, HH, WW, stride):
    # x：输入的图像或特征图，其形状为 (N, C, H, W)，其中 N 是批量大小，C 是通道数，H 和 W 是图像的高度和宽度。
    # HH：卷积核的高度。
    # WW：卷积核的宽度。
    # stride：卷积操作中的步长。
    #将输入图像 x 转换为列形式，以便在卷积中进行矩阵乘法。内部处理填充，并调整切片以匹配卷积参数。
        N, C, H, W = x.shape
        #x.shape 将返回数组的维度。在卷积神经网络中，x 通常表示一个四维数组，其中包含批量大小（N），通道数（C），高度（H）和宽度（W）
        H_padded = H + 2 * self.padding
        W_padded = W + 2 * self.padding
        #指定的填充值计算填充后的高度和宽度
        out_h = (H_padded - HH) // stride + 1
        out_w = (W_padded - WW) // stride + 1
        # H_padded 是经过填充后的输入高度。
        # HH 是卷积核的高度。
        # stride 是卷积操作中的步长，即卷积核在输入数据上移动的距离。
        #// 是整除操作符，它用于返回两个数相除的商，但结果是向下取整的整数。这与常规的除法 / 不同，后者返回浮点结果
        col = cp.zeros((N, C, HH, WW, out_h, out_w))
        # 创建一个六维张量，用于存储转换后的数据，其中每个位置都会对应一个卷积窗口的数据。
        # 外层括号 是函数调用的一部分，表示你正在调用 cp.zeros() 这个函数。
        # 内层括号 实际上是创建一个元组，这个元组定义了数组的维度和大小。
        # 对输入进行填充
        x_padded = cp.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        # 使用零填充方式对输入图像进行填充，确保卷积操作时边缘也能够被卷积核覆盖。
        # (0, 0): 第一个和第二个维度（通常对应于批次大小和通道数）不进行填充。
        # (self.padding, self.padding): 第三个和第四个维度（通常是图像的高度和宽度）在两边各填充 self.padding 个元素。这样做是为了确保卷积操作可以适当地应用于图像边缘，扩大图像的感受野，或保持空间尺寸。
        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        i_start = i * stride
                        j_start = j * stride
                        i_end = i_start + HH
                        j_end = j_start + WW
                        if i_end > H + 2 * self.padding or j_end > W + 2 * self.padding:
                            continue
                        # 创建一个临时的小矩阵填充到滤波器尺寸
                        temp_slice = x_padded[n, c, i_start:i_end, j_start:j_end]
                        # 用0填充至完整的滤波器尺寸
                        padded_slice = cp.zeros((HH, WW))
                        padded_slice[:temp_slice.shape[0], :temp_slice.shape[1]] = temp_slice
                        col[n, c, :, :, i, j] = padded_slice
                        #这个循环遍历每个可能的卷积窗口位置，从填充后的图像中提取相应的数据片段，然后将这些数据片段填充到固定大小的小矩阵中，并存储在 col 张量中。
        # 这个循环遍历每个可能的卷积窗口位置，从填充后的图像中提取相应的数据片段，然后将这些数据片段填充到固定大小的小矩阵中，并存储在 col 张量中。
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, C * HH * WW)
        # 将六维张量转置和重塑为二维矩阵，使得每一列都对应一个展开的卷积窗口
        return col


    def forward(self, x):
        """
        x - shape (N, C, H, W)
        return the result of Conv2d with shape (N, O, H', W')
        """
        self.x = x
        N, C, H, W = x.shape
        F, _, HH, WW = self.weight.shape
        H_out = (H + 2 * self.padding - HH) // self.stride + 1
        W_out = (W + 2 * self.padding - WW) // self.stride + 1
        print(f"Input shape: {x.shape}")
        print(f"Filter shape: {self.weight.shape}")
        print(f"Output height: {H_out}, Output width: {W_out}")

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
        print(f"Padded input shape: {x_padded.shape}")
        # 使用im2col处理
        x_col = self.im2col(x_padded, HH, WW, self.stride)
        print(f"im2col output shape: {x_col.shape}")

        w_col = self.weight.reshape(F, -1)
        print(f"Weight matrix shape for matmul: {w_col.shape}")
        
        # 确保x_col的形状正确，进行转置以匹配w_col的列数
        x_col = x_col.T  # 现在x_col的形状应该是 (25, 131072)
        print(f"x_col reshaped for matmul: {x_col.shape}")

        out = w_col @ x_col + self.bias[:, None]  # 确保偏置广播正确
        print(f"Output shape before reshape: {out.shape}")

        # 确保矩阵维度正确对齐
        # out = w_col @ x_col.T + self.bias[:, None]  # 注意这里使用了x_col的转置
        out = out.reshape(F, H_out, W_out, N)
        out = out.transpose(3, 0, 1, 2)  # 转换回(N, F, H_out, W_out)的形状
        print(f"Final output shape: {out.shape}")
        return out
    
    def col2im(self, cols, x_shape, HH, WW, stride, padding):
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = cp.zeros((N, C, H_padded, W_padded))

        out_h = (H_padded - HH) // stride + 1
        out_w = (W_padded - WW) // stride + 1
        
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
        print("backward: dy_shape={}, lr={}".format(dy.shape, lr))

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
        print("backward: dw_shape={}, db_shape={}".format(dw.shape, db.shape))

        dw = dy_col @ x_col.T
        dw = dw.reshape(self.weight.shape)
        db = cp.sum(dy, axis=(0, 2, 3))

        dx_col = self.weight.reshape(F, -1).T @ dy_col
        dx = self.col2im(dx_col, (N, C, H, W), self.kernel_size, self.kernel_size, self.stride, self.padding)
        print("backward: dx_shape={}".format(dx.shape))

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

    def get_weights(self):
        return {'weight': self.weight, 'bias': self.bias}

    def set_weights(self, weights):
        self.weight = weights['weight']
        self.bias = weights['bias']
        
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


