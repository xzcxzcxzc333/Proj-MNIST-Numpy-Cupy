import numpy as np
import cupy as cp
from module import Conv2d, ReLU, MaxPool2d, AvgPool2d, Linear, BatchNorm, CrossEntropyLoss, Adam
import struct
import glob
import tqdm
import cv2
import time
import sys

def load_mnist(path, kind='train'):
    image_path = glob.glob('./%s*3-ubyte' % (kind))[0]
    label_path = glob.glob('./%s*1-ubyte' % (kind))[0]

    with open(label_path, "rb") as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, "rb") as impath:
        magic, num, rows, cols = struct.unpack('>IIII', impath.read(16))
        images = np.fromfile(impath, dtype=np.uint8).reshape(len(labels), 28*28)

    return images, labels

class LeNet5:
    def __init__(self):
        # 第一层卷积：输入通道数1（灰度图像），输出通道数6，卷积核大小5x5，步长1，填充2
        self.conv1 = Conv2d(1, 6, 5, 1, 2)
        # 批量归一化层，输入通道数6
        self.bn1 = BatchNorm(6)
        # 激活函数ReLU
        self.relu1 = ReLU()
        # 平均池化层，池化大小2x2
        self.pool1 = AvgPool2d(2)
        # 第二层卷积：输入通道数6，输出通道数16，卷积核大小5x5
        self.conv2 = Conv2d(6, 16, 5)
        # 批量归一化层，输入通道数16
        self.bn2 = BatchNorm(16)
        # 激活函数ReLU
        self.relu2 = ReLU()
        # 平均池化层，池化大小2x2
        self.pool2 = AvgPool2d(2)
        # 全连接层，输入大小16*5*5（卷积输出展平后的大小），输出大小120
        self.fc1 = Linear(16*5*5, 120)
        # 批量归一化层，输入大小120
        self.bn3 = BatchNorm(120)
        # 激活函数ReLU
        self.relu3 = ReLU()
        # 全连接层，输入大小120，输出大小84
        self.fc2 = Linear(120, 84)
        # 批量归一化层，输入大小84
        self.bn4 = BatchNorm(84)
        # 激活函数ReLU
        self.relu4 = ReLU()
        # 全连接层，输入大小84，输出大小10（10个类别）
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        # 第一层卷积前向传播
        x = self.conv1.forward(x)
        # 批量归一化
        x = self.bn1.forward(x)
        # 激活函数ReLU
        x = self.relu1.forward(x)
        # 平均池化
        x = self.pool1.forward(x)
        # 第二层卷积前向传播
        x = self.conv2.forward(x)
        # 批量归一化
        x = self.bn2.forward(x)
        # 激活函数ReLU
        x = self.relu2.forward(x)
        # 平均池化
        x = self.pool2.forward(x)
        # 展平操作，将多维张量展平成二维张量
        x = x.reshape(x.shape[0], -1)
        # 全连接层1前向传播
        x = self.fc1.forward(x)
        # 批量归一化
        x = self.bn3.forward(x)
        # 激活函数ReLU
        x = self.relu3.forward(x)
        # 全连接层2前向传播
        x = self.fc2.forward(x)
        # 批量归一化
        x = self.bn4.forward(x)
        # 激活函数ReLU
        x = self.relu4.forward(x)
        # 全连接层3前向传播，得到最终输出
        x = self.fc3.forward(x)
        return x

    def backward(self, dy, lr):
        # 全连接层3反向传播
        dy = self.fc3.backward(dy, lr)
        # 批量归一化反向传播
        dy = self.bn4.backward(dy)
        # 激活函数ReLU反向传播
        dy = self.relu4.backward(dy)
        # 全连接层2反向传播
        dy = self.fc2.backward(dy, lr)
        # 批量归一化反向传播
        dy = self.bn3.backward(dy)
        # 激活函数ReLU反向传播
        dy = self.relu3.backward(dy)
        # 全连接层1反向传播
        dy = self.fc1.backward(dy, lr)
        # 将二维张量恢复为多维张量
        dy = dy.reshape(dy.shape[0], 16, 5, 5)
        # 平均池化反向传播
        dy = self.pool2.backward(dy)
        # 激活函数ReLU反向传播
        dy = self.relu2.backward(dy)
        # 批量归一化反向传播
        dy = self.bn2.backward(dy)
        # 第二层卷积反向传播
        dy = self.conv2.backward(dy, lr)
        # 平均池化反向传播
        dy = self.pool1.backward(dy)
        # 激活函数ReLU反向传播
        dy = self.relu1.backward(dy)
        # 批量归一化反向传播
        dy = self.bn1.backward(dy)
        # 第一层卷积反向传播
        dy = self.conv1.backward(dy, lr)

if __name__ == '__main__':
    train_images, train_labels = load_mnist("mnist_dataset", kind="train")
    test_images, test_labels = load_mnist("mnist_dataset", kind="t10k")

    train_images = train_images.astype(np.float16) / 256
    test_images = test_images.astype(np.float16) / 256

    train_images = train_images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    test_images = test_images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    model = LeNet5()
    loss_fn = CrossEntropyLoss()

    epochs = 10
    learning_rate = 0.001
    batch_size = 64

    optimizer = Adam([model.conv1.weight, model.conv1.bias, model.conv2.weight, model.conv2.bias, 
                      model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias, 
                      model.fc3.weight, model.fc3.bias], lr=learning_rate)

    for epoch in range(epochs):
        model.bn1.training = True
        model.bn2.training = True
        model.bn3.training = True
        model.bn4.training = True

        permutation = np.random.permutation(len(train_images))
        total_batches = len(train_images) // batch_size
        with tqdm.tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
            for i in range(0, len(train_images), batch_size):
                indices = permutation[i:i + batch_size]
                inputs = train_images[indices]
                targets = train_labels[indices]
                outputs = model.forward(inputs)
                loss = loss_fn(outputs, targets)
                grad_outputs = loss_fn.backward()
                model.backward(grad_outputs, learning_rate)
                optimizer.update([model.conv1.weight, model.conv1.bias, model.conv2.weight, model.conv2.bias, 
                                  model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias, 
                                  model.fc3.weight, model.fc3.bias],
                                 [model.conv1.weight_grad, model.conv1.bias_grad, model.conv2.weight_grad, model.conv2.bias_grad, 
                                  model.fc1.weight_grad, model.fc1.bias_grad, model.fc2.weight_grad, model.fc2.bias_grad, 
                                  model.fc3.weight_grad, model.fc3.bias_grad])
                correct = (np.argmax(outputs, axis=1) == targets).sum()
                accuracy = correct / batch_size
                pbar.set_postfix(Loss=f'{loss:.2f}', Acc=f'{accuracy:.3f}')
                pbar.update()

        print(f'Epoch {epoch + 1}, Loss: {loss}')

    model.bn1.training = False
    model.bn2.training = False
    model.bn3.training = False
    model.bn4.training = False

    correct = 0
    total = len(test_labels)
    for i in range(0, total, batch_size):
        inputs = test_images[i:i + batch_size]
        targets = test_labels[i:i + batch_size]
        outputs = model.forward(inputs)
        predicted = np.argmax(outputs, axis=1)
        correct += (predicted == targets).sum()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
