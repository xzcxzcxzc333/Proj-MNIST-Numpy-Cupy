import numpy as np
import cupy as cp
from module import Conv2d, Sigmoid, MaxPool2d, AvgPool2d, Linear, ReLU, Tanh, CrossEntropyLoss
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
        
        self.conv1 = Conv2d(1, 6, 5, 1, 2)
        self.relu1 = Sigmoid()
        self.pool1 = AvgPool2d(2)
        self.conv2 = Conv2d(6, 16, 5)
        self.relu2 = Sigmoid()
        self.pool2 = AvgPool2d(2)
        self.fc1 = Linear(16*5*5, 120)
        self.relu3 = Sigmoid()
        self.fc2 = Linear(120, 84)
        self.relu4 = Sigmoid()
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        x = self.relu4.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, dy, lr):
        dy = self.fc3.backward(dy, lr)
        dy = self.relu4.backward(dy)
        dy = self.fc2.backward(dy, lr)
        dy = self.relu3.backward(dy)
        dy = self.fc1.backward(dy, lr)
        dy = dy.reshape(dy.shape[0], 16, 5, 5)  # Reshape back to conv2 output shape
        dy = self.pool2.backward(dy)
        dy = self.relu2.backward(dy)
        dy = self.conv2.backward(dy, lr)
        dy = self.pool1.backward(dy)
        dy = self.relu1.backward(dy)
        dy = self.conv1.backward(dy, lr)

if __name__ == '__main__':
    
    train_images, train_labels = load_mnist("mnist_dataset", kind="train")
    test_images, test_labels = load_mnist("mnist_dataset", kind="t10k")

    train_images = train_images.astype(np.float16) / 256
    test_images = test_images.astype(np.float16) / 256
    
    # ----------------------------请完成网络的训练和测试----------------------------
    # 确保图像是四维的 (N, C, H, W)
    train_images = train_images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    test_images = test_images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0    
    # 初始化模型和损失函数
    model = LeNet5()
    loss_fn = CrossEntropyLoss()

    # 设置训练参数
    epochs = 10
    learning_rate = 0.01
    batch_size = 64

    # 训练循环
    for epoch in range(epochs):
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
                correct = (np.argmax(outputs, axis=1) == targets).sum()
                accuracy = correct / batch_size
                pbar.set_postfix(Loss=f'{loss:.2f}', Acc=f'{accuracy:.3f}')
                pbar.update()

        # 输出训练信息
        print(f'Epoch {epoch + 1}, Loss: {loss}')

    # 测试模型
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
    

    
            
            

