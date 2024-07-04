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
    # net = LeNet5()
    # criterion = CrossEntropyLoss()

    # epochs = 10
    # batch_size = 64
    # learning_rate = 0.01

    # for epoch in range(epochs):
    #     permutation = np.random.permutation(len(train_images))
    #     train_images = train_images[permutation]
    #     train_labels = train_labels[permutation]

    #     for i in tqdm.tqdm(range(0, len(train_images), batch_size)):
    #         x_batch = train_images[i:i+batch_size]
    #         y_batch = train_labels[i:i+batch_size]

    #         # Forward pass
    #         outputs = net.forward(x_batch)
    #         loss, dy = criterion(outputs, y_batch)

    #         # Backward pass
    #         net.backward(dy, learning_rate)

    #     print(f'Epoch {epoch + 1}, Loss: {loss}')

    # # Evaluate on test set
    # correct = 0
    # total = 0
    # for i in range(0, len(test_images), batch_size):
    #     x_batch = test_images[i:i+batch_size]
    #     y_batch = test_labels[i:i+batch_size]

    #     outputs = net.forward(x_batch)
    #     predictions = np.argmax(outputs, axis=1)
    #     correct += (predictions == y_batch).sum()
    #     total += len(y_batch)

    # accuracy = correct / total
    # print(f'Test Accuracy: {accuracy}')    
    
    # 将数据转换为 CuPy 数组
    train_images = cp.array(train_images, dtype=cp.float16).reshape(-1, 1, 28, 28)
    train_labels = cp.array(train_labels)
    test_images = cp.array(test_images, dtype=cp.float16).reshape(-1, 1, 28, 28)
    test_labels = cp.array(test_labels)

    model = LeNet5()
    criterion = CrossEntropyLoss()
    
    lr = 0.001
    epochs = 10
    batch_size = 64
    # 检查 CuPy 是否能正确使用 GPU
    # cp.cuda.Device(0).use()
    # print(f"Using GPU: {cp.cuda.get_device_name(0)}")

    for epoch in range(epochs):
        permutation = cp.random.permutation(train_images.shape[0])
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        losses = []
        
        with tqdm.tqdm(total=train_images.shape[0], desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for i in range(0, train_images.shape[0], batch_size):
                x_batch = train_images[i:i+batch_size]
                y_batch = train_labels[i:i+batch_size]

                # For example, time the forward pass
                start_time = time.time()
                outputs = model.forward(x_batch)
                end_time = time.time()
                print(f"Forward pass time: {end_time - start_time} seconds")
                sys.stdout.flush()
                # Assuming `criterion` is an instance of `CrossEntropyLoss`
                # Similarly, time the backward pass
                start_time = time.time()
                loss, _ = criterion(outputs, y_batch)
                losses.append(loss)

                dy = criterion.backward()
                model.backward(dy, lr)
                end_time = time.time()
                print(f"Backward pass time: {end_time - start_time} seconds")
                sys.stdout.flush()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {cp.mean(cp.array(losses))}")
        sys.stdout.flush()
    # 测试
    correct = 0
    total = test_images.shape[0]
    
    for i in range(0, total, batch_size):
        x_batch = test_images[i:i+batch_size]
        y_batch = test_labels[i:i+batch_size]
        
        outputs = model.forward(x_batch)
        predictions = cp.argmax(outputs, axis=1)
        correct += cp.sum(predictions == y_batch)
    # 转换为 NumPy 数组并提取标量值
    accuracy = (correct / total).get().item()
    print(f"Test Accuracy: {accuracy:.4f}")
    sys.stdout.flush()
    # ---------------------------------------------------------------------------
    
    

    
            
            

