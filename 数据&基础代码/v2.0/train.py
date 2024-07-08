import numpy as np
import cupy as cp
import struct
import glob
import cv2
import time
import sys
from tqdm import tqdm
from module import Conv2d, MaxPool2d, Linear, ReLU, Sigmoid, CrossEntropyLoss

def load_mnist(path, kind='train'):
    image_path = glob.glob('./%s*3-ubyte' % (kind))[0]
    label_path = glob.glob('./%s*1-ubyte' % (kind))[0]

    if not label_path or not image_path:
        raise FileNotFoundError(f"Could not find files in {path} for {kind} data. Make sure the files exist and the paths are correct.")


    with open(label_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28)

    return images, labels

class LeNet5:
    def __init__(self):
        self.conv1 = Conv2d(1, 6, 5)
        self.pool1 = MaxPool2d(2)
        self.conv2 = Conv2d(6, 16, 5)
        self.pool2 = MaxPool2d(2)
        self.fc1 = Linear(16*4*4, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.loss = CrossEntropyLoss()

    def forward(self, x):
        x = self.relu.forward(self.conv1.forward(x))
        x = self.pool1.forward(x)
        x = self.relu.forward(self.conv2.forward(x))
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu.forward(self.fc1.forward(x))
        x = self.relu.forward(self.fc2.forward(x))
        x = self.fc3.forward(x)
        return x

    def backward(self, d_out, lr):
        d_out = self.fc3.backward(d_out, lr)
        d_out = self.relu.backward(d_out, lr)
        d_out = self.fc2.backward(d_out, lr)
        d_out = self.relu.backward(d_out, lr)
        d_out = self.fc1.backward(d_out, lr)
        d_out = d_out.reshape(d_out.shape[0], 16, 4, 4)
        d_out = self.pool2.backward(d_out, lr)
        d_out = self.relu.backward(d_out, lr)
        d_out = self.conv2.backward(d_out, lr)
        d_out = self.pool1.backward(d_out, lr)
        d_out = self.relu.backward(d_out, lr)
        d_out = self.conv1.backward(d_out, lr)

def train(model, train_images, train_labels, epochs=10, lr=0.001):
     for epoch in range(epochs):
        correct = 0
        total_loss = 0
        with tqdm(total=len(train_images) // 64, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for i in range(0, len(train_images), 64):
                x_batch = cp.array(train_images[i:i+64].reshape(-1, 1, 28, 28))
                y_batch = cp.array(train_labels[i:i+64])

                logits = model.forward(x_batch)
                loss = model.loss.forward(logits, y_batch)
                total_loss += loss
                preds = cp.argmax(logits, axis=1)
                correct += cp.sum(preds == y_batch)

                # Backpropagation and gradient descent
                d_out = model.loss.backward(lr)
                model.backward(d_out, lr)

                pbar.set_postfix(Acc=f'{correct/((i+64) if (i+64) < len(train_images) else len(train_images)):.3f}', Loss=f'{total_loss/((i+64) if (i+64) < len(train_images) else len(train_images)):.2f}')
                pbar.update(1)
                
def test(model, test_images, test_labels):
    x_test = cp.array(test_images.reshape(-1, 1, 28, 28))
    y_test = cp.array(test_labels)

    logits = model.forward(x_test)
    preds = cp.argmax(logits, axis=1)
    accuracy = cp.mean(preds == y_test)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

if __name__ == "__main__":
    train_images, train_labels = load_mnist("mnist_dataset", kind="train")
    test_images, test_labels = load_mnist("mnist_dataset", kind="t10k")

    model = LeNet5()
    train(model, train_images, train_labels, epochs=10, lr=0.001)
    test(model, test_images, test_labels)
