import pandas as pd
import csv
import matplotlib.pyplot as plt

# 定义输出CSV文件路径
csv_file_path = 'training_log.csv'

# # 绘制 Accuracy 曲线图
# plt.figure(figsize=(10, 5))
# plt.plot(data['step'], data['accuracy'], label='Accuracy')
# plt.xlabel('Step')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy Progress')
# plt.legend()
# plt.grid(True)
# plt.savefig('training_accuracy_progress.png')
# plt.show()

# # 绘制 Loss 曲线图
# plt.figure(figsize=(10, 5))
# plt.plot(data['step'], data['loss'], label='Loss', color='red')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.title('Training Loss Progress')
# plt.legend()
# plt.grid(True)
# plt.savefig('training_loss_progress.png')
# plt.show()

# 从CSV文件中读取数据
steps = []
acc_values = []
loss_values = []

with open(csv_file_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # 跳过表头
    for row in csv_reader:
        steps.append(int(row[0]))
        acc_values.append(float(row[1]))
        loss_values.append(float(row[2]))

# 绘制Acc和Loss的图表
plt.figure(figsize=(12, 6))

# 绘制Acc的图表
plt.subplot(1, 2, 1)
plt.plot(steps, acc_values, label='Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Accuracy over Steps')
plt.legend()

# 绘制Loss的图表
plt.subplot(1, 2, 2)
plt.plot(steps, loss_values, label='Loss', color='red')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over Steps')
plt.legend()

plt.tight_layout()
# 保存图表到文件
plt.savefig('acc&loss-3.png')
plt.show()