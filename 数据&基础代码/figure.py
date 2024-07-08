import re
import csv
import matplotlib.pyplot as plt

# 定义日志文件路径
log_file_path = 'output.log'
# 定义输出CSV文件路径
csv_file_path = 'output-2.csv'

# 正则表达式模式，用于匹配Acc和Loss
pattern = re.compile(r'Acc=([\d.]+), Loss=([\d.]+)')

# 存储Acc和Loss的列表
acc_list = []
loss_list = []

# 读取日志文件并提取Acc和Loss值
with open(log_file_path, 'r') as file:
    for step, line in enumerate(file, start=1):
        match = pattern.search(line)
        if match:
            acc = float(match.group(1))
            loss = float(match.group(2))
            acc_list.append((step, acc, loss))

# 将提取到的数据写入CSV文件
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Step', 'Acc', 'Loss'])  # 写入表头
    csv_writer.writerows(acc_list)  # 写入数据

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
plt.savefig('acc&loss-2.png')
plt.show()
