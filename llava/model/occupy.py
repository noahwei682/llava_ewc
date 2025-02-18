import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading

# 定义一个更复杂的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1000, 8192)  # 增加神经元数量
        self.fc2 = nn.Linear(8192, 8192)
        self.fc3 = nn.Linear(8192, 8192)
        self.fc4 = nn.Linear(8192, 10)  # 增加一层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# 设置参数
num_gpus = min(8, torch.cuda.device_count())
batch_size = 16384  # 增大批处理大小
num_epochs = 1000  # 无限循环

# 创建模型和优化器
def run_on_gpu(gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    while True:
        # 创建随机输入和标签
        inputs = torch.randn(batch_size, 1000).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 控制循环速度，确保持续占用 GPU
        time.sleep(0.01)  # 可以进一步减少以增加计算频率

# 启动线程
threads = []
for i in range(num_gpus):
    t = threading.Thread(target=run_on_gpu, args=(i,))
    t.start()
    threads.append(t)

# 等待线程完成（实际上不会停止）
for t in threads:
    t.join()