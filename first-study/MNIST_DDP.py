import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 这个有点问题两次结果不好

#它基于 Pytorch 官方在 MNIST 上创建和训练模型的 示例。


class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
#把模型放入 CUDA 设备:
device = "cuda"

#构建一些基本的 PyTorch DataLoaders:

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

train_dset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=64)

model = BasicNet().to(device)

#构建 PyTorch optimizer (优化器):
optimizer = optim.AdamW(model.parameters(), lr=1e-3)


#评估循环会计算训练后模型在测试数据集上的准确度：
model.train()
# for batch_idx, (data, target) in enumerate(train_loader):
#     data, target = data.to(device), target.to(device)
#     output = model(data)
#     loss = F.nll_loss(output, target)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()


# 创建进程组函数
def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    # 设置端口和ip
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12300"

    # Initialize the process group
    #当前进程的rank,用于标识不同的进程。"gloo":使用gloo后端进行分布式通信。rank和world_size用于标识每个进程并交互协作
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
# 销毁
def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()



def train(model, rank, world_size):
    setup(rank, world_size)
    #将模型移动到该进程对应的设备上,设备id即为rank。
    model = model.to(rank)
    #device_ids设置为当前rank,表示该DDP副本在对应的设备上。
#这样每个进程上都存在一个DDP模型副本,它们分布在不同设备上
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)
    # Train for one epoch
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    cleanup()


model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        #output batch和预测的10个类别的置信度。然后就找最大
        pred = output.argmax(dim=1, keepdim=True)
        #目标标签 reshape 为和pred同样的形状
        correct += pred.eq(target.view_as(pred)).sum().item()
print(f'Accuracy: {100. * correct / len(test_loader.dataset)}')
