import time
import torch
from .snn import SNN
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.clock_driven import functional


def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    input_dims = 28
    epochs = 10
    times = 10

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    # 加载数据集 并转换为张量

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # 创建数据加载器

    model = SNN(input_dims, times).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 初始化模型 损失函数和优化器

    print("Start Training")

    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            functional.reset_net(model)
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        # 训练模型

        train_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            functional.reset_net(model)
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        # 评估模型

        acc = 100 * correct / total

        print(f"Epoch: {epoch+1}, Loss: {train_loss:.2f}, Accuracy: {acc:.2f}%, Time: {time.time() - start_time:.2f}")

    model_path = "./model/model.pt"
    torch.save(model.state_dict(), model_path)
    print("Model saved")
    # 保存模型
    # 十轮训练准确率为98%




