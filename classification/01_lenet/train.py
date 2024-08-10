import numpy as np
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # 训练集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)

    # 测试集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000, shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)  # 转换成迭代器
    val_image, val_label = next(val_data_iter) # 通过next函数获取一批数据

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):
        running_loss = 0.0  # 累加在训练过程中的损失
        for step, data in enumerate(train_loader, start=0):  # 遍历训练集样本，返回每一批数据data和对应的step步数
            inputs, labels = data  # 将数据分离成input和label
            optimizer.zero_grad()  # 清零历史损失梯度
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 每次计算完loss之后累加到running_loss
            if step % 500 == 499:  # 每隔500step打印一次信息
                with torch.no_grad():  # 不去计算每个节点的误差损失梯度
                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)  # 通过item拿到对应数值

                    print('[%d, %5d] train_loss: %.3f val_acc: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))  # 输出epoch、step、平均训练误差和val_acc
                    running_loss = 0.0

        print('Finished Training')
        save_path = './Lenet.pth'
        torch.save(net.state_dict(), save_path)  # 保存网络所有参数


if __name__ == '__main__':
    main()


