# coding=utf-8

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from resnet import ResNet18
from FishEyeDataset import FishEyeDataset
from tensorboardX import SummaryWriter

writer = SummaryWriter('./log')
# writer.add_scalar(name+'/TrainLoss', train_loss, epoch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['无鱼眼', '有鱼眼']

if torch.cuda.is_available():
    print(torch.cuda.device_count() ) # 返回GPU数目
    print(torch.cuda.get_device_name(0) ) # 返回GPU名称，设备索引默认从0开始
    print(torch.cuda.current_device())  # 返回当前设备索引

# 超参数
EPOCH = 150
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128
LR = 0.1

# 准备数据集并预处理
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
#     transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert image to tensor.
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Subtract mean
#     std=[0.229, 0.224, 0.225]     # Divide by standard deviation
#     )])

dataset = FishEyeDataset(root='../', train=True)
dataset = FishEyeDataset(root='../', train=True)
dataset_length = len(dataset)
train_size = int(0.8*dataset_length)
train_set, validate_set = torch.utils.data.random_split(dataset, [train_size, dataset_length - train_size])
print("train_size: ", train_size, "dataset_size: ", dataset_length)
trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(validate_set, batch_size=BATCH_SIZE, shuffle=True)

# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# Net structure
# from torchsummary import summary
# summary(net, (3, 336, 192))

# 训练
if __name__ == "__main__":
    best_acc = 0
    print("Start Training, Resnet-18!")

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for step, batch_data in enumerate(trainloader):
            # 准备数据
            length = len(trainloader)
            # print("trainloader: ", length)

            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            # print(outputs.shape, outputs, labels)

            loss = criterion(outputs, labels)
            # print(loss, labels.shape)

            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (step + 1 + epoch * length), sum_loss / (step + 1), 100. * correct / total))
            writer.add_scalar('TrainLoss', sum_loss, step + epoch * length)
            writer.add_scalar('TrainAcc', 100. * correct / total, step + epoch * length)

        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            acc = 100. * correct / total
            print('测试分类准确率为：%.3f%%' % (acc))
            writer.add_scalar('TestAcc', acc, epoch)

            # # 将每次测试结果实时写入acc.txt文件中
            # print('Saving model......')
            # torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))



