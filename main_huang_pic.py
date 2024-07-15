# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:27:18 2024

@author: 37458
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, resnet34, resnet50 
from torchvision.models import alexnet
from torchvision.models import vgg16, vgg19
from torchvision.models import squeezenet1_0, squeezenet1_1
from torchvision.models import densenet121, densenet169, densenet201
from torchvision.models import inception_v3
from torchvision.models import wide_resnet50_2, wide_resnet101_2
from torchvision.models import mobilenet_v2
from torchvision.models import shufflenet_v2_x1_0, shufflenet_v2_x0_5
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from MLLSTM import MLLSTM
# from Bamboo import BambooNet
from dnn1_40 import MultiModalAttentionModel,DNN4,SimpleCNN,ShuffleNetBinary,DeepFC,DeepLSTM,ResNetBinary
# from dnn1_40_two import MultiModalAttentionModel,LSTM,ShuffleNetBinary,ResNetBinary,EfficientNetCustom
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc

path = 'D:/博士生/学习/论文发表/是否包含集群二分类/data9/'
# path = 'D:/博士生/学习/论文发表/是否包含集群二分类/test/'
num_epochs = 50
rows = 27

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        file_names = os.listdir(path)
        self.imgs = []

        class_0_samples = 0
        class_1_samples = 0
        
        for file_name in file_names:
            label = int(file_name.split('_')[0])
            img_path = os.path.join(path, file_name)
            fh = np.loadtxt(img_path, dtype=np.float32)
            has_nan = np.isnan(fh).any()
            if has_nan:
                print(f"在文件 {file_name}中发现了NaN值")
                
            num_samples = int(fh.shape[0] / rows)
        
            if label == 0:
                class_0_samples += num_samples
            elif label == 1:
                class_1_samples += num_samples
        
            for i in range(int(fh.shape[0] / rows)):
                img = fh[i * rows:(i + 1) * rows, :]
                # img = fh[i * rows+1: i * rows + 2, :]
                if not np.isnan(img).any():
                    self.imgs.append((img, label))

        self.transform = transform
        print("类别 0 的样本数:", class_0_samples)
        print("类别 1 的样本数:", class_1_samples)
        
    def __getitem__(self, index):
        img, label = self.imgs[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
    
    def get_total_samples(self):
        return len(self.imgs)
    

# 创建整个数据集
shuffled_dataset = MyDataset(path, transform=transforms.ToTensor())

    
# 重新划分成训练集和测试集
train_ratio = 0.7
train_size = int(train_ratio * len(shuffled_dataset))
test_size = len(shuffled_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(shuffled_dataset, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
            true_positive += ((predicted == 1) & (label == 1)).sum().item()
            false_positive += ((predicted == 1) & (label == 0)).sum().item()
            false_negative += ((predicted == 0) & (label == 1)).sum().item()
            true_negative += ((predicted == 0) & (label == 0)).sum().item()

    accuracy = 100 * correct / total
    # 计算灵敏度 (Sensitivity)
    sensitivity = true_positive / (true_positive + false_negative +0.1)
    # 计算阳性预测值 (Positive Predictive Value, PPV)
    ppv = true_positive / (true_positive + false_positive+0.1)
    # 计算特异度 (Specificity)
    specificity = true_negative / (true_negative + false_positive+0.1)
    # 计算 F1 值 (F1 measure)
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity+0.1)
    # print(f'True Positive: {true_positive}')
    # print(f'False Positive: {false_positive}')
    # print(f'False Negative: {false_negative}')
    # print(f'True Negative: {true_negative}')
    return accuracy, sensitivity, specificity, ppv, f1

# 使用resnet
# model = resnet18(num_classes=2)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)


# lstm_model = DeepLSTM(input_size=60, hidden_size=128*4, output_size=4)#80多
# cnn_model = SimpleCNN(output_size=4)#90.6
# dnn_model = DNN4(output_size=4)#80多
# ShuffleNet_model = ShuffleNetBinary(output_size=4)#87
# model = MultiModalAttentionModel(lstm_model, cnn_model, dnn_model, ShuffleNet_model, attention_heads=4, output_size=16)

# model = ShuffleNetBinary(output_size=2)
# model = MLLSTM(input_size=60)
# print(model)

# model = ShuffleNetBinary(output_size=2)
# lstm_model = LSTM(input_size=400, hidden_size=64, output_size=4)#76多

# lstm_model = ShuffleNetBinary(output_size=2)
# ShuffleNet_model = ResNetBinary(output_size=4)
# ShuffleNet_model = EfficientNetCustom(output_size=2)
# model = MultiModalAttentionModel(lstm_model, ShuffleNet_model, attention_heads=4, output_size=4)#76
# model = EfficientNetCustom(output_size=2)

model = ResNetBinary(output_size=2)

# model = ShuffleNetBinary(input_channels=24, output_size=2)

# model = mobilenet_v2(pretrained=True)
# model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # 调整输入通道
# model.classifier[1] = nn.Linear(1280, 2)  # 2 是输出的类别数

# model = wide_resnet101_2(pretrained=True)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)  # 调整输入通道
# model.fc = nn.Linear(2048, 2)  # 2 是输出的类别数

# model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)  # 你可以根据需要选择不同的 EfficientNet 版本
# model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, bias=False)  # 调整输入通道数

# model = shufflenet_v2_x1_0(pretrained=True)  # You can change the version if needed
# model.fc = nn.Linear(1024, 2)  # 更改全连接层的输出类别为2
# model._modules['conv1'][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 调用模型的 forward 方法
model = model.to(device)
# model.to(device)

weight=torch.from_numpy(np.array([0.5,0.5])).float()#.cuda()
criterion = nn.CrossEntropyLoss(weight=weight)
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 创建余弦退火学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

def evaluate_model_with_roc(model, dataloader, device):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            output = model(data)
            all_scores.append(output.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    fpr, tpr, _ = roc_curve(all_labels, all_scores[:, 1])  # 取出1类的概率作为预测值
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def train_with_overfitting_handling(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, device, num_epochs, print_freq=9000):
    accuracies = []  # 用于存储每个epoch的accuracy
    roc_aucs = []  # 用于存储每个epoch的ROC AUC值

    for epoch in range(num_epochs):
        model.train()
        
        for i, (data, label) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()  # 在每个 iteration 结束后更新学习率
            if (i + 1) % print_freq == 0:
                print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_dataloader), loss.item(), scheduler.get_lr()[0]))

        accuracy, sensitivity, specificity, ppv, f1 = evaluate_model(model, test_dataloader, device)
        accuracies.append(accuracy)  # 将accuracy添加到列表中
        print('Epoch [{}/{}], Accuracy: {:.2f}%, Sensitivity: {:.2f}, Specificity: {:.2f}, PPV: {:.2f}, F1: {:.2f}'.format(epoch + 1, num_epochs, accuracy, sensitivity, specificity, ppv, f1))

        fpr, tpr, roc_auc = evaluate_model_with_roc(model, test_dataloader, device)
        roc_aucs.append(roc_auc)

        if ((epoch + 1) % 5 == 0):
            torch.save(model.state_dict(), "mymodule_{}.pth".format((epoch + 1)))

    # 绘制accuracy变化曲线
    plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.savefig("accuracy_curve.png")  # 保存曲线图
    plt.show()

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")  # 保存ROC曲线图
    plt.show()

    # 保存accuracy和ROC AUC值数据到Excel文档
    result_data = pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Accuracy': accuracies, 'ROC_AUC': roc_aucs})
    result_data.to_excel("result_data_{}.xlsx".format(model.__class__.__name__), index=False)

train_with_overfitting_handling(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, device, num_epochs)