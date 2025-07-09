import torch
import torch.nn as nn
import torchaudio.transforms as T
import os
import torchaudio
import sys
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchaudio.transforms import MelSpectrogram, Resample, TimeStretch, FrequencyMasking, TimeMasking
from data_process import prepare_datasets
from model import BabyCryClassifier
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report
from resnet import AudioClassifier
from AlexNet import AudioClassifierAlexNet
import torch.nn.functional as F
from torch.optim import AdamW
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
from Resnet_BiLSTM import ResNet50DualAttention


if __name__ == "__main__":
    DATA_ROOT = "D:/pycharmproject/baby_cry_classify/baby_cry_data/split_train/"
    #DATA_ROOT = "D:/pycharmproject/baby_cry_classify/audio/"
    #model = BabyCryClassifier()

    model = ResNet50DualAttention().to(device)
    model.fc = nn.Linear(2048, 10).to(device)
    state_dict = torch.load("best_BiLSTM_model_val.pth")
    model.load_state_dict(state_dict["model_state_dict"])
    # model = AudioClassifierAlexNet().to(device)
    #model.resnet.fc = nn.Linear(2048, 6)
    model.fc = nn.Linear(2048, 6).to(device)
    """
    # 冻结除全连接层外的所有参数
    for param in model.parameters():
        param.requires_grad = False  # 冻结卷积层等预训练参数

    # 仅允许全连接层参与训练
    for param in model.resnet.fc.parameters():
        param.requires_grad = True  # 解冻全连接层
    """

    learning_rate = 0.001
    epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05)
    #optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_loader, test_loader = prepare_datasets(DATA_ROOT)

    model = model.to(device)

    best_val_loss = float('inf')
    best_model_path = "best_BiLSTM_model2_val.pth"
    train_losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = F.cross_entropy(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()  # 更新学习率

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best_model_path)

    # 步骤1：初始化模型结构（必须与保存时的架构完全一致）
    final_model = ResNet50DualAttention().to(device)
    final_model.fc = nn.Linear(2048, 6).to(device)
    #final_model.resnet.fc = nn.Linear(2048, 6).to(device)
    #final_model = models.resnet18(weights=None).to(device)
    # 步骤2：加载参数
    state_dict = torch.load("best_BiLSTM_model2_val.pth")
    final_model.load_state_dict(state_dict["model_state_dict"])
    final_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device,dtype=torch.float32)
            y_test = y_test.to(device)
            with autocast():
                outputs = final_model(x_test)
                loss = criterion(outputs, y_test)

            # 统计基础指标
            test_loss += loss.item() * x_test.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

            # 保存详细预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

    # 计算综合指标
    avg_test_loss = test_loss / total
    test_accuracy = 100 * correct / total
    print(f"\nTest Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
