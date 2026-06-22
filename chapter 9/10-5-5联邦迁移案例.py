import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# 1. 数据集预处理类
class SplitFashionMNIST(Dataset):
    def __init__(self, dataset, split_type='top', has_labels=True):
        """
        初始化分割后的FashionMNIST数据集
        Args:
            dataset: 原始FashionMNIST数据集
            split_type: 'top' 或 'bottom'，表示图像的上半部分或下半部分
            has_labels: 是否包含标签
        """
        self.dataset = dataset
        self.split_type = split_type
        self.has_labels = has_labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.has_labels:
            img, label = self.dataset[idx]
        else:
            img = self.dataset[idx][0]
            label = -1  # 无标签数据
        
        # 垂直分割图像(28x28 -> 14x28)
        if self.split_type == 'top':
            split_img = img[:, :14, :]  # 上半部分
        else:
            split_img = img[:, 14:, :]  # 下半部分
            
        return split_img, label

# 2. 特征提取器模型
class PartialFeatureExtractor(nn.Module):
    def __init__(self, input_height=14):
        """
        初始化部分特征提取器
        Args:
            input_height: 输入图像高度(分割后为14)
        """
        super(PartialFeatureExtractor, self).__init__()
        
        # CNN特征提取网络
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 计算全连接层输入维度
        self._init_fc_dim(input_height)
        
    def _init_fc_dim(self, input_height):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_height, 28)
            dummy_output = self.conv_layers(dummy_input)
            self.fc_input_dim = dummy_output.numel()
            
    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return features

# 3. 特征融合分类器
class FusionClassifier(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, num_classes=10):
        """
        初始化融合分类器
        Args:
            input_dim_a: 客户端A特征维度
            input_dim_b: 客户端B特征维度
            num_classes: 分类类别数
        """
        super(FusionClassifier, self).__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim_a + input_dim_b, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, num_classes)
        )
        
    def forward(self, features_a, features_b):
        combined = torch.cat([features_a, features_b], dim=1)
        return self.fc_layers(combined)

# 4. 联邦迁移学习训练过程
class FederatedTransferLearning:
    def __init__(self, clientA, clientB, fusion_model, device='cuda'):
        self.clientA = clientA
        self.clientB = clientB
        self.fusion_model = fusion_model
        self.device = device
        
    def train_step(self, batch_a, batch_b, optimizer):
        self.fusion_model.train()
        
        # 提取特征
        features_a = self.clientA.forward(batch_a[0].to(self.device))
        features_b = self.clientB.forward(batch_b[0].to(self.device))

        # 融合预测
        outputs = self.fusion_model(features_a, features_b)
        loss = nn.CrossEntropyLoss()(outputs, batch_a[1].to(self.device))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, test_loader_a, test_loader_b):
        self.fusion_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_a, batch_b in zip(test_loader_a, test_loader_b):
                features_a = self.clientA.forward(batch_a[0].to(self.device))
                features_b = self.clientB.forward(batch_b[0].to(self.device))

                outputs = self.fusion_model(features_a, features_b)
                _, predicted = outputs.max(1)
                
                total += batch_a[1].size(0)
                correct += predicted.eq(batch_a[1].to(self.device)).sum().item()
                
        return correct / total
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 加载数据集
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建分割数据集
    clientA_train = SplitFashionMNIST(train_dataset, split_type='top', has_labels=True)
    clientB_train = SplitFashionMNIST(train_dataset, split_type='bottom', has_labels=False)
    clientA_test = SplitFashionMNIST(test_dataset, split_type='top', has_labels=True)
    clientB_test = SplitFashionMNIST(test_dataset, split_type='bottom', has_labels=False)
    
    # 创建数据加载器
    train_loader_a = DataLoader(clientA_train, batch_size=64, shuffle=True)
    train_loader_b = DataLoader(clientB_train, batch_size=64, shuffle=True)
    test_loader_a = DataLoader(clientA_test, batch_size=64, shuffle=False)
    test_loader_b = DataLoader(clientB_test, batch_size=64, shuffle=False)
    
    # 初始化模型
    feature_extractor_a = PartialFeatureExtractor().to(device)
    feature_extractor_b = PartialFeatureExtractor().to(device)
    
    # 计算特征维度
    dummy_input = torch.randn(1, 1, 14, 28).to(device)
    feature_dim = feature_extractor_a(dummy_input).shape[1]
    
    fusion_model = FusionClassifier(feature_dim, feature_dim).to(device)
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(
        list(feature_extractor_a.parameters()) + 
        list(feature_extractor_b.parameters()) + 
        list(fusion_model.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 创建联邦迁移学习实例
    ftl = FederatedTransferLearning(
        clientA=feature_extractor_a,
        clientB=feature_extractor_b,
        fusion_model=fusion_model,
        device=device
    )
    
    # 初始化记录列表
    training_records = {
        'epoch': [],
        'loss': [],
        'accuracy': []
    }
    
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (batch_a, batch_b) in enumerate(zip(train_loader_a, train_loader_b)):
            loss = ftl.train_step(batch_a, batch_b, optimizer)
            total_loss += loss
            
        scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss/len(train_loader_a)
        
        # 评估当前epoch的性能
        acc = ftl.evaluate(test_loader_a, test_loader_b)
        
        # 记录训练数据
        training_records['epoch'].append(epoch + 1)
        training_records['loss'].append(avg_loss)
        training_records['accuracy'].append(acc * 100)
        
        # 每5轮打印一次结果
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Acc: {acc*100:.2f}%')
    
    # 将结果保存为CSV文件

    
    # 创建DataFrame
    results_df = pd.DataFrame(training_records)
    
    # 保存为CSV文件
    results_df.to_csv('federated_transfer_learning_results.csv', index=False)
    print("\n训练结果已保存至 federated_transfer_learning_results.csv")
    

if __name__ == '__main__':
    main()