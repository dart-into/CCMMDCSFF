import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from ResNet_ImageNet import ACmix_ResNet


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row[60]
        label = row[61]
        image = load_and_preprocess_image(image_path)
        feature1 = torch.tensor(row[0:20].values.astype(float), dtype=torch.float32)
        feature2 = torch.tensor(row[20:40].values.astype(float), dtype=torch.float32)
        feature3 = torch.tensor(row[40:60].values.astype(float), dtype=torch.float32)
        return feature1, feature2, feature3, image, label


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size, Image.ANTIALIAS)
    image = transforms.ToTensor()(image)
    return image


# 创建自定义数据集
data = pd.read_csv('./load_front_gray_HSL_2_shuffle.csv', encoding='gbk')  # 请替换成你的CSV文件路径

mean = data.iloc[:, :60].mean()
std = data.iloc[:, :60].std()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
dataset = CustomDataset(data, transform=transform)

# 划分数据集为训练集和测试集
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


# 创建模型
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.change1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1, padding=0)
        self.change2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, padding=0)
        self.change3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, padding=0)

        self.feature1_layer1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.feature2_layer1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.feature3_layer1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same')

        self.tanh = nn.Sigmoid()

        self.feature1_layer2 = nn.MaxPool1d(2)
        self.feature2_layer2 = nn.MaxPool1d(2)
        self.feature3_layer2 = nn.MaxPool1d(2)

        self.feature1_layer3 = nn.BatchNorm1d(32)
        self.feature2_layer3 = nn.BatchNorm1d(32)
        self.feature3_layer3 = nn.BatchNorm1d(32)

        self.feature1_layer4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.feature2_layer4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.feature3_layer4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same')

        self.feature1_layer5 = nn.MaxPool1d(2)
        self.feature2_layer5 = nn.MaxPool1d(2)
        self.feature3_layer5 = nn.MaxPool1d(2)

        self.feature1_layer6 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.feature2_layer6 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.feature3_layer6 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same')

        self.feature1_layer7 = nn.MaxPool1d(kernel_size=2, padding=1)
        self.feature2_layer7 = nn.MaxPool1d(kernel_size=2, padding=1)
        self.feature3_layer7 = nn.MaxPool1d(kernel_size=2, padding=1)

        self.feature1_layer8 = nn.Dropout(0.5)
        self.feature2_layer8 = nn.Dropout(0.5)
        self.feature3_layer8 = nn.Dropout(0.5)

        self.resnet = models.resnet18(pretrained=True)

        self.fc1 = nn.Linear(2152, num_classes)

    def forward(self, feature1, feature2, feature3, image):
        feature1_out1 = self.feature1_layer1(feature1)
        feature2_out1 = self.feature2_layer1(feature2)
        feature3_out1 = self.feature3_layer1(feature3)

        feature1_act1 = self.tanh(feature1_out1)
        feature2_act1 = self.tanh(feature2_out1)
        feature3_act1 = self.tanh(feature3_out1)

        feature1_out2 = self.feature1_layer2(feature1_act1)
        feature2_out2 = self.feature2_layer2(feature2_act1)
        feature3_out2 = self.feature3_layer2(feature3_act1)

        feature1_out3 = self.feature1_layer3(feature1_out2)
        feature2_out3 = self.feature2_layer3(feature2_out2)
        feature3_out3 = self.feature3_layer3(feature3_out2)

        feature1_change1 = self.change1(feature1)
        feature2_change1 = self.change1(feature2)
        feature3_change1 = self.change1(feature3)

        # feature1_change1
        # 生成随机的索引，范围在 [0, tensor.size(2))，即最后一个维度的范围
        random_indices = torch.randint(0, feature1_change1.size(2), (10,))

        # 使用随机生成的索引来选择张量的最后一个维度上的元素
        feature1_change1 = feature1_change1[:, :, random_indices]

        # feature2_change1
        # 生成随机的索引，范围在 [0, tensor.size(2))，即最后一个维度的范围
        random_indices = torch.randint(0, feature2_change1.size(2), (10,))

        # 使用随机生成的索引来选择张量的最后一个维度上的元素
        feature2_change1 = feature2_change1[:, :, random_indices]

        # feature3_change3
        # 生成随机的索引，范围在 [0, tensor.size(2))，即最后一个维度的范围
        random_indices = torch.randint(0, feature3_change1.size(2), (10,))

        # 使用随机生成的索引来选择张量的最后一个维度上的元素
        feature3_change1 = feature3_change1[:, :, random_indices]

        feature1_out4 = self.feature1_layer4(feature1_out3 + feature1_change1)
        feature2_out4 = self.feature2_layer4(feature2_out3 + feature2_change1)
        feature3_out4 = self.feature3_layer4(feature3_out3 + feature3_change1)

        feature1_act2 = self.tanh(feature1_out4)
        feature2_act2 = self.tanh(feature2_out4)
        feature3_act2 = self.tanh(feature3_out4)

        feature1_out5 = self.feature1_layer5(feature1_act2)
        feature2_out5 = self.feature2_layer5(feature2_act2)
        feature3_out5 = self.feature3_layer5(feature3_act2)

        feature1_change2 = self.change2(feature1)
        feature2_change2 = self.change2(feature2)
        feature3_change2 = self.change2(feature3)

        # feature1_change2
        # 生成随机的索引，范围在 [0, tensor.size(2))，即最后一个维度的范围
        random_indices = torch.randint(0, feature1_change2.size(2), (5,))

        # 使用随机生成的索引来选择张量的最后一个维度上的元素
        feature1_change2 = feature1_change2[:, :, random_indices]

        # feature2_change2
        # 生成随机的索引，范围在 [0, tensor.size(2))，即最后一个维度的范围
        random_indices = torch.randint(0, feature2_change2.size(2), (5,))

        # 使用随机生成的索引来选择张量的最后一个维度上的元素
        feature2_change2 = feature2_change2[:, :, random_indices]

        # feature3_change2
        # 生成随机的索引，范围在 [0, tensor.size(2))，即最后一个维度的范围
        random_indices = torch.randint(0, feature3_change2.size(2), (5,))
        # 使用随机生成的索引来选择张量的最后一个维度上的元素
        feature3_change2 = feature3_change2[:, :, random_indices]

        feature1_change3 = self.change3(feature1_out3)
        feature2_change3 = self.change3(feature2_out3)
        feature3_change3 = self.change3(feature3_out3)

        # feature1_change3
        # 生成随机的索引，范围在 [0, tensor.size(2))，即最后一个维度的范围
        random_indices = torch.randint(0, feature1_change3.size(2), (5,))

        # 使用随机生成的索引来选择张量的最后一个维度上的元素
        feature1_change3 = feature1_change3[:, :, random_indices]

        # feature2_change3
        # 生成随机的索引，范围在 [0, tensor.size(2))，即最后一个维度的范围
        random_indices = torch.randint(0, feature2_change3.size(2), (5,))

        # 使用随机生成的索引来选择张量的最后一个维度上的元素
        feature2_change3 = feature2_change3[:, :, random_indices]

        # feature3_change3
        # 生成随机的索引，范围在 [0, tensor.size(2))，即最后一个维度的范围
        random_indices = torch.randint(0, feature3_change3.size(2), (5,))
        # 使用随机生成的索引来选择张量的最后一个维度上的元素
        feature3_change3 = feature3_change3[:, :, random_indices]

        feature1_out6 = self.feature1_layer6(
            feature1_out5 + feature1_change2 + feature1_change3 + feature2_out5 + feature3_out5)
        feature2_out6 = self.feature2_layer6(
            feature2_out5 + feature2_change2 + feature2_change3 + feature1_out5 + feature3_out5)
        feature3_out6 = self.feature3_layer6(
            feature3_out5 + feature3_change2 + feature3_change3 + feature1_out5 + feature2_out5)

        feature1_act3 = self.tanh(feature1_out6)
        feature2_act3 = self.tanh(feature2_out6)
        feature3_act3 = self.tanh(feature3_out6)

        feature1_out7 = self.feature1_layer7(feature1_act3)
        feature2_out7 = self.feature2_layer7(feature2_act3)
        feature3_out7 = self.feature3_layer7(feature3_act3)

        feature1_out8 = self.feature1_layer8(feature1_out7)
        feature2_out8 = self.feature2_layer8(feature2_out7)
        feature3_out8 = self.feature3_layer8(feature3_out7)

        image_out = self.resnet(image)

        feature1_out9 = feature1_out8.view(feature1_out8.size(0), -1)
        feature2_out9 = feature2_out8.view(feature2_out8.size(0), -1)
        feature3_out9 = feature3_out8.view(feature3_out8.size(0), -1)
        image_out = image_out.view(image_out.size(0), -1)

        # 最小-最大缩放处理
        feature1_out10 = (feature1_out9 - torch.min(feature1_out9)) / (
                torch.max(feature1_out9) - torch.min(feature1_out9))
        feature2_out10 = (feature2_out9 - torch.min(feature2_out9)) / (
                torch.max(feature2_out9) - torch.min(feature2_out9))
        feature3_out10 = (feature3_out9 - torch.min(feature3_out9)) / (
                torch.max(feature3_out9) - torch.min(feature3_out9))
        image_out = (image_out - torch.min(image_out)) / (torch.max(image_out) - torch.min(image_out))

        combined = torch.cat([feature1_out10, feature2_out10, feature3_out10, image_out], dim=1)
        output = self.fc1(combined)
        return output


# 创建模型实例，指定类别数量
num_classes = 18  # 你的分类类别数量
model = MyModel(num_classes)

# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 35
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for features1, features2, features3, images, labels in train_loader:
        optimizer.zero_grad()
        features1 = features1.unsqueeze(1)  # 添加一个维度表示通道数
        features2 = features2.unsqueeze(1)  # 添加一个维度表示通道数
        features3 = features3.unsqueeze(1)  # 添加一个维度表示通道数
        outputs = model(features1, features2, features3, images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += labels.size(0)
        total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

    train_accuracy = total_correct / total_samples
    print(
        f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {total_loss / len(train_loader):.4f} - Training Accuracy: {train_accuracy:.4f}')

# 在训练循环之外，计算测试集的正确率
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for features1, features2, features3, images, labels in test_loader:
        features1 = features1.unsqueeze(1)  # 添加一个维度表示通道数
        features2 = features2.unsqueeze(1)  # 添加一个维度表示通道数
        features3 = features3.unsqueeze(1)  # 添加一个维度表示通道数

        outputs = model(features1, features2, features3, images)
        total_samples += labels.size(0)
        total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

test_accuracy = total_correct / total_samples
print(f'Test Accuracy: {test_accuracy:.4f}')

# 在整体测试准确率后面添加计算每个类别的正确率
class_correct = {i: 0 for i in range(num_classes)}
class_total = {i: 0 for i in range(num_classes)}

model.eval()
with torch.no_grad():
    for features1, features2, features3, images, labels in test_loader:
        features1 = features1.unsqueeze(1)
        features2 = features2.unsqueeze(1)
        features3 = features3.unsqueeze(1)

        outputs = model(features1, features2, features3, images)
        _, predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i].item()  # 获取标签的整数值
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 输出每个类别的正确率
for i in range(num_classes):
    if class_total[i] > 0:
        print(f"Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%")
    else:
        print(f"No test samples for class {i}")
