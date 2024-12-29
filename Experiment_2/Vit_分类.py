import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from pathlib import Path

from tqdm import tqdm
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 使用卷积层进行 patch embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # 输入 x 的形状: (B, C, H, W)
        x = self.proj(x)  # (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, H//patch_size * W//patch_size)
        x = x.transpose(1, 2)  # (B, H//patch_size * W//patch_size, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            n_heads=12,
            num_layers=12,
            mlp_ratio=4,
            dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 定义 TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
        )

        # 定义 TransformerEncoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x):
        # 输入 x 的形状: (B, N, C)
        # 转换为 (N, B, C) 以适应 TransformerEncoder
        x = x.transpose(0, 1)  # (N, B, C)
        x = self.transformer_encoder(x)  # (N, B, C)
        x = x.transpose(0, 1)  # (B, N, C)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_classes=1000,
            embed_dim=768,
            depth=3,
            n_heads=8,
            mlp_ratio=4,
            dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.n_patches = self.patch_embed.n_patches

        # 添加 class token 和 position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        # 使用封装的 TransformerEncoder
        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            num_layers=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, C)

        # 添加 class token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, C)

        # 添加 position embedding
        x = x + self.pos_embed

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (B, N+1, C)

        # 提取 class token 并分类
        cls_token_final = x[:, 0]  # (B, C)
        x = self.norm(cls_token_final)
        x = self.head(x)  # (B, n_classes)
        return x


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform, classes):
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.transform = transform
        # 用于存储预处理后的特征和标签
        self.processed_features = []
        self.labels = []

        # 在初始化时就完成所有图像的预处理
        self._preprocess_all_images()

    def _preprocess_all_images(self):
        print("开始预处理图像...")
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            for img_path in cls_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # 加载并预处理图像
                    image = Image.open(img_path).convert('RGB')
                    processed_image = self.transform(image)

                    # 存储预处理后的特征和对应的标签
                    self.processed_features.append(processed_image)
                    self.labels.append(self.class_to_idx[cls])

        # 将处理好的特征转换为一个大的张量
        self.processed_features = torch.stack(self.processed_features)
        self.labels = torch.tensor(self.labels)
        print(f"预处理完成！共处理了 {len(self.labels)} 张图像")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 直接返回预处理好的特征和标签
        return self.processed_features[idx], self.labels[idx]

    def get_all_data(self):
        """返回所有预处理好的数据"""
        return self.processed_features, self.labels


def get_data_loaders(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    classes = sorted([d.name for d in Path(train_dir).iterdir() if d.is_dir()])

    train_dataset = CustomImageDataset(train_dir, transform=transform, classes=classes)
    test_dataset = CustomImageDataset(test_dir, transform=transform, classes=classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=2)

    return train_loader, test_loader, len(train_dataset.classes)


def train_model(model, trainloader, epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/image_classifier_experiment')

    model = model.to(device)
    global_step = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            # Log training loss to TensorBoard
            writer.add_scalar('Training/BatchLoss', loss.item(), global_step)

            if i % 20 == 19:
                avg_loss = running_loss / 20
                print(f'[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}')
                writer.add_scalar('Training/AverageLoss', avg_loss, global_step)
                running_loss = 0.0

        # Calculate and log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Training/LearningRate', current_lr, epoch)

    writer.close()


def evaluate_model(model, testloader, device='cuda'):
    writer = SummaryWriter('runs/image_classifier_evaluation')

    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, pred in zip(labels, predicted):
                label_idx = label.item()
                if label_idx not in class_correct:
                    class_correct[label_idx] = 0
                    class_total[label_idx] = 0
                class_total[label_idx] += 1
                if label_idx == pred.item():
                    class_correct[label_idx] += 1

    # Log overall accuracy to TensorBoard
    overall_accuracy = 100 * correct / total
    writer.add_scalar('Evaluation/OverallAccuracy', overall_accuracy, 0)
    print(f'总体准确率: {overall_accuracy:.2f}%')

    # Log per-class accuracy to TensorBoard
    for idx in class_correct.keys():
        class_accuracy = 100 * class_correct[idx] / class_total[idx]
        writer.add_scalar(f'Evaluation/ClassAccuracy/Class_{idx}', class_accuracy, 0)
        print(f'类别 {idx} 准确率: {class_accuracy:.2f}%')

    writer.close()


import time


def main():
    train_dir = '/Users/linghs/Downloads/train27/train27'
    test_dir = '/Users/linghs/Downloads/train27/val'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    trainloader, testloader, num_classes = get_data_loaders(train_dir, test_dir)
    print(f'类别数量: {num_classes}')

    model = VisionTransformer(n_classes=num_classes)
    start_time = time.time()
    train_model(model, trainloader, epochs=10, device=device)
    evaluate_model(model, testloader, device=device)
    print(f"cost time:{time.time() - start_time}")  # cost time:158.05460810661316

    torch.save(model.state_dict(), 'image_vit_classifier.pth')


if __name__ == '__main__':
    main()
