import torch
import timm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class CustomDataset(Dataset):
    def __init__(self, data_dir, classes, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            for img_path in cls_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    """训练模型并返回训练历史"""
    history = {
        'train_loss': [], 'train_acc': [],
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # 更新学习率
        scheduler.step()

        # 记录指标
        train_acc = train_correct / train_total

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}')

    return history


def evaluate_model(model, test_loader, device):
    """评估模型并返回各项指标"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    # 设置随机种子
    torch.manual_seed(42)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dir = '/Users/linghs/Downloads/train27/train27'
    test_dir = '/Users/linghs/Downloads/train27/val'

    classes = sorted([d.name for d in Path(train_dir).iterdir() if d.is_dir()])

    # 加载数据集
    train_dataset = CustomDataset(train_dir, classes, transform=transform)
    test_dataset = CustomDataset(test_dir, classes, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    num_classes = len(train_dataset.classes)

    # 模型配置
    models = {
        'mobilenet': timm.create_model('mobilenetv3_small_050', pretrained=True, num_classes=num_classes),
        'resnet': timm.create_model('resnet50', pretrained=True, num_classes=num_classes),
        'convnext': timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
    }

    # 训练参数
    num_epochs = 10
    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # 记录训练开始时间
        start_time = time.time()

        # 训练模型
        history = train_model(
            model, train_loader,
            criterion, optimizer, scheduler,
            num_epochs, device
        )

        # 记录训练时间
        training_time = time.time() - start_time

        # 评估模型
        metrics = evaluate_model(model, test_loader, device)

        results[model_name] = {
            'training_time': training_time,
            'history': history,
            'metrics': metrics
        }

    # 打印比较结果
    print("\nModel Comparison Results:")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Training Time: {result['training_time']:.2f} seconds")
        print(f"Test Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"Test Precision: {result['metrics']['precision']:.4f}")
        print(f"Test Recall: {result['metrics']['recall']:.4f}")
        print(f"Test F1 Score: {result['metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
