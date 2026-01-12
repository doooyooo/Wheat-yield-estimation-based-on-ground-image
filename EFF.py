import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
img_size = 1024

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# ======================
# 自定义数据集
# ======================
class SegmentationDataset(Dataset):
    def __init__(self, rgb_dir, mask_dir, image_size=380, transform=None):
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.images = [img for img in os.listdir(rgb_dir)
                       if img.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.rgb_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((self.image_size, self.image_size))(mask)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.1).float()  # 二值化

        return image, mask

# ======================
# EfficientNet-B4 分割模型
# ======================
class EfficientNetB4Segmenter(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetB4Segmenter, self).__init__()
        efficientnet = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.encoder = efficientnet.features  # 输出通道1792

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1792, 896, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(896, 448, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(448, 224, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(224, 112, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(112, 56, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(56, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ======================
# 计算多种验证指标
# ======================
def calculate_metrics(outputs, masks):
    preds = torch.sigmoid(outputs) > 0.5
    preds = preds.float()
    masks = masks.float()

    tp = (preds * masks).sum()
    fp = (preds * (1 - masks)).sum()
    fn = ((1 - preds) * masks).sum()
    tn = ((1 - preds) * (1 - masks)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return accuracy.item(), iou.item(), dice.item(), precision.item(), recall.item()

# ======================
# 训练函数
# ======================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50,
                model_name='efficientnetb4', task_type='segmentation'):
    os.makedirs(f'train_seg_{img_size}_model', exist_ok=True)
    best_val_loss = float('inf')
    history = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        # ---------- Train ----------
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for inputs, masks in train_iter:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            train_iter.set_postfix({'loss': loss.item()})

        train_loss = running_loss / len(train_loader.dataset)

        # ---------- Val ----------
        model.eval()
        val_loss = 0.0
        total_acc, total_iou, total_dice, total_prec, total_rec = 0, 0, 0, 0, 0

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for inputs, masks in val_iter:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)

                acc, iou, dice, prec, rec = calculate_metrics(outputs, masks)

                val_loss += loss.item() * inputs.size(0)
                total_acc += acc * inputs.size(0)
                total_iou += iou * inputs.size(0)
                total_dice += dice * inputs.size(0)
                total_prec += prec * inputs.size(0)
                total_rec += rec * inputs.size(0)

                val_iter.set_postfix({'val_loss': loss.item(), 'IoU': iou})

        val_loss /= len(val_loader.dataset)
        val_acc = total_acc / len(val_loader.dataset)
        val_iou = total_iou / len(val_loader.dataset)
        val_dice = total_dice / len(val_loader.dataset)
        val_prec = total_prec / len(val_loader.dataset)
        val_rec = total_rec / len(val_loader.dataset)

        scheduler.step(val_loss)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'val_precision': val_prec,
            'val_recall': val_rec
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'train_seg_{img_size}_model/{model_name}_best_{task_type}.pth')

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")

    # ---------- 保存曲线 ----------
    plt.figure(figsize=(10, 5))
    plt.plot([h['train_loss'] for h in history], label='Training Loss')
    plt.plot([h['val_loss'] for h in history], label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'train_seg_{img_size}_model/{model_name}_loss_curve_{task_type}.png')

    # ---------- 保存CSV ----------
    df = pd.DataFrame(history)
    df.to_csv(f'train_seg_{img_size}_model/{model_name}_history_{task_type}.csv', index=False)

    return model

# ======================
# 主函数
# ======================
def main():
    params = {
        'rgb_dir': r"..\dataset\spike\VGG\train",
        'mask_dir': r"..\dataset\spike\VGG\segmented",
        'image_size': img_size,
        'batch_size': 4,
        'epochs': 200,
        'lr': 1e-4,
        'num_workers': 4,
        'task_type': 'spike'
    }

    transform = transforms.Compose([
        transforms.Resize((params['image_size'], params['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = SegmentationDataset(
        rgb_dir=params['rgb_dir'],
        mask_dir=params['mask_dir'],
        image_size=params['image_size'],
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                              num_workers=params['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False,
                            num_workers=params['num_workers'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EfficientNetB4Segmenter(num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=params['epochs'],
        model_name='efficientnetb4',
        task_type=params['task_type']
    )

    torch.save(trained_model.state_dict(), f'train_seg_{img_size}_model/efficientnetb4_final_{params["task_type"]}.pth')
    print("EfficientNet-B4 model training completed and saved.")

if __name__ == "__main__":
    main()
