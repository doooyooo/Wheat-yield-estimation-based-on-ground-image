import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

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


# ---------------- 数据集 ----------------
class SegmentationDataset(Dataset):
    def __init__(self, rgb_dir, mask_dir, image_size=1024, transform=None):
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


# ---------------- Patch Embedding ----------------
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=1024, patch_size=32, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = x + self.pos_embed
        return x


# ---------------- ViT 分割模型 ----------------
class ViTSegmenter(nn.Module):
    def __init__(self, num_classes=1, image_size=1024, patch_size=32,
                 embed_dim=768, num_heads=12, num_layers=12, hidden_dim=3072):
        super(ViTSegmenter, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_grid_size = image_size // patch_size

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.to_feature_map = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2).view(
            x.size(0), -1, self.patch_grid_size, self.patch_grid_size
        )
        x = self.to_feature_map(x)
        x = self.decoder(x)
        return x


# ---------------- 评价指标 ----------------
def compute_metrics(outputs, masks, threshold=0.5):
    preds = (torch.sigmoid(outputs) > threshold).float()
    masks = masks.float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection / (union + 1e-6)).mean().item()

    dice = (2 * intersection / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)).mean().item()

    acc = (preds == masks).float().mean().item()

    return iou, dice, acc


# ---------------- 训练函数 ----------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50,
                model_name='model', task_type='spike'):
    os.makedirs('train_seg_1024_model', exist_ok=True)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_iou, running_dice, running_acc = 0.0, 0.0, 0.0, 0.0
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

            iou, dice, acc = compute_metrics(outputs, masks)
            running_loss += loss.item() * inputs.size(0)
            running_iou += iou * inputs.size(0)
            running_dice += dice * inputs.size(0)
            running_acc += acc * inputs.size(0)

            train_iter.set_postfix({'loss': loss.item(), 'IoU': iou, 'Dice': dice, 'Acc': acc})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_iou = running_iou / len(train_loader.dataset)
        epoch_dice = running_dice / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # ---------------- 验证 ----------------
        model.eval()
        val_loss, val_iou, val_dice, val_acc = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for inputs, masks in val_iter:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)

                iou, dice, acc = compute_metrics(outputs, masks)

                val_loss += loss.item() * inputs.size(0)
                val_iou += iou * inputs.size(0)
                val_dice += dice * inputs.size(0)
                val_acc += acc * inputs.size(0)

                val_iter.set_postfix({'val_loss': loss.item(), 'IoU': iou, 'Dice': dice, 'Acc': acc})

        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'train_seg_{img_size}_model/{model_name}_best_{task_type}.pth')

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}, Dice: {epoch_dice:.4f}, Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Acc: {val_acc:.4f}")

    # 绘制Loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'train_seg_{img_size}_model/{model_name}_loss_curve_{task_type}.png')
    return model


# ---------------- 主函数 ----------------
def main():
    params = {
        'rgb_dir': r"..\dataset\spike\VGG\train",
        'mask_dir': r"..\dataset\spike\VGG\segmented",
        'image_size': img_size,
        'batch_size': 4,
        'epochs': 200,
        'lr': 1e-5,
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

    model = ViTSegmenter(
        num_classes=1,
        image_size=1024,
        patch_size=32,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072
    ).to(device)

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
        model_name="vit",
        task_type=params['task_type']
    )

    torch.save(trained_model.state_dict(), f'train_seg_{img_size}_model/vit_final_{params["task_type"]}.pth')
    print("Model training completed and saved.")


if __name__ == "__main__":
    main()
