import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List
import os
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -----------------------
# Dataset
# -----------------------
class FundusDataset(Dataset):
    def __init__(self, images, masks, domain_labels, transform=None):
        self.images = images
        self.masks = masks
        self.domain_labels = domain_labels
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)

        image = image.astype(np.float32) / 255.0
        return {
            'image': torch.FloatTensor(image),
            'mask': torch.LongTensor(mask)
        }

# -----------------------
# U-Net backbone
# -----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5); x = torch.cat([x4, x], dim=1); x = self.conv1(x)
        x = self.up2(x);  x = torch.cat([x3, x], dim=1); x = self.conv2(x)
        x = self.up3(x);  x = torch.cat([x2, x], dim=1); x = self.conv3(x)
        x = self.up4(x);  x = torch.cat([x1, x], dim=1); x = self.conv4(x)

        return self.outc(x)

# -----------------------
# Loss: CE + Dice + EDT
# -----------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0,3,1,2).float()
        intersection = (inputs * targets).sum(dim=(2,3))
        union = inputs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = (2.*intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class EDTLoss(nn.Module):
    """欧几里得距离变换损失（对每个前景类别分别计算）"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, preds, targets):
        # preds: (B, C, H, W)，包含背景、视盘、视杯
        # targets: (B, H, W)，取值 ∈ {0,1,2}
        preds = F.softmax(preds, dim=1)  # 转成概率分布
        B, C, H, W = preds.shape
        loss_per_sample = []

        for b in range(B):
            target_np = targets[b].cpu().numpy()
            dist_map = np.zeros((C, H, W), dtype=np.float32)

            for cls in range(1, C):  # 跳过背景（cls=0）
                mask = (target_np == cls).astype(np.uint8)
                if mask.sum() > 0:
                    # 背景到目标边界的距离
                    dist = distance_transform_edt(1 - mask)
                    #对距离做归一化
                    dist = dist / (dist.max() + 1e-8)
                    dist_map[cls] = dist

            dist_map = torch.from_numpy(dist_map).to(preds.device)
            # 对每个类别的预测概率加权惩罚
            loss_sample = (preds[b] * dist_map).sum(dim=(1, 2)).mean()
            loss_per_sample.append(loss_sample)

        return self.weight * torch.stack(loss_per_sample).mean()


class SegmentationLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, edt_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.5, 1.0], device=device))
        self.dice = DiceLoss()
        self.edt = EDTLoss(weight=edt_weight)
        self.ce_w, self.dice_w, self.edt_w = ce_weight, dice_weight, edt_weight

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        edt_loss = self.edt(preds, targets)

        total_loss = (self.ce_w * ce_loss +
                      self.dice_w * dice_loss +
                      self.edt_w * edt_loss)
        # 返回 tuple，方便 Trainer 打印
        return total_loss, {"ce": ce_loss.item(),
                            "dice": dice_loss.item(),
                            "edt": edt_loss.item()}


# -----------------------
# Trainer
# -----------------------
class Trainer:
    def __init__(self, model, device='cuda', lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = SegmentationLoss()
        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, ce_sum, dice_sum, edt_sum = 0, 0, 0, 0
        for batch in tqdm(dataloader, desc="Training"):
            imgs, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
            preds = self.model(imgs)
            loss,loss_dict = self.criterion(preds, masks)

            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            total_loss += loss.item()
            ce_sum += loss_dict["ce"]
            dice_sum += loss_dict["dice"]
            edt_sum += loss_dict["edt"]
        n = len(dataloader)
        return (total_loss / n,
                {"ce": ce_sum / n, "dice": dice_sum / n, "edt": edt_sum / n})

    def evaluate(self, dataloader,num_classes = 3):
        self.model.eval()
        total_loss, ce_sum, dice_sum, edt_sum = 0, 0, 0, 0
        dice_sums = {cls: 0.0 for cls in range(1, num_classes)}
        count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                imgs, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                preds = self.model(imgs)
                loss,loss_dict = self.criterion(preds, masks)

                total_loss += loss.item()
                ce_sum += loss_dict["ce"]
                dice_sum += loss_dict["dice"]
                edt_sum += loss_dict["edt"]

                pred_labels = torch.argmax(preds, dim=1)
                dice_per_class = (self.compute_dice_per_class(pred_labels, masks,num_classes = num_classes))

                for cls, d in dice_per_class.items():
                    dice_sums[cls] += d.item()
                count += 1

        n = len(dataloader)
        mean_loss = total_loss / n
        dice_avg = {cls: dice_sums[cls] / count for cls in dice_sums}
        mean_dice = np.mean(list(dice_avg.values()))
        loss_breakdown = {"ce": ce_sum / n, "dice": dice_sum / n, "edt": edt_sum / n}

        return mean_dice, mean_loss, dice_avg, loss_breakdown

    def compute_dice_per_class(self, pred, target, num_classes=3, smooth=1e-5):
        """
        分别计算每个类别的 Dice
        pred: (B,H,W) int
        target: (B,H,W) int
        """
        dices = {}
        for cls in range(1, num_classes):  # 跳过背景 (0)
            pred_fg = (pred == cls).float()
            target_fg = (target == cls).float()
            inter = (pred_fg * target_fg).sum()
            union = pred_fg.sum() + target_fg.sum()
            dices[cls] = (2 * inter + smooth) / (union + smooth)
        return dices


def create_domain_dataloaders(images, masks, domain_labels, train_domains, batch_size=8):
    """创建每个训练域的数据加载器"""
    dataloaders = {}
    for domain_id in train_domains:
        domain_indices = np.where(np.array(domain_labels) == domain_id)[0]
        domain_images = [images[i] for i in domain_indices]
        domain_masks = [masks[i] for i in domain_indices]
        domain_labels_filtered = [domain_labels[i] for i in domain_indices]
        dataset = FundusDataset(domain_images, domain_masks, domain_labels_filtered)
        dataloaders[domain_id] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloaders

def run_leave_one_domain_out(images, masks, domain_labels, num_classes=3, num_epochs=100, batch_size=8, device='cuda'):
    """运行留一域验证 (LODO)"""
    unique_domains = list(set(domain_labels))
    results = {}
    domain_pbar = tqdm(unique_domains, desc="Leave-One-Domain-Out Validation")

    for test_domain in domain_pbar:
        train_domains = [d for d in unique_domains if d != test_domain]

        # train dataloaders (多域拼接成一个大 DataLoader)
        train_dataloaders = create_domain_dataloaders(images, masks, domain_labels, train_domains, batch_size=batch_size)
        train_dataset_all = torch.utils.data.ConcatDataset([dl.dataset for dl in train_dataloaders.values()])
        train_loader = DataLoader(train_dataset_all, batch_size=batch_size, shuffle=True)

        # test dataloader
        test_indices = np.where(np.array(domain_labels) == test_domain)[0]
        test_images = [images[i] for i in test_indices]
        test_masks = [masks[i] for i in test_indices]
        test_dataset = FundusDataset(test_images, test_masks, [test_domain]*len(test_images))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # model + trainer
        model = UNet(in_channels=3, num_classes=num_classes)
        trainer = Trainer(model, device=device, lr=1e-4)

        best_dice, best_epoch = 0, -1
        for epoch in range(num_epochs):
            train_loss,train_breakdown = trainer.train_epoch(train_loader)
            mean_dice, val_loss, dice_avg,val_breakdown  = trainer.evaluate(test_loader,num_classes=num_classes)

            # 用验证集 loss 更新 scheduler
            trainer.scheduler.step(val_loss)

            lr_now = trainer.optimizer.param_groups[0]['lr']

            print(f"[Domain {test_domain}] Epoch {epoch}: "
                  f"TrainLoss={train_loss:.4f} "
                  f"(CE={train_breakdown['ce']:.4f}, Dice={train_breakdown['dice']:.4f}, EDT={train_breakdown['edt']:.4f}) | "
                  f"ValLoss={val_loss:.4f} "
                  f"(CE={val_breakdown['ce']:.4f}, Dice={val_breakdown['dice']:.4f}, EDT={val_breakdown['edt']:.4f}) | "
                  f"DiscDice={dice_avg[1]:.4f}, CupDice={dice_avg[2]:.4f}, MeanDice={mean_dice:.4f}, "
                  f"LR={lr_now:.6f}")

            if mean_dice > best_dice:
                best_dice, best_epoch = mean_dice, epoch
                torch.save(model.state_dict(), f"best_unet_edt_domain{test_domain}.pth")

        results[test_domain] = {"best_dice": best_dice, "best_epoch": best_epoch}
        domain_pbar.set_postfix({"TestDomain": test_domain, "BestDice": f"{best_dice:.4f}"})

    domain_pbar.close()

    # 打印总体结果
    print("\n========== Overall Results ==========")
    mean_dice = np.mean([results[d]["best_dice"] for d in results])
    print(f"Average Mean Dice: {mean_dice:.4f}")
    for d in results:
        print(f"Domain {d}: Best Dice={results[d]['best_dice']:.4f} (Epoch {results[d]['best_epoch']})")
    return results


def read_data(root_dir = r'D:\dev\python\Datasets\Fundus-doFE\Fundus',domains = [0,1,2,3]):
    """
    Args:
        root_dir: 根目录
        domains: 读取数据的域
    Returns:
        images,masks(np.uint8),domain_labels(int)

    """
    images = []
    masks = []
    domain_labels = []
    print('Reading data...')
    for i in domains:
        file_dir = os.path.join(root_dir, 'Domain' + str(i + 1), 'train\ROIs\image')
        image_paths = [os.path.join(file_dir, image_name) for image_name in os.listdir(file_dir)]
        test_file_dir = file_dir.replace('train', 'test')
        image_paths.extend([os.path.join(test_file_dir, image_name) for image_name in os.listdir(test_file_dir)])

        images.extend([np.array(Image.open(image_path).convert('RGB').resize((256,256),Image.LANCZOS)) for image_path in image_paths])

        mask_paths = [image_path.replace('image', 'mask') for image_path in image_paths]
        masks.extend([convert_mask(mask_path) for mask_path in mask_paths])
        domain_labels.extend([int(image_path.split(os.sep)[-5][-1]) - 1 for image_path in image_paths])
    print(f'Read {len(images)} images from {root_dir}')
    return images,masks,domain_labels

def convert_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert('L').resize((256,256),Image.NEAREST))
    converted = np.zeros_like(mask,dtype = np.uint8)

    #将视杯映射为1
    converted[mask == 128] = 1
    #将视盘映射为2
    converted[mask == 0] = 2
    return converted

# 使用示例
if __name__ == "__main__":
    # 假设你有以下数据
    # images: List[np.ndarray] - 图像列表
    # masks: List[np.ndarray] - 分割标签列表
    # domain_labels: List[int] - 域标签列表 [0,1,2,3]

    # 示例数据（实际使用时替换为你的真实数据）
    images,masks,domain_labels = read_data(root_dir = '/hy-tmp/Fundus-doFE/Fundus',domains = [0,1,2,3])
    # 运行留一域验证
    results = run_leave_one_domain_out(images, masks, domain_labels, num_classes=3)

    print("EDT眼底图像分割框架已准备就绪！")
    print("使用说明：")
    print("1. 准备你的images, masks, domain_labels数据")
    print("2. 调用run_leave_one_domain_out()函数")
    print("3. 系统将自动进行四轮留一域验证")
    print("4. 每轮训练会保存最佳模型到本地")