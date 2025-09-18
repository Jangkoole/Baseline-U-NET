import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from matplotlib import pyplot as plt
from EDT import UNet, FundusDataset, convert_mask  # 复用训练时的定义
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------- Dice计算 ----------------
def compute_dice_per_class(pred, target, num_classes=3, smooth=1e-5):
    dices = {}
    for cls in range(1, num_classes):  # 跳过背景 (0)
        pred_fg = (pred == cls).float()
        target_fg = (target == cls).float()
        inter = (pred_fg * target_fg).sum()
        union = pred_fg.sum() + target_fg.sum()
        dices[cls] = (2 * inter + smooth) / (union + smooth)
    return dices


# ---------------- 可视化函数 ----------------
def visualize_result(image, ground_truth, prediction, cup_dice, disc_dice, save_path):
    """
    可视化分割结果并保存
    """
    # 将图像转换为适合显示的格式
    if image.shape[0] == 3:  # 如果是RGB图像
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)
    else:  # 如果是灰度图像
        image = image[0]  # 取第一个通道
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 创建可视化图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 显示真实分割
    axes[1].imshow(ground_truth, cmap='jet')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # 显示预测分割
    axes[2].imshow(prediction, cmap='jet')
    axes[2].set_title(f'Prediction\nCup Dice: {cup_dice:.3f}, Disc Dice: {disc_dice:.3f}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------- 测试函数 ----------------
def test_model(model_path, images, masks, domain_labels, test_domain, num_classes=3, device=device, save_vis=True):
    print(f"\n[Testing] Loading model from {model_path}")

    # 创建保存目录
    save_dir = f"./EDT_results/test_domain{test_domain}"
    if save_vis:
        os.makedirs(save_dir, exist_ok=True)

    # 创建测试集 dataloader
    test_indices = np.where(np.array(domain_labels) == test_domain)[0]
    test_images = [images[i] for i in test_indices]
    test_masks = [masks[i] for i in test_indices]
    test_labels = [domain_labels[i] for i in test_indices]

    test_dataset = FundusDataset(test_images, test_masks, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 加载模型
    model = UNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 评估
    dice_sums = {cls: 0.0 for cls in range(1, num_classes)}
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing on Domain {test_domain}")):
            imgs, masks_gt = batch['image'].to(device), batch['mask'].to(device)
            preds = model(imgs)

            loss = F.cross_entropy(preds, masks_gt)
            total_loss += loss.item()

            pred_labels = torch.argmax(preds, dim=1)

            # 逐样本计算 Dice & 可视化
            for i in range(imgs.size(0)):
                dice_per_class = compute_dice_per_class(pred_labels[i], masks_gt[i], num_classes)
                for cls, d in dice_per_class.items():
                    dice_sums[cls] += d.item()

                if save_vis:
                    img_np = imgs[i].cpu().numpy()
                    gt_np = masks_gt[i].cpu().numpy()
                    pred_np = pred_labels[i].cpu().numpy()

                    save_path = os.path.join(save_dir, f"sample_{batch_idx * test_loader.batch_size + i}.png")
                    visualize_result(
                        img_np, gt_np, pred_np,
                        cup_dice=dice_per_class[2].item(),
                        disc_dice=dice_per_class[1].item(),
                        save_path=save_path
                    )

            count += imgs.size(0)

    # 平均指标
    dice_avg = {cls: dice_sums[cls] / count for cls in dice_sums}
    mean_dice = np.mean(list(dice_avg.values()))
    mean_loss = total_loss / len(test_loader)

    print(f"\n=== Test Results on Domain {test_domain} ===")
    print(f" Disc Dice: {dice_avg[1]:.4f}")
    print(f" Cup Dice : {dice_avg[2]:.4f}")
    print(f" Mean Dice: {mean_dice:.4f}")
    print(f" CE Loss  : {mean_loss:.4f}")

    return {
        "disc_dice": dice_avg[1],
        "cup_dice": dice_avg[2],
        "mean_dice": mean_dice,
        "loss": mean_loss
    }


# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    # 假设数据和函数复用自 EDT.py
    from EDT import read_data

    # 读取完整数据
    images, masks, domain_labels = read_data(
        root_dir=r"D:\dev\python\Datasets\Fundus-doFE\Fundus",
        domains=[3]
    )

    # 指定测试域
    test_domain = 3
    model_path = f"./EDT_results/best_unet_edt_domain{test_domain}.pth"

    results = test_model(model_path, images, masks, domain_labels, test_domain, num_classes=3, save_vis=True)
