import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import cv2
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from IRM3 import MultiLayerIRMSegmentationModel,FundusDataset,read_data
#from IRM_baseline import MultiLayerIRMSegmentationModel,FundusDataset,read_data

def calculate_dice_coefficient(pred, target, class_idx):
    """
    计算特定类别的Dice系数

    参数:
        pred: 预测的分割图 (H, W)
        target: 真实的分割图 (H, W)
        class_idx: 要计算的类别索引 (0:背景, 1:视杯, 2:视盘)

    返回:
        dice: Dice系数
    """
    # 二值化预测和真实标签
    pred_binary = (pred == class_idx).astype(np.int32)
    target_binary = (target == class_idx).astype(np.int32)

    # 计算交集和并集
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)

    # 避免除以零
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    # 计算Dice系数
    dice = 2.0 * intersection / union
    return dice


def test_model_on_domain(model, test_loader, device, save_dir=None):
    """
    在测试域上评估模型性能

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备 (cpu或cuda)
        save_dir: 可选，保存可视化结果的目录

    返回:
        metrics: 包含性能指标的字典
    """
    model.eval()
    cup_dices = []
    disc_dices = []

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # 前向传播
            outputs = model(images)

            # 获取预测结果
            if isinstance(outputs, dict):  # 如果模型返回字典
                preds = outputs['out']
            else:  # 如果模型直接返回张量
                preds = outputs

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            # 计算每个样本的Dice系数
            for j in range(len(preds)):
                pred = preds[j]
                mask = masks[j]

                # 计算视杯Dice系数 (类别1)
                cup_dice = calculate_dice_coefficient(pred, mask, 1)
                cup_dices.append(cup_dice)

                # 计算视盘Dice系数 (类别2)
                disc_dice = calculate_dice_coefficient(pred, mask, 2)
                disc_dices.append(disc_dice)

                # 可选：保存可视化结果
                if save_dir and i < 10:  # 只保存前10个样本的可视化
                    visualize_result(images[j].cpu().numpy(), mask, pred,
                                     cup_dice, disc_dice,
                                     os.path.join(save_dir, f"sample_{i}_{j}.png"))

    # 计算平均Dice系数
    avg_cup_dice = np.mean(cup_dices)
    avg_disc_dice = np.mean(disc_dices)
    avg_dice = (avg_cup_dice + avg_disc_dice) / 2

    metrics = {
        'cup_dice': avg_cup_dice,
        'disc_dice': avg_disc_dice,
        'avg_dice': avg_dice,
        'cup_dices': cup_dices,
        'disc_dices': disc_dices
    }

    return metrics


def visualize_result(image, ground_truth, prediction, cup_dice, disc_dice, save_path):
    """
    可视化分割结果并保存

    参数:
        image: 原始图像 (C, H, W)
        ground_truth: 真实分割图 (H, W)
        prediction: 预测分割图 (H, W)
        cup_dice: 视杯Dice系数
        disc_dice: 视盘Dice系数
        save_path: 保存路径
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


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型参数 (需要与训练时一致)
    model_params = {
        'in_channels': 3,
        'num_classes': 3
    }

    # 创建模型实例
    model = MultiLayerIRMSegmentationModel(**model_params).to(device)

    # 加载训练好的权重
    checkpoint_path = 'E:/best_irm_model_test_domain_3.pth'  # 替换为您的检查点路径
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载模型权重从: {checkpoint_path}")

    # 准备测试数据
    images, masks, domain_labels = read_data(root_dir=r'D:\dev\python\Datasets\Fundus-doFE\Fundus', domains = [3])
    test_dataset = FundusDataset(
        images=images,
        masks=masks,
        domain_labels=domain_labels
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # 测试模型
    save_dir = './test_Domain4/'  # 保存可视化结果的目录
    metrics = test_model_on_domain(model, test_loader, device, save_dir)

    # 打印结果
    print("\n测试结果:")
    print(f"平均视杯Dice系数: {metrics['cup_dice']:.4f}")
    print(f"平均视盘Dice系数: {metrics['disc_dice']:.4f}")
    print(f"平均Dice系数: {metrics['avg_dice']:.4f}")

    # 可选：保存详细结果
    results_path = os.path.join(save_dir, 'test_metrics.npy')
    np.save(results_path, metrics)
    print(f"详细结果已保存至: {results_path}")


if __name__ == '__main__':
    main()