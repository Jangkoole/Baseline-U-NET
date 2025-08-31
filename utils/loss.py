import torch
import torch.nn as nn


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=None, epsilon=1e-5):
        """
        带权重的多通道 Dice 损失

        参数:
            weights (list/tensor): 每个通道的权重，如 [1.0, 0.8] 表示第一个通道权重为1.0，第二个为0.8
            epsilon (float): 平滑系数，避免除零错误
        """
        super(WeightedDiceLoss, self).__init__()
        self.epsilon = epsilon

        # 默认权重：两个通道权重相等
        if weights is None:
            weights = [1.0, 1.0]

        # 将权重转换为张量
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, pred, target):
        """
        计算带权重的 Dice 损失

        参数:
            pred (tensor): 模型预测值，形状 [B, 2, H, W]
            target (tensor): 目标标签，形状 [B, 2, H, W]

        返回:
            loss (tensor): 加权 Dice 损失
        """
        # 确保权重在正确的设备上 (CPU/GPU)
        if self.weights.device != pred.device:
            self.weights = self.weights.to(pred.device)

        # 计算每个通道的交集、预测和、目标和
        intersection = torch.sum(pred * target, dim=(2, 3))  # [B, 2]
        pred_sum = torch.sum(pred, dim=(2, 3))  # [B, 2]
        target_sum = torch.sum(target, dim=(2, 3))  # [B, 2]

        # 计算每个通道的 Dice 系数
        dice_per_channel = (2. * intersection + self.epsilon) / (pred_sum + target_sum + self.epsilon)  # [B, 2]

        # 计算每个通道的 Dice 损失 (1 - Dice系数)
        dice_loss_per_channel = 1 - dice_per_channel  # [B, 2]

        # 应用通道权重
        weighted_loss = dice_loss_per_channel * self.weights  # [B, 2]

        # 计算每个样本的加权损失 (对通道求和)
        per_sample_loss = torch.sum(weighted_loss, dim=1)  # [B]

        # 返回批次平均损失
        return torch.mean(per_sample_loss)