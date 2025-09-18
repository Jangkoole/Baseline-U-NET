import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple
import torch.autograd as autograd
import os
from PIL import Image
from tqdm import tqdm

class FundusDataset(Dataset):
    """眼底图像数据集"""

    def __init__(self, images, masks, domain_labels, transform=None):
        self.images = images
        self.masks = masks
        self.domain_labels = domain_labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        domain = self.domain_labels[idx]

        # 确保图像是 [C, H, W] 格式
        if len(image.shape) == 3 and image.shape[2] == 3:  # H, W, C
            image = image.transpose(2, 0, 1)  # 转换为 C, H, W

        # 归一化图像到 [0, 1]
        image = image.astype(np.float32) / 255.0

        if self.transform:
            # 这里应用数据增强
            pass

        return {
            'image': torch.FloatTensor(image),
            'mask': torch.LongTensor(mask),
            'domain': domain
        }


class DoubleConv(nn.Module):
    """U-Net的双卷积块"""

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

    def forward(self, x):
        return self.double_conv(x)


class UNetEncoder(nn.Module):
    """U-Net编码器作为特征提取器Φ"""

    def __init__(self, in_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

    def forward(self, x):
        x1 = self.inc(x)  # 64, H, W
        x2 = self.down1(x1)  # 128, H/2, W/2
        x3 = self.down2(x2)  # 256, H/4, W/4
        x4 = self.down3(x3)  # 512, H/8, W/8
        x5 = self.down4(x4)  # 1024, H/16, W/16
        return [x1, x2, x3, x4, x5]


class UNetDecoder(nn.Module):
    """U-Net解码器"""

    def __init__(self, num_classes):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, features,return_intermediate=False):
        x1, x2, x3, x4, x5 = features

        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        decode_feat1 = self.conv1(x)

        x = self.up2(decode_feat1)
        x = torch.cat([x3, x], dim=1)
        decode_feat2 = self.conv2(x)

        x = self.up3(decode_feat2)
        x = torch.cat([x2, x], dim=1)
        decode_feat3 = self.conv3(x)

        x = self.up4(decode_feat3)
        x = torch.cat([x1, x], dim=1)
        decode_feat4 = self.conv4(x)

        output =  self.outc(decode_feat4)

        if return_intermediate:
            return output,[decode_feat1,decode_feat2,decode_feat3,decode_feat4]
        return output


class MultiLayerIRMSegmentationModel(nn.Module):
    """IRM分割模型"""

    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.feature_extractor = UNetEncoder(in_channels)  # 这是Φ
        self.decoder = UNetDecoder(num_classes)

        # 多层IRM分类器 - 在不同特征层面应用约束
        self.irm_classifiers = nn.ModuleDict({
            # 在解码器的不同层应用IRM约束
            'decode_stage1': nn.Conv2d(512, num_classes, 1, bias=False),
            'decode_stage2': nn.Conv2d(256, num_classes, 1, bias=False),
            'decode_stage3': nn.Conv2d(128, num_classes, 1, bias=False),
            'decode_stage4': nn.Conv2d(64, num_classes, 1, bias=False)
        })

        #最后的分类层
        self.final_classifier = nn.Conv2d(num_classes, num_classes, 1, bias=False)

        # 初始化所有分类器权重为1.0
        for classifier in self.irm_classifiers.values():
            nn.init.ones_(classifier.weight)
        nn.init.ones_(self.final_classifier.weight)

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)

        if return_features:
            decoded, intermediate_features = self.decoder(features, return_intermediate=True)
            return decoded, features, intermediate_features

        decoded = self.decoder(features)
        # 最终预测
        output = self.final_classifier(decoded)
        return output


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 输入是logits，需要先进行sigmoid或softmax
        inputs = F.softmax(inputs, dim=1)

        # 将目标转换为one-hot编码
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        # 计算交集和并集
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 返回1 - Dice系数作为损失
        return 1 - dice.mean()

class MultiLayerIRMLoss(nn.Module):
    """IRM损失函数"""

    def __init__(self, num_classes=2, irm_lambda=1.0,dice_weight = 0.5,ce_weight = 0.5,stage_weights = None):
        super().__init__()
        self.num_classes = num_classes
        self.irm_lambda = irm_lambda
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # 各解码阶段的权重 - 后期阶段权重更大
        self.stage_weights = stage_weights or {
            'decode_stage1': 0.2,  # 最粗糙的融合特征
            'decode_stage2': 0.4,  # 中等分辨率
            'decode_stage3': 0.7,  # 高分辨率
            'decode_stage4': 1.0,  # 最终特征，权重最大
        }

    def compute_decode_stage_penalty(self, model, images_by_domain, targets_by_domain):
        """计算多层IRM惩罚"""
        total_penalty = 0

        for domain_id in images_by_domain:
            images = images_by_domain[domain_id]
            targets = targets_by_domain[domain_id]

            # 获取解码器的中间特征
            _, _, intermediate_features = model(images, return_features=True)

            domain_penalty = 0

            # 对每个解码阶段计算IRM惩罚
            for stage_idx, (stage_name, stage_weight) in enumerate(self.stage_weights.items()):
                decode_features = intermediate_features[stage_idx]  # decode_feat1, decode_feat2, etc.
                classifier = model.irm_classifiers[stage_name]

                # 下采样targets到对应尺寸
                target_size = decode_features.shape[2:]
                if target_size != targets.shape[1:]:
                    downsampled_targets = F.interpolate(
                        targets.unsqueeze(1).float(),
                        size=target_size,
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    downsampled_targets = targets

                # 计算该阶段的IRM惩罚
                penalty = self.compute_single_stage_penalty(
                    decode_features, downsampled_targets, classifier
                )
                domain_penalty += stage_weight * penalty

            total_penalty += domain_penalty

        return total_penalty


    def compute_single_stage_penalty(self, logits, targets, classifier):
        """计算单个解码器阶段IRM梯度惩罚"""
        # 创建虚拟的w=1.0分类器权重
        dummy_w = torch.ones_like(classifier.weight,requires_grad=True)

        # 使用虚拟权重进行预测
        dummy_logits = F.conv2d(logits, dummy_w)
        #dummy_loss = self.seg_loss(dummy_logits, targets)
        dummy_loss = (self.dice_weight * self.dice_loss(dummy_logits, targets)
                      + self.ce_weight * self.ce_loss(dummy_logits, targets))

        # 计算关于虚拟权重的梯度
        grad = autograd.grad(
            outputs=dummy_loss,
            inputs=dummy_w,
            create_graph=True,
            retain_graph=True
        )[0]

        return grad.pow(2).sum()
    def forward(self,model, images_by_domain, targets_by_domain):
        """
        predictions_by_domain: {domain_id: logits}
        targets_by_domain: {domain_id: targets}
        classifier_weight: 分类器的权重参数
        """
        total_seg_loss = 0

        for domain_id in images_by_domain:
            images = images_by_domain[domain_id]
            targets = targets_by_domain[domain_id]

            outputs = model(images)
            # 分割损失
            #seg_loss = self.seg_loss(logits, targets)
            seg_loss = (self.dice_weight * self.dice_loss(outputs, targets)
                        + self.ce_weight * self.ce_loss(outputs, targets))
            total_seg_loss += seg_loss

        #计算多层IRM惩罚
        irm_penalty = self.compute_decode_stage_penalty(
            model, images_by_domain, targets_by_domain
        )

        # 总损失
        total_loss = total_seg_loss + self.irm_lambda * irm_penalty

        return {
            'total_loss': total_loss,
            'seg_loss': total_seg_loss,
            'irm_penalty': irm_penalty
        }


class MultiLayerIRMTrainer:
    """IRM训练器"""

    def __init__(self, model, train_domains=[0, 1, 2], test_domain=3,
                 irm_lambda=1.0, lr=1e-4, device='cuda',stage_weights = None):
        self.model = model.to(device)
        self.train_domains = train_domains
        self.test_domain = test_domain
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = MultiLayerIRMLoss(irm_lambda=irm_lambda,stage_weights = stage_weights)

        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )

    def train_epoch(self, dataloaders: Dict[int, DataLoader]):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'seg': 0, 'irm': 0}

        # 创建域数据迭代器
        domain_iters = {d: iter(dataloaders[d]) for d in self.train_domains}

        # 计算最小batch数量
        min_batches = min(len(dataloaders[d]) for d in self.train_domains)

        pbar = tqdm(total=min_batches,desc = f'Training Epoch')

        for batch_idx in range(min_batches):
            images_by_domain = {}
            targets_by_domain = {}

            # 从每个训练域获取一个batch
            for domain_id in self.train_domains:
                try:
                    batch = next(domain_iters[domain_id])
                except StopIteration:
                    domain_iters[domain_id] = iter(dataloaders[domain_id])
                    batch = next(domain_iters[domain_id])

                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                images_by_domain[domain_id] = images
                targets_by_domain[domain_id] = masks

            # 计算多层IRM损失
            loss_dict = self.criterion(
                self.model,
                images_by_domain,
                targets_by_domain,
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            self.optimizer.step()

            # 记录损失
            epoch_losses['total'] += loss_dict['total_loss'].item()
            epoch_losses['seg'] += loss_dict['seg_loss'].item()
            epoch_losses['irm'] += loss_dict['irm_penalty'].item()

            # 更新进度条，显示各阶段损失信息
            postfix_dict = {
                'Total': f"{loss_dict['total_loss'].item():.4f}",
                'Seg': f"{loss_dict['seg_loss'].item():.4f}",
                'IRM': f"{loss_dict['irm_penalty'].item():.4f}"
            }

            pbar.set_postfix(postfix_dict)
            pbar.update(1)

        pbar.close()
        # 返回平均损失
        for key in epoch_losses:
            epoch_losses[key] /= min_batches

        return epoch_losses

    def evaluate(self, test_dataloader: DataLoader):
        """在测试域上评估"""
        self.model.eval()
        total_loss = 0
        dice_scores = {'cup':[],'disc':[],'mean':[]}

        # 使用tqdm进度条
        pbar = tqdm(test_dataloader, desc=f"Evaluating on Domain {self.test_domain}")

        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)

                # 计算多类别Dice分数
                batch_dice = self.compute_multiclass_dice(predictions, masks)
                dice_scores['cup'].extend(batch_dice['cup'])
                dice_scores['disc'].extend(batch_dice['disc'])
                dice_scores['mean'].extend(batch_dice['mean'])

                # 计算损失
                loss = F.cross_entropy(outputs, masks)
                total_loss += loss.item()

                # 更新进度条
                pbar.set_postfix({
                    'Cup Dice': f"{np.mean(dice_scores['cup']):.4f}",
                    'Disc Dice': f"{np.mean(dice_scores['disc']):.4f}",
                    'Mean Dice': f"{np.mean(dice_scores['mean']):.4f}"
                })
        pbar.close()
        # 返回平均指标
        return {
            'dice_cup': np.mean(dice_scores['cup']),
            'dice_disc': np.mean(dice_scores['disc']),
            'dice_mean': np.mean(dice_scores['mean']),
            'loss': total_loss / len(test_dataloader)
        }

    def compute_multiclass_dice(self, pred, target):
        """计算多类别Dice分数 - 修改原来的compute_dice_score"""
        batch_size = pred.shape[0]
        dice_results = {'cup': [], 'disc': [], 'mean': []}

        for i in range(batch_size):
            # 视杯 Dice (类别1)
            cup_dice = self.compute_single_class_dice(pred[i], target[i], class_id=1)
            dice_results['cup'].append(cup_dice)

            # 视盘 Dice (类别2)
            disc_dice = self.compute_single_class_dice(pred[i], target[i], class_id=2)
            dice_results['disc'].append(disc_dice)

            # 平均Dice
            mean_dice = (cup_dice + disc_dice) / 2
            dice_results['mean'].append(mean_dice)

        return dice_results

    def compute_single_class_dice(self, pred, target, class_id):
        """计算单个类别的Dice分数"""
        smooth = 1e-5
        pred_class = (pred == class_id).float()
        target_class = (target == class_id).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()


    def train(self, dataloaders: Dict[int, DataLoader], test_dataloader: DataLoader,
              num_epochs=100):
        """完整训练流程"""
        best_dice = 0

        # 使用tqdm进度条显示epoch进度
        epoch_pbar = tqdm(range(num_epochs), desc="Overall Training Progress")

        for epoch in epoch_pbar:
            # 设置当前epoch
            self.current_epoch = epoch

            # 训练
            train_losses = self.train_epoch(dataloaders)

            # 更新epoch进度条
            epoch_pbar.set_postfix({
                'Total Loss': f"{train_losses['total']:.4f}",
                'Seg Loss': f"{train_losses['seg']:.4f}",
                'IRM Loss': f"{train_losses['irm']:.4f}"
            })

            # 验证
            if epoch % 5 == 0:  # 每5个epoch验证一次
                eval_results = self.evaluate(test_dataloader)

                print(f'Train Losses - Total: {train_losses["total"]:.4f}, '
                      f'Seg: {train_losses["seg"]:.4f}, IRM: {train_losses["irm"]:.4f}')
                print(f'Test Results - Cup Dice: {eval_results["dice_cup"]:.4f}, '
                      f'Disc Dice: {eval_results["dice_disc"]:.4f}, '
                      f'Mean Dice: {eval_results["dice_mean"]:.4f}, '
                      f'Loss: {eval_results["loss"]:.4f}')

                # 学习率调度
                self.scheduler.step(eval_results['loss'])

                # 保存最佳模型 - 使用平均Dice作为指标
                if eval_results['dice_mean'] > best_dice:
                    best_dice = eval_results['dice_mean']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'dice_cup': eval_results['dice_cup'],
                        'dice_disc': eval_results['dice_disc'],
                        'dice_mean': eval_results['dice_mean'],
                        'epoch': epoch
                    }, f'best_irm_model_test_domain_{self.test_domain}.pth')
                    print(f'New best model saved! Mean Dice: {best_dice:.4f}')
        epoch_pbar.close()

class AdaptiveIRMTrainer(MultiLayerIRMTrainer):
    """自适应IRM训练器，动态调整λ"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_irm_lambda = kwargs.get('irm_lambda', 1.0)
        self.warmup_epochs = 20  # 前20个epoch较小的IRM权重

        # 保存初始阶段权重，用于动态调整
        self.initial_stage_weights = self.criterion.stage_weights.copy()

    def get_irm_lambda(self, epoch):
        """动态调整IRM权重"""
        if epoch < self.warmup_epochs:
            # 线性增长
            return self.initial_irm_lambda * (epoch / self.warmup_epochs)
        else:
            return self.initial_irm_lambda

    def get_adaptive_stage_weights(self, epoch):
        """动态调整各阶段权重"""
        # 前期主要约束后面的阶段，后期逐步加强前面阶段的约束
        if epoch < 20:
            # 前20个epoch，主要约束最后两个阶段
            return {
                'decode_stage1': 0.05,  # 很小的权重
                'decode_stage2': 0.1,
                'decode_stage3': 0.4,
                'decode_stage4': 1.0,  # 最高权重
            }
        elif epoch < 50:
            # 20-50 epoch，逐步加强中间阶段约束
            return {
                'decode_stage1': 0.1,
                'decode_stage2': 0.2,
                'decode_stage3': 0.6,
                'decode_stage4': 1.0,
            }
        else:
            # 50+ epoch，使用原始权重或更平衡的权重
            return self.initial_stage_weights

    def train_epoch(self, dataloaders):
        # 动态调整IRM权重
        current_lambda = self.get_irm_lambda(self.current_epoch)
        self.criterion.irm_lambda = current_lambda

        # 动态调整阶段权重
        current_stage_weights = self.get_adaptive_stage_weights(self.current_epoch)
        self.criterion.stage_weights = current_stage_weights

        return super().train_epoch(dataloaders)

    def train(self, dataloaders, test_dataloader, num_epochs=100):
        best_dice = 0

        # 使用tqdm进度条显示epoch进度
        epoch_pbar = tqdm(range(num_epochs), desc="Overall Training Progress")

        for epoch in epoch_pbar:
            self.current_epoch = epoch
            print(f'\nEpoch {epoch + 1}/{num_epochs}')

            # 动态调整IRM权重和阶段权重
            current_lambda = self.get_irm_lambda(epoch)
            current_stage_weights = self.get_adaptive_stage_weights(epoch)

            self.criterion.irm_lambda = current_lambda
            self.criterion.stage_weights = current_stage_weights

            print(f"Current IRM lambda: {current_lambda:.4f}")
            print(f"Current stage weights: {current_stage_weights}")


            # 训练
            train_losses = self.train_epoch(dataloaders)

            # 更新epoch进度条
            epoch_pbar.set_postfix({
                'Total Loss': f"{train_losses['total']:.4f}",
                'Seg Loss': f"{train_losses['seg']:.4f}",
                'IRM Loss': f"{train_losses['irm']:.4f}",
                'IRM Lambda': f"{current_lambda:.4f}",
                'S1-W': f"{current_stage_weights['decode_stage1']:.2f}",
                'S4-W': f"{current_stage_weights['decode_stage4']:.2f}"
            })

            # 验证
            if epoch % 5 == 0:  # 每5个epoch验证一次
                eval_results = self.evaluate(test_dataloader)

                print(f'Train Losses - Total: {train_losses["total"]:.4f}, '
                      f'Seg: {train_losses["seg"]:.4f}, IRM: {train_losses["irm"]:.4f}')
                print(f'Test Results - Cup Dice: {eval_results["dice_cup"]:.4f}, '
                      f'Disc Dice: {eval_results["dice_disc"]:.4f}, '
                      f'Mean Dice: {eval_results["dice_mean"]:.4f}, '
                      f'Loss: {eval_results["loss"]:.4f}')

                # 学习率调度
                self.scheduler.step(eval_results['loss'])

                # 保存最佳模型 - 使用平均Dice作为指标
                if eval_results['dice_mean'] > best_dice:
                    best_dice = eval_results['dice_mean']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'dice_cup': eval_results['dice_cup'],
                        'dice_disc': eval_results['dice_disc'],
                        'dice_mean': eval_results['dice_mean'],
                        'epoch': epoch,
                        'irm_lambda': current_lambda,
                        'stage_weights': current_stage_weights
                    }, f'best_irm_model_test_domain_{self.test_domain}.pth')
                    print(f'New best model saved! Mean Dice: {best_dice:.4f}')
        epoch_pbar.close()

def create_domain_dataloaders(images, masks, domain_labels, train_domains, batch_size=4):
    """创建每个域的数据加载器"""
    dataloaders = {}

    for domain_id in train_domains:
        # 筛选属于当前域的数据
        domain_indices = np.where(np.array(domain_labels) == domain_id)[0]
        domain_images = [images[i] for i in domain_indices]
        domain_masks = [masks[i] for i in domain_indices]
        domain_labels_filtered = [domain_labels[i] for i in domain_indices]

        # 创建数据集和数据加载器
        dataset = FundusDataset(domain_images, domain_masks, domain_labels_filtered)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders[domain_id] = dataloader

    return dataloaders


def run_leave_one_domain_out(images, masks, domain_labels, num_classes=2):
    """运行留一域验证"""
    unique_domains = list(set(domain_labels))
    results = {}

    # 使用tqdm进度条显示域验证进度
    domain_pbar = tqdm(unique_domains, desc="Leave-One-Domain-Out Validation")

    for test_domain in domain_pbar:
        # print(f"\n{'=' * 50}")
        # print(f"Testing on Domain {test_domain}")
        # print(f"{'=' * 50}")
        domain_pbar.set_description(f"Testing on Domain {test_domain}")

        # 确定训练域
        train_domains = [d for d in unique_domains if d != test_domain]

        # 创建数据加载器
        train_dataloaders = create_domain_dataloaders(
            images, masks, domain_labels, train_domains
        )

        # 创建测试数据加载器
        test_indices = np.where(np.array(domain_labels) == test_domain)[0]
        test_images = [images[i] for i in test_indices]
        test_masks = [masks[i] for i in test_indices]
        test_domain_labels = [domain_labels[i] for i in test_indices]

        test_dataset = FundusDataset(test_images, test_masks, test_domain_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        # 创建模型和训练器
        model = MultiLayerIRMSegmentationModel(num_classes=num_classes)

        # 设置层权重 - 可以根据实验调整
        stage_weights = {
            'decode_stage1': 0.1,  # 低级特征约束较轻
            'decode_stage2': 0.2,  # 中级特征约束中等
            'decode_stage3': 0.5,  # 高级特征约束较重
            'decode_stage4': 1.0  # 最终输出约束最重
        }

        trainer = AdaptiveIRMTrainer(
            model=model,
            train_domains=train_domains,
            test_domain=test_domain,
            irm_lambda=1.0,  # 可调整
            lr=1e-4,
            stage_weights = stage_weights
        )

        # 训练
        trainer.train(train_dataloaders, test_dataloader, num_epochs=200)

        # 最终评估
        final_results = trainer.evaluate(test_dataloader)
        results[test_domain] = final_results

        # 更新进度条
        domain_pbar.set_postfix({
            'Cup Dice': f"{final_results['dice_cup']:.4f}",
            'Disc Dice': f"{final_results['dice_disc']:.4f}",
            'Mean Dice': f"{final_results['dice_mean']:.4f}"
        })

        print(f"Domain {test_domain} Final Results:")
        print(f"  Cup Dice: {final_results['dice_cup']:.4f}")
        print(f"  Disc Dice: {final_results['dice_disc']:.4f}")
        print(f"  Mean Dice: {final_results['dice_mean']:.4f}")
    domain_pbar.close()

    # 打印总体结果
    print(f"\n{'=' * 50}")
    print("Overall Results:")
    print(f"{'=' * 50}")
    avg_cup_dice = np.mean([results[d]['dice_cup'] for d in results])
    avg_disc_dice = np.mean([results[d]['dice_disc'] for d in results])
    avg_mean_dice = np.mean([results[d]['dice_mean'] for d in results])

    print(f"Average Cup Dice across all test domains: {avg_cup_dice:.4f}")
    print(f"Average Disc Dice across all test domains: {avg_disc_dice:.4f}")
    print(f"Average Mean Dice across all test domains: {avg_mean_dice:.4f}")

    for domain in results:
        r = results[domain]
        print(f"Domain {domain}: Cup={r['dice_cup']:.4f}, Disc={r['dice_disc']:.4f}, Mean={r['dice_mean']:.4f}")

    return results

def read_data(root_dir = r'D:\dev\python\Datasets\Fundus-doFE\Fundus',domains = [0,1,2,3]):
    """
    Args:
        root_dir: 根目录
        domains: 要读取的域

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

    print("IRM眼底图像分割框架已准备就绪！")
    print("使用说明：")
    print("1. 准备你的images, masks, domain_labels数据")
    print("2. 调用run_leave_one_domain_out()函数")
    print("3. 系统将自动进行四轮留一域验证")
    print("4. 每轮训练会保存最佳模型到本地")