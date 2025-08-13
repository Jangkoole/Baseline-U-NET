from datetime import datetime
import os
import os.path as osp

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer

from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
#from networks.deeplabv3 import *
from networks.Unet import UNet
from tqdm import tqdm

local_path = osp.dirname(osp.abspath(__file__))


def centroids_init(model, data_dir, datasetTrain, composed_transforms):
    """
        初始化多源域特征中心（质心）

        参数:
            model: 待初始化的模型
            data_dir: 数据集根目录
            datasetTrain: 训练数据集索引列表
            composed_transforms: 数据预处理组合

        返回:
            空间压缩后的特征中心张量 [3, 304, 64, 64]
    """
    #初始化全零质心张量：3源域 * 304个通道 * 64 * 64特征图;原因是提取的特征图的形状是B*304*64*64，B设置为1
    centroids = torch.zeros(3, 304, 64, 64).cuda() # 3 means the number of source domains
    model.eval() #模型设置为评估模式（禁用dropout、BatchNorm等训练专用层）

    # Calculate initial centroids only on training data.
    with torch.set_grad_enabled(False): #停止计算梯度以节省内存
        count = 0
        # tranverse each training source domain
        for index in datasetTrain:
            domain = DL.FundusSegmentation(base_dir=data_dir, phase='train', splitid=[index],
                                           transform=composed_transforms)
            dataloader = DataLoader(domain, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

            for id, sample in tqdm(enumerate(dataloader)):
                sample=sample[0] #解包样本：因为batch_size=1,由B*C*W*H变成C*W*H
                inputs = sample['image'].cuda()
                features = model(inputs, extract_feature=True) #换模型会报错；返回形状[1,304,64,64]

                # Calculate the sum features from the same domain;累加到当前域上用于后面求平均值
                centroids[count:count+1] += features

            # Average summed features with class count; 后半部分相当于做了个显式广播机制，事实上无需手动扩展维度
            centroids[count] /= torch.tensor(len(dataloader)).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
            count += 1
    # Calculate the mean features for each domain
    # 空间维度压缩：计算每个空间位置的平均特征
    # 第一步：沿宽度维度(W)平均 → [3, 304, 64, 1]
    # 第二步：沿高度维度(H)平均 → [3, 304, 1, 1]
    ave = torch.mean(torch.mean(centroids, 3, True), 2, True) # size [3, 304]

    # 将压缩后的特征向量扩展回原始特征图尺寸
    # expand_as: 复制特征向量到每个空间位置
    # contiguous: 确保内存连续布局（加速后续计算）
    return ave.expand_as(centroids).contiguous()  # size [3, 304, 64, 64]

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')

    parser.add_argument('--datasetTrain', nargs='+', type=int, default=1, help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=1, help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training the model')
    parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
    parser.add_argument('--max-epoch', type=int, default=120, help='max epoch')
    parser.add_argument('--stop-epoch', type=int, default=80, help='stop epoch')
    parser.add_argument('--interval-validate', type=int, default=10, help='interval epoch number to valid the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate',)
    parser.add_argument('--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr')
    parser.add_argument('--lam', type=float, default=0.9, help='momentum of memory update',)
    parser.add_argument('--data-dir', default='./Fundus-doFE/Fundus/', help='data root path')
    parser.add_argument('--pretrained-model', default='../../../models/pytorch/fcn16s_from_caffe.pth', help='pretrained model of FCN16s',)
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+',)
    args = parser.parse_args()

    now = datetime.now()
    #args.out = osp.join(local_path, 'logs', 'test'+str(args.datasetTest[0]), 'lam'+str(args.lam), now.strftime('%Y%m%d_%H%M%S.%f'))
    args.out = osp.join(local_path, 'logs', 'test' + str(args.datasetTest[0]),
                        now.strftime('%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    torch.cuda.manual_seed(1337)

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(256),
        # tr.RandomCrop(512),
        # tr.RandomRotate(),
        # tr.RandomFlip(),
        # tr.elastic_transform(),
        # tr.add_salt_pepper_noise(),
        # tr.adjust_light(),
        # tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomCrop(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = DL.FundusSegmentation(base_dir=args.data_dir, phase='train', splitid=args.datasetTrain,
                                                         transform=composed_transforms_tr)
    train_loader = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                       transform=composed_transforms_ts)
    val_loader = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. model
    #model = DeepLab(num_classes=2, num_domain=3, backbone='mobilenet', output_stride=args.out_stride, lam=args.lam).cuda()
    model = UNet(n_channels = 3, n_classes = 2, bilinear=False).cuda()
    print('parameter numer:', sum([p.numel() for p in model.parameters()]))

    # load weights
    # 这么多步骤的原因：预训练模型可能与真实模型结构有所不同（增加/修改了层），下面几步都是为了能正确加载模型参数
    if args.resume:
        # 加载检查点文件（包含模型权重、当前训练轮次、优化器状态、损失函数值等），通常是一个.pt文件
        checkpoint = torch.load(args.resume)
        #获取预训练模型的状态字典（权重字典）
        pretrained_dict = checkpoint['model_state_dict']
        #获取当前模型的状态字典
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys 过滤掉不需要的键（当前模型不存在的键）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        print('Before ', model.centroids.data) #centroids表示每个源域的特征中心（也成为原型或质心）
        model.centroids.data = centroids_init(model, args.data_dir, args.datasetTrain, composed_transforms_ts)
        print('Before ', model.centroids.data)
        # model.freeze_para()

    start_epoch = 0
    start_iteration = 0

    # 3. optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99)
    )

    trainer = Trainer.Trainer(
        cuda=cuda,
        model=model,
        lr=args.lr,
        lr_decrease_rate=args.lr_decrease_rate,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()
