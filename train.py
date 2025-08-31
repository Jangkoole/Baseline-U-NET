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
from networks.Unet import UNet
from networks.Unet_styleTransform import UNet_styleTransform

local_path = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument('--datasetTrain', nargs='+', type=int, default=1, help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=4, help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training the model')
    parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
    parser.add_argument('--max-epoch', type=int, default=120, help='max epoch')
    parser.add_argument('--stop-epoch', type=int, default=80, help='stop epoch')

    parser.add_argument('--mixstyle_layers', nargs='+', type=str, default=['layer0', 'layer1'], help='layer0-4, layers that use mixStyle')
    parser.add_argument('--random_type', type=str, default='TriD', help='TriD/MixStyle/EFDMix')
    parser.add_argument('--random_prob', type=float, default=0.5,help='probability of using random mixStyle')

    parser.add_argument('--interval-validate', type=int, default=10, help='interval epoch number to valid the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate',)
    parser.add_argument('--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr')
    parser.add_argument('--lam', type=float, default=0.9, help='momentum of memory update',)
    parser.add_argument('--data-dir', default='./Fundus-doFE/Fundus/', help='data root path')
    parser.add_argument('--pretrained-model', default='../../../models/pytorch/fcn16s_from_caffe.pth', help='pretrained model of FCN16s',)
    args = parser.parse_args()

    now = datetime.now()
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
    #model = UNet(n_channels = 3, n_classes = 2, bilinear=False).cuda()
    model = UNet_styleTransform(n_channels=3,n_classes=2,bilinear=False,
                                mixStyle_layers=args.mixstyle_layers,
                                random_type=args.random_type,
                                p=args.random_prob).cuda()
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
