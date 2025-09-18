from PIL import Image
import numpy as np
import torch
import argparse
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
import tqdm

#1 插值方式测试
# label1 = Image.open(r"D:\dev\python\Datasets\Fundus-doFE\Fundus\Domain4\train\ROIs\mask\V0001.png")
# #label1.show(title = 'original image')
# label2 = label1.resize((256, 256),Image.BILINEAR)
# #label2.show(title = 'resize by Image.BILINEAR')
# label3 = label1.resize((256, 256),Image.NEAREST)
# #label3.show(title = 'resize by Image.NEAREST')
# label2_npy = np.array(label2)
# label3_npy = np.array(label3)
# print(np.unique(label2_npy, return_counts=True))
# print(label2_npy.dtype)
# print(np.unique(label3_npy, return_counts=True))
# print(label3_npy.dtype)
#
# label2_tensor = torch.from_numpy(label2_npy.astype(np.int64)) #转化为int64类型的标签，确保后续交叉熵损失（nn.CrossEntropyLoss()计算不出错）
# print(label2_tensor.dtype)
# label3_tensor = torch.from_numpy(label3_npy.astype(np.int64))
# print(label3_tensor.dtype)


# parser = argparse.ArgumentParser(
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# )
#
# parser.add_argument('--data-dir', default='./Fundus-doFE/Fundus/', help='data root path')
# parser.add_argument('--dataset-train',nargs='+', type=int, default=1,
#                     help='train folder id')
# parser.add_argument('--dataset-test',nargs='+',type=int,default=4,
#                     help='test folder id contain images ROIs to test one of [1,2,3,4]')
# parser.add_argument('--batch-size',type=int,default=8,help='batch size for training the model')
# args = parser.parse_args()

#2 argparse返回数据类型测试
#print(args)
#print(type(args.dataset_train[0]))
#print(type(args.dataset_test[0]))
#运行命令：python code_test.py --dataset-train 2 3 4 --dataset-test 1，发现都是int数据类型的列表


#3 Dataloader返回训练和测试和验证数据的小批次数据的形式
# composed_transforms_tr = transforms.Compose([
#     tr.RandomScaleCrop(256),
#     tr.Normalize_tf(),
#     tr.ToTensor()
# ])
#
# composed_transforms_ts = transforms.Compose([
#     tr.RandomCrop(256),
#     tr.Normalize_tf(),
#     tr.ToTensor()
# ])
# domain = DL.FundusSegmentation(base_dir=args.data_dir, phase='train', splitid=args.dataset_train,
#                                                          transform=composed_transforms_tr)
# train_loader = DataLoader(domain, batch_size=args.batch_size, shuffle=True)
#
# domain_val = DL.FundusSegmentation(base_dir=args.data_dir, phase='test', splitid=args.dataset_test,
#                                        transform=composed_transforms_ts)
# val_loader = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False)
#
#
# for batch_idx, sample in enumerate(val_loader):
#     print(type(sample))
#     print(sample.keys())
#     print('shape valid image:',sample['image'].shape)
#     print('shape valid label:',sample['label'].shape)
#     print('type valid image:', type(sample['image']))
#     print('type valid label:', type(sample['label']))
#     print('data type valid image:', sample['image'].dtype)
#     print('data type valid label:', sample['label'].dtype)
# print()
# for batch_idx, sample in enumerate(train_loader):
#     print(type(sample))
#     print(type(sample[0]))
#     print('shape train image:',sample[0]['image'].shape)
#     print('shape train label:',sample[0]['label'].shape)
#     print('type train image:', type(sample[0]['image']))
#     print('type train label:', type(sample[0]['label']))
#     print('data type train image:',sample[0]['image'].dtype)
#     print('data type train label:', sample[0]['label'].dtype)
#     print("shape train image name: ",len(sample[0]['img_name']))
#运行命令：python code_test.py --data-dir D:\dev\python\Datasets\Fundus-doFE\Fundus --dataset-train 2 3 4 --dataset-test 1
#结论：返回的格式是sample的格式，堆叠形式是按照图片、标签、图片名等沿批次堆叠;
#也可以通过自定义collate_fn来定义批次堆叠格式
#标签值经过tr.Normalize_tf()变成两通道的one-hot编码格式(2通道),数据格式变成Tensor,数据类型是image(torch.float32),label(torch.float32)
#之所以把label的数据类型保存为torch.float32，可能是因为后面不用计算交叉熵损失（nn.CrossEntropyLoss），只需要计算Dice等

import os
from PIL import Image


def read_data(root_dir=r'D:\dev\python\Datasets\Fundus-doFE\Fundus', domain_num=4):
    images = []
    masks = []
    domain_labels = []
    print('Reading data...')
    last = 0
    for i in range(domain_num):
        file_dir = os.path.join(root_dir, 'Domain' + str(i + 1), 'train\ROIs\image')
        image_paths = [os.path.join(file_dir, image_name) for image_name in os.listdir(file_dir)]
        test_file_dir = file_dir.replace('train', 'test')
        image_paths.extend([os.path.join(test_file_dir, image_name) for image_name in os.listdir(test_file_dir)])
        # for path in image_paths:
        #     print(path)

        images.extend([np.array(Image.open(image_path)) for image_path in image_paths])

        mask_paths = [image_path.replace('image', 'mask') for image_path in image_paths]
        masks.extend([convert_mask(mask_path) for mask_path in mask_paths])
        domain_labels.extend([int(image_path.split(os.sep)[-5][-1]) - 1 for image_path in image_paths])
        #print(domain_labels)
        print(f'Read {(len(image_paths) - last)} images from Domain {i}')
    print(f'Read {len(images)} images from {root_dir}')
    return images, masks, domain_labels


def convert_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    converted = np.zeros_like(mask, dtype=np.uint8)

    # 将视杯映射为1
    converted[mask == 128] = 1
    # 将视盘映射为2
    converted[mask == 0] = 2
    return converted
if __name__ == '__main__':
    images,masks,domain_labels = read_data()
    print(len(masks),masks[0].dtype)