import os.path as osp
import numpy as np
import os
import cv2
from skimage import morphology
import scipy
from PIL import Image
from matplotlib.pyplot import imsave
# from keras.preprocessing import image
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from skimage import measure, draw
import torch
from skimage.morphology import disk, erosion, dilation, opening, closing, white_tophat

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def get_largest_fillhole(binary):
    """
    处理二值掩码：保留最大连通区域并填充孔洞
        参数:
            binary (numpy.ndarray): 输入的二值掩码图像（值为0和1）
        返回:
            numpy.ndarray: 处理后的二值掩码（仅含最大区域且无孔洞）
    """
    label_image = label(binary) #对二值图像进行连通区域标记，为每个独立区域分配唯一ID（背景为0），有效ID从1开始
    regions = regionprops(label_image) #获取所有连通区域的属性列表（面积、质心、边界框等）
    area_list = []
    for region in regions:
        area_list.append(region.area) #包含所有连通区域的面积
    if area_list:
        idx_max = np.argmax(area_list) #去除所有区域中面积最大的区域的索引，注意与label_image标签的索引相差1
        binary[label_image != idx_max + 1] = 0 #将label_image中，除与(idx_max+1)相同的标签索引，其余全设置为0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int)) #填充处理后图像的孔洞并返回结果

def postprocessing(prediction, threshold=0.75, dataset='G'):
    if dataset[0] == 'D':
        # prediction = prediction.numpy()
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        disc_mask = (disc_mask > 0.5)  # return binary mask
        cup_mask = (cup_mask > 0.1)  # return binary mask
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        # for i in range(5):
        #     disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        #     cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy
    else:
        prediction = torch.sigmoid(prediction).data.cpu().numpy()#将logit转化成概率值

        #以下作为效果对比
        # disc_mask = scipy.signal.medfilt2d(disc_mask, 7) #中值滤波，用于平滑边缘和去除椒盐噪声
        # cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.erosion(disc_mask, morphology.diamond(3))  # 使用3*3的菱形核进行腐蚀操作，return 0,1
        # cup_mask = morphology.erosion(cup_mask, morphology.diamond(3))  # return 0,1

        prediction_copy = np.copy(prediction)
        prediction_copy = (prediction_copy > threshold)  # return binary mask
        prediction_copy = prediction_copy.astype(np.uint8) #将bool类型掩码设置成uint8的格式
        disc_mask = prediction_copy[1]
        cup_mask = prediction_copy[0]
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        # selem = disk(6)
        # disc_mask = morphology.closing(disc_mask, selem)
        # cup_mask = morphology.closing(cup_mask, selem)
        # print(sum(disc_mask))


        return prediction_copy


def joint_val_image(image, prediction, mask):
    ratio = 0.5
    _pred_cup = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _pred_disc = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _mask = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    image = np.transpose(image, (1, 2, 0)) #C*W*H -> W*H*C

    _pred_cup[:, :, 0] = prediction[0] #三通道图
    _pred_cup[:, :, 1] = prediction[0]
    _pred_cup[:, :, 2] = prediction[0]
    _pred_disc[:, :, 0] = prediction[1]
    _pred_disc[:, :, 1] = prediction[1]
    _pred_disc[:, :, 2] = prediction[1]
    _mask[:,:,0] = mask[0]
    _mask[:,:,1] = mask[1]

    pred_cup = np.add(ratio * image, (1 - ratio) * _pred_cup)
    pred_disc = np.add(ratio * image, (1 - ratio) * _pred_disc)
    mask_img = np.add(ratio * image, (1 - ratio) * _mask)

    joint_img = np.concatenate([image, mask_img, pred_cup, pred_disc], axis=1) #沿宽度通道拼接成一长条的图
    return joint_img


def save_val_img(path, epoch, img):
    name = osp.join(path, "visualization", "epoch_%d.png" % epoch)
    out = osp.join(path, "visualization")
    if not osp.exists(out):
        os.makedirs(out)
    img_shape = img[0].shape #img的形状是 B*C*W*H,img的形状是 C*W*H
    stack_image = np.zeros([len(img) * img_shape[0], img_shape[1], img_shape[2]])
    for i in range(len(img)): #len(img)返回第一维的大小，即批次大小B
        stack_image[i * img_shape[0] : (i + 1) * img_shape[0], :, : ] = img[i]
    imsave(name, stack_image)





def save_per_img(patch_image, data_save_path, img_name, prob_map, gt=None, mask_path=None, ext="bmp"):
    path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')
    path0 = os.path.join(data_save_path, 'original_image', img_name.split('.')[0]+'.png')
    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    disc_map = prob_map[0] #结合后面，反向推理出输入的prob_map形状是C*H*W，两通道
    cup_map = prob_map[1]

    ## 清除视盘概率图边界（避免边缘伪影影响轮廓检测）
    size = disc_map.shape
    disc_map[:, 0] = np.zeros(size[0]) #左边界置零
    disc_map[:, size[1] - 1] = np.zeros(size[0]) #右边界置零
    disc_map[0, :] = np.zeros(size[1]) #上边界置零
    disc_map[size[0] - 1, :] = np.zeros(size[1]) #下边界置零

    size = cup_map.shape
    cup_map[:, 0] = np.zeros(size[0])
    cup_map[:, size[1] - 1] = np.zeros(size[0])
    cup_map[0, :] = np.zeros(size[1])
    cup_map[size[0] - 1, :] = np.zeros(size[1])

    # disc_mask = (disc_map > 0.75) # return binary mask
    # cup_mask = (cup_map > 0.75)
    # disc_mask = disc_mask.astype(np.uint8)
    # cup_mask = cup_mask.astype(np.uint8)


    contours_disc = measure.find_contours(disc_map, 0.5) #以0.5为阈值寻找边界
    contours_cup = measure.find_contours(cup_map, 0.5)

    #绘制预测值的轮廓
    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]

    disc_mask = get_largest_fillhole(gt[0].numpy()).astype(np.uint8)  # return 0,1
    cup_mask = get_largest_fillhole(gt[1].numpy()).astype(np.uint8)

    #绘制真实值的轮廓
    contours_disc = measure.find_contours(disc_mask, 0.5)
    contours_cup = measure.find_contours(cup_mask, 0.5)
    red = [255, 0, 0]
    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red


    patch_image = patch_image.astype(np.uint8)
    patch_image = Image.fromarray(patch_image)

    patch_image.save(path1)