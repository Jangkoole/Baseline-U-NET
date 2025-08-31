import torch
import numpy as np

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

def _upscan(f):
    for i, fi in enumerate(f):
        if fi == np.inf: continue
        for j in range(1,i+1):
            x = fi+j*j
            if f[i-j] < x: break
            f[i-j] = x

def distance_transform(bitmap):
    f = np.where(bitmap, 0.0, np.inf)
    for ibatch in range(f.shape[0]):
        for i in range(f.shape[1]):
            _upscan(f[ibatch, i, :])
            _upscan(f[ibatch, i,::-1])
        for i in range(f.shape[2]):
            _upscan(f[ibatch, :,i])
            _upscan(f[ibatch, ::-1,i])
            np.sqrt(f[ibatch], f[ibatch])
    return f

def WatershedCrossEntropy(input, target):

    # Distance Transform
    discmap = target.data.cpu()[:, 0, :, :]
    cupmap = target.data.cpu()[:, 1, :, :]
    disc_DT = distance_transform(discmap)
    cup_DT = distance_transform(cupmap)
    disc_DT = torch.from_numpy(disc_DT).float()
    cup_DT = torch.from_numpy(cup_DT).float()

    disc_DT = discmap * (1.0 - disc_DT/torch.max(disc_DT)) + 1.0
    cup_DT = cupmap * (1.0 - cup_DT/torch.max(cup_DT)) + 1.0

    disc_DT = disc_DT.cuda()
    cup_DT = cup_DT.cuda()

    CEloss = bce(input, target)

    return torch.mean(disc_DT* CEloss[:, 0 , :, :]+
                      cup_DT*CEloss[:, 1 , :, :])

def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool_)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool_)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value


def dice_coeff_2label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    # pred = torch.sigmoid(pred)
    # pred = pred.data.cpu()
    # pred[pred > 0.75] = 1
    # pred[pred <= 0.75] = 0
    # print target.shape
    # print pred.shape
    if len(pred.shape) == 3: #2*H*W
        return dice_coefficient_numpy(pred[0, ...], target[0, ...]), dice_coefficient_numpy(pred[1, ...], target[1, ...])
    else:
        dice_cup = []
        dice_disc = []
        for i in range(pred.shape[0]):
            cup, disc = dice_coefficient_numpy(pred[i, 0, ...], target[i, 0, ...]), dice_coefficient_numpy(pred[i, 1, ...], target[i, 1, ...])
            dice_cup.append(cup)
            dice_disc.append(disc)
    return sum(dice_cup) / len(dice_cup), sum(dice_disc) / len(dice_disc)

