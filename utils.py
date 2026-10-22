import torch
import torchvision
import pandas as pd
import torch.nn as nn
import sys
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from image_transformer import TwoCropTransform
import timm
from torch.utils import model_zoo


def load_robust_model(model_name):
    if model_name in ['res50_sin', 'res50_sin_in', 'res50_sin_fine_in','res50_augmix']:
        local_weights = {
            'res50_sin': 'C:/Users/lab3080/.cache/torch/hub/checkpoints/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'res50_sin_in': 'C:/Users/lab3080/.cache/torch/hub/checkpoints/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'res50_sin_fine_in': 'C:/Users/lab3080/.cache/torch/hub/checkpoints/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
            'res50_augmix': 'C:/Users/lab3080/.cache/torch/hub/checkpoints/checkpoint.pth.tar'
        }
        
        model_t = torchvision.models.resnet50(pretrained=False)
        model_t = torch.nn.DataParallel(model_t).cuda()
        
        checkpoint = torch.load(local_weights[model_name])
        model_t.load_state_dict(checkpoint["state_dict"])
    elif model_name == 'adv_incv3':
        model_t = timm.create_model('adv_inception_v3', pretrained=True)
    elif model_name == 'ens_inc_res_v2':
        model_t = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
    return model_t

# Load ImageNet model to evaluate
def load_model(model_name):
    # Load Targeted Model
    if model_name == 'dense201':
        model_t = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg19':
        model_t = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg16':
        model_t = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'googlenet':
        model_t = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1)
    elif model_name == 'incv3':
        model_t = torchvision.models.inception_v3( weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    elif model_name == 'res152':
        model_t = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
    elif model_name == 'res50':
        model_t = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'dense121':
        model_t = torchvision.models.densenet121( weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    elif model_name == 'incv4':
        model_t = timm.create_model('inception_v4', pretrained=True)
    elif model_name == "inc_res_v2":
        model_t = timm.create_model('inception_resnet_v2', pretrained=True)
    elif model_name in ['res50_sin', 'res50_sin_in', 'res50_sin_fine_in', 'adv_incv3', 'ens_inc_res_v2','res50_augmix']:
        model_t = load_robust_model(model_name)
    elif model_name in ['vit_base_patch16_224', 'pit_b_224','visformer_small', 'swin_tiny_patch4_window7_224']:
        model_t = timm.create_model(model_name, pretrained=True)
    else:
        raise ValueError
    return model_t


def fix_labels(args, test_set):
    val_dict = {}
    with open("val.txt") as file:
        for line in file:
            (key, val) = line.split(',')
            val_dict[key.split('.')[0]] = int(val.strip())

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        org_label = val_dict[test_set.samples[i][0].split('/')[-1].split('.')[0]]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


# This will fix labels for NIPS ImageNet
def fix_labels_nips(args, test_set, pytorch=False, target_flag=False):
    """
    Map NIPS 2017 dev-set image filenames to labels via images.csv.
    pytorch=True -> labels are shifted from [1..1000] to [0..999].
    target_flag=True -> use TargetClass instead of TrueLabel.
    """
    filenames = [os.path.basename(p) for p, _ in test_set.samples]

    csv_path = os.path.join(args.data_dir, "images.csv")
    image_classes = pd.read_csv(csv_path)

    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(
        image_classes, on="ImageId", how="left"
    )
    if image_metadata["TrueLabel"].isna().any():
        missing = [filenames[i] for i, x in enumerate(image_metadata["TrueLabel"].isna()) if x]
        raise RuntimeError(
            f"{len(missing)} filenames not found in images.csv, e.g.: {missing[:5]}\n"
            f"Make sure --data_dir points to the folder containing images.csv and the images/ subfolder."
        )

    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = image_metadata["TargetClass"].tolist()
    val_dict = {f: [true_classes[i], target_classes[i]] for i, f in enumerate(filenames)}

    new_samples = []
    for path, _ in test_set.samples:
        fname = os.path.basename(path)
        org_label = val_dict[fname][1] if target_flag else val_dict[fname][0]
        if pytorch:
            org_label -= 1
        new_samples.append((path, org_label))

    test_set.samples = new_samples
    return test_set


def get_classes(label_flag):
    if label_flag == 'N8':
        label_set = np.array([150, 426, 843, 715, 952, 507, 590, 62])
    elif label_flag == 'C20':
        label_set = np.array([4, 65, 70, 160, 249, 285, 334, 366, 394, 396, 458, 580, 593, 681, 815, 822, 849,
                              875, 964, 986])
    elif label_flag == 'C50':
        label_set = np.array([9, 71, 74, 86, 102, 141, 150, 181, 188, 223, 245, 275, 308, 332, 343, 352, 386,
                              405, 426, 430, 432, 450, 476, 501, 510, 521, 529, 546, 554, 567, 588, 597, 640,
                              643, 688, 712, 715, 729, 817, 830, 853, 876, 878, 883, 894, 906, 917, 919, 940,
                              988])
    elif label_flag == 'C100':
        label_set = np.array([6, 8, 31, 41, 43, 47, 48, 50, 56, 57, 66, 89, 93, 107, 121, 124, 130, 156, 159,
                              168, 170, 172, 178, 180, 202, 206, 214, 219, 220, 230, 248, 252, 269, 304, 323,
                              325, 339, 351, 353, 356, 368, 374, 379, 387, 395, 401, 435, 449, 453, 464, 472,
                              496, 504, 505, 509, 512, 527, 530, 542, 575, 577, 604, 636, 638, 647, 682, 683,
                              687, 704, 711, 713, 730, 733, 739, 746, 747, 763, 766, 774, 778, 783, 799, 809,
                              832, 843, 845, 846, 891, 895, 907, 930, 937, 946, 950, 961, 963, 972, 977, 984,
                              998])
    elif label_flag == 'C200':
        label_set = np.array([7, 12, 13, 14, 16, 22, 25, 36, 49, 58, 75, 84, 88, 104, 105, 112, 113, 114, 115,
                              117, 120, 134, 140, 143, 144, 155, 158, 165, 173, 182, 183, 194, 196, 200, 204,
                              207, 212, 218, 225, 231, 242, 244, 250, 261, 262, 266, 270, 277, 282, 288, 292,
                              297, 301, 310, 316, 320, 321, 327, 330, 348, 357, 359, 361, 365, 371, 375, 381,
                              382, 389, 407, 409, 411, 412, 413, 414, 418, 422, 436, 437, 445, 446, 448, 456,
                              461, 468, 470, 471, 474, 475, 480, 484, 486, 489, 491, 495, 500, 502, 506, 511,
                              514, 515, 526, 531, 535, 544, 547, 549, 561, 562, 566, 582, 591, 598, 603, 605,
                              610, 611, 612, 613, 616, 618, 619, 621, 627, 635, 641, 648, 653, 654, 656, 657,
                              658, 661, 662, 672, 673, 680, 686, 689, 691, 693, 697, 700, 705, 706, 707, 716,
                              725, 735, 743, 750, 752, 760, 768, 772, 776, 781, 790, 791, 796, 798, 800, 802,
                              811, 819, 823, 824, 828, 833, 834, 836, 848, 855, 874, 890, 893, 898, 903, 922,
                              923, 928, 931, 935, 936, 939, 943, 944, 945, 948, 955, 960, 967, 969, 970, 971,
                              980, 983, 990, 992, 999])
    else:
        raise ValueError
    return label_set


def normalize(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
    return t



def get_data(train_dir, scale_size, img_size):
    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    train_set = datasets.ImageFolder(train_dir, TwoCropTransform(data_transform, img_size))
    train_size = len(train_set)
    print('Total data size:', train_size)
    return train_set
