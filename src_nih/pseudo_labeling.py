import pandas as pd
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
import os
import time
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
from albumentations import *
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import timm
import pickle
from sklearn.model_selection import train_test_split

import apex
from apex import amp

from warnings import filterwarnings
filterwarnings("ignore")

device = torch.device('cuda')

import argparse

import os
import cv2
import numpy as np
import pandas as pd
import albumentations
from albumentations import *
import torch
from torch.utils.data import Dataset
import ast

def get_transforms(image_size):
    transforms_train = albumentations.Compose([
        albumentations.RandomResizedCrop(image_size, image_size, scale=(0.9, 1), p=1),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
        albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.2),
        albumentations.OneOf([
            albumentations.GaussNoise(var_limit=[10, 50]),
            albumentations.GaussianBlur(),
            albumentations.MotionBlur(),
            albumentations.MedianBlur(),
        ], p=0.2),
        albumentations.Resize(image_size, image_size),
        albumentations.OneOf([
            JpegCompression(),
            Downscale(scale_min=0.1, scale_max=0.15),
        ], p=0.2),
        IAAPiecewiseAffine(p=0.2),
        IAASharpen(p=0.2),
        albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=5, p=0.5),
        albumentations.Normalize(),
    ])

    transforms_valid = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_valid


def get_transforms_needle(image_size):
    transforms_train = albumentations.Compose([
        albumentations.RandomResizedCrop(image_size, image_size, scale=(0.9, 1), p=1),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
        albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.2),
        albumentations.OneOf([
            albumentations.GaussNoise(var_limit=[10, 50]),
            albumentations.GaussianBlur(),
            albumentations.MotionBlur(),
            albumentations.MedianBlur(),
        ], p=0.2),
        albumentations.Resize(image_size, image_size),
        albumentations.OneOf([
            JpegCompression(),
            Downscale(scale_min=0.1, scale_max=0.15),
        ], p=0.2),
        IAAPiecewiseAffine(p=0.2),
        IAASharpen(p=0.2),
        albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=5, p=0.5),
        albumentations.Normalize(),
    ])

    transforms_valid = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_valid

class RANZERDataset_nih(Dataset):
    def __init__(self, df, transform=None):

        self.df = df.reset_index(drop=True)
        self.file_names = df['file_path'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        return torch.tensor(img).float()

class RANZCRResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        pretrained_path = '/home/ubuntu/lyh/RANZCR_kaggle/src_1/resnet200d_ra2-bdba9bf9.pth'
        self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                output = self.fc(dropout(pooled_features))
            else:
                output += self.fc(dropout(pooled_features))
        output /= len(self.dropouts)
        return output

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # for faster training, but not deterministic

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

# read csv
root_dir =  '/home/ubuntu/lyh/RANZCR_kaggle/'
data_dir = f'{root_dir}/input/ranzcr-clip-catheter-line-classification/train'
df_nih = pd.read_csv(f'{root_dir}/src_nih/nih_train.csv')
#df_nih['file_path'] = df_nih['path']
df_nih['file_path'] = df_nih['path'].apply(lambda x: x.replace('..', root_dir))

df_ranzcr = pd.read_csv(f'{root_dir}/input/how-to-properly-split-folds/train_folds.csv')
df_ranzcr['file_path'] = df_ranzcr.StudyInstanceUID.apply(lambda x: os.path.join(data_dir, f'{x}.jpg'))

# prepare dataset
transforms_train, transforms_valid = get_transforms(512)

dataset_train = RANZERDataset_nih(df_nih, transform=transforms_valid)

train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=64,
                                           shuffle=False,
                                           num_workers=4)

# prepare model

enet_type = ['resnet200d'] * 5
model_paths = [f'{root_dir}/src_1/weights_exp1/resnet200d_fold0_best_AUC_0.9457.pth',
              f'{root_dir}/src_1/resnet200d_fold1_best_AUC_0.9527.pth',
              f'{root_dir}/src_1/resnet200d_fold2_best_AUC_0.9530.pth',
             f'{root_dir}/src_1/resnet200d_fold3_best_AUC_0.9528.pth',
             f'{root_dir}/src_1/resnet200d_fold4_best_AUC_0.9499.pth']

for model_path in model_paths:
    break

device = torch.device('cuda')
model = RANZCRResNet200D('resnet200d', out_dim=11, pretrained=True)
model = model.to(device)

from collections import OrderedDict

new_state_dict = OrderedDict()
state_dict = torch.load(model_path)
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = apex.parallel.convert_syncbn_model(model)
optimizer = optim.Adam(model.parameters())
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = nn.DataParallel(model)
model.eval()

PREDS = []
for images in tqdm(train_loader):
    images = images.to(device)
    logits = model(images)
    PREDS += [logits.sigmoid().detach().cpu()]

PREDS = torch.cat(PREDS).numpy()

target_cols = df_ranzcr.iloc[:, 1:12].columns.tolist()
pseudo_df = pd.DataFrame(PREDS)
pseudo_df.columns = target_cols
pseudo_df['file_path'] = df_nih['file_path']