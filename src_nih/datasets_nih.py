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
    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.file_names = df['path'].values
        self.mode = mode
        self.transform = transform
        self.labels = df['Label'].values

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
        label = self.labels[index]
        target = torch.zeros((13))
        target[label] = 1
        if self.mode == 'test':
            return torch.tensor(img).float()
        else:
            return torch.tensor(img).float(), target

COLOR_MAP = {'ETT - Abnormal': (255, 0, 0),
             'ETT - Borderline': (0, 255, 0),
             'ETT - Normal': (0, 0, 255),
             'NGT - Abnormal': (255, 255, 0),
             'NGT - Borderline': (255, 0, 255),
             'NGT - Incompletely Imaged': (0, 255, 255),
             'NGT - Normal': (128, 0, 0),
             'CVC - Abnormal': (0, 128, 0),
             'CVC - Borderline': (0, 0, 128),
             'CVC - Normal': (128, 128, 0),
             'Swan Ganz Catheter Present': (128, 0, 128),
            }


class RANZERDataset_pre(Dataset):
    def __init__(self, df, df_annotations, target_cols, annot_size=50, transform=None):
        self.df = df
        self.df_annotations = df_annotations
        self.study_names = df['StudyInstanceUID'].values
        self.file_paths = df['file_path'].values
        self.labels = df[target_cols].values
        self.annot_size = annot_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_name = self.study_names[idx]
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        query_string = f"StudyInstanceUID == '{study_name}'"
        df = self.df_annotations.query(query_string)
        for i, row in df.iterrows():
            label = row["label"]
            data = np.array(ast.literal_eval(row["data"]))
            for d in data:
                image[d[1]-self.annot_size//2:d[1]+self.annot_size//2,
                      d[0]-self.annot_size//2:d[0]+self.annot_size//2,
                      :] = COLOR_MAP[label]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        image = image.transpose(2, 0, 1)
        label = torch.tensor(self.labels[idx]).float()
        return image, label