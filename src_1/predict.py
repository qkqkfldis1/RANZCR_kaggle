import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
batch_size = 1
image_size = 512
tta = True
submit = True
enet_type = ['efficientnet_b5'] * 3
model_path = ['./weights/tf_efficientnet_b5_fold4_best_AUC_0.9373.pth',
              './weights/tf_efficientnet_b5_fold1_best_AUC_0.9376.pth',
              './weights/tf_efficientnet_b5_fold2_best_AUC_0.9408.pth',]
# you can save GPU quota using fast sub attached in the last markdown file
fast_sub = False
fast_sub_path = '../input/xxxxxx/your_submission.csv'


import os
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
import numpy as np
DEBUG = False
import time
import cv2
import PIL.Image
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import timm
from albumentations import *
from albumentations.pytorch import ToTensorV2
device = torch.device('cuda') if not DEBUG else torch.device('cpu')

class RANZCRResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output

class RANZCREffiNet(nn.Module):
    def __init__(self, model_name='efficientnet_b0', out_dim=11, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_ch = self.model.classifier.in_features
        #n_features = self.model.fc.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model.forward_features(x)
        pooled_features = self.pooling(features).view(bs, -1) # TODO Add harddrop
        output = self.fc(pooled_features)
        return output

transforms_test = albumentations.Compose([
    Resize(image_size, image_size),
    Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225],
     ),
    ToTensorV2()
])


class RANZCRDataset(Dataset):
    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.labels = df[target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        img = cv2.imread(row.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
        label = torch.tensor(self.labels[index]).float()
        if self.mode == 'test':
            return img
        else:
            return img, label


test = pd.read_csv('../input/ranzcr-clip-catheter-line-classification/sample_submission.csv')
test['file_path'] = test.StudyInstanceUID.apply(lambda x: os.path.join('../input/ranzcr-clip-catheter-line-classification/test', f'{x}.jpg'))
target_cols = test.iloc[:, 1:12].columns.tolist()

test_dataset = RANZCRDataset(test, 'test', transform=transforms_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,  num_workers=1)


def inference_func(test_loader):
    model.eval()
    bar = tqdm(test_loader)
    LOGITS = []
    PREDS = []

    with torch.no_grad():
        for batch_idx, images in enumerate(bar):
            x = images.to(device)
            logits = model(x)
            LOGITS.append(logits.cpu())
            PREDS += [logits.sigmoid().detach().cpu()]
        PREDS = torch.cat(PREDS).cpu().numpy()
        LOGITS = torch.cat(LOGITS).cpu().numpy()
    return PREDS


def tta_inference_func(test_loader):
    model.eval()
    bar = tqdm(test_loader)
    PREDS = []
    LOGITS = []

    with torch.no_grad():
        for batch_idx, images in enumerate(bar):
            x = images.to(device)
            x = torch.stack([x, x.flip(-1)], 0)  # hflip
            x = x.view(-1, 3, image_size, image_size)
            logits = model(x)
            logits = logits.view(batch_size, 2, -1).mean(1)
            PREDS += [logits.sigmoid().detach().cpu()]
            LOGITS.append(logits.cpu())
        PREDS = torch.cat(PREDS).cpu().numpy()

    return PREDS


#if submit:
test_preds = []
for i in range(len(enet_type)):
    if enet_type[i] == 'resnet200d':
        print('resnet200d loaded')
        model = RANZCRResNet200D(enet_type[i], out_dim=len(target_cols), pretrained=False)
        model = model.to(device)
    elif 'efficientnet' in enet_type[i]:
        print('efficientnet loaded')
        model = RANZCREffiNet(enet_type[i], out_dim=len(target_cols), pretrained=False)
        model = model.to(device)

    model.load_state_dict(torch.load(model_path[i]))
    if tta:
        test_preds += [tta_inference_func(test_loader)]
    else:
        test_preds += [inference_func(test_loader)]

submission = pd.read_csv('../input/ranzcr-clip-catheter-line-classification/sample_submission.csv')
submission[target_cols] = np.mean(test_preds, axis=0)
submission.to_csv(f'./submission_{enet_type[0]}.csv', index=False)
#else:
#    pd.read_csv('../input/ranzcr-clip-catheter-line-classification/sample_submission.csv').to_csv('submission.csv', index=False)