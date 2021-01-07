# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:41.549032Z","iopub.status.busy":"2020-12-14T19:52:41.548343Z","iopub.status.idle":"2020-12-14T19:52:41.553026Z","shell.execute_reply":"2020-12-14T19:52:41.552269Z"},"papermill":{"duration":0.033458,"end_time":"2020-12-14T19:52:41.553131","exception":false,"start_time":"2020-12-14T19:52:41.519673","status":"completed"},"tags":[]}
# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = '../'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = '../input/ranzcr-clip-catheter-line-classification/train'


# %% [markdown] {"papermill":{"duration":0.02073,"end_time":"2020-12-14T19:52:41.594446","exception":false,"start_time":"2020-12-14T19:52:41.573716","status":"completed"},"tags":[]}
# # CFG

# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:41.647322Z","iopub.status.busy":"2020-12-14T19:52:41.646482Z","iopub.status.idle":"2020-12-14T19:52:41.650102Z","shell.execute_reply":"2020-12-14T19:52:41.649535Z"},"papermill":{"duration":0.034829,"end_time":"2020-12-14T19:52:41.650216","exception":false,"start_time":"2020-12-14T19:52:41.615387","status":"completed"},"tags":[]}
# ====================================================
# CFG
# ====================================================
class CFG:
    debug = False
    device = 'GPU'  # ['TPU', 'GPU']
    nprocs = 1  # [1, 8]
    print_freq = 100
    num_workers = 4
    model_name = 'resnet200d_320'
    teacher = '../input/ranzcr-resnet200d-3-stage-training-step1/resnet200d_320_fold0_best_loss_cpu.pth'
    weights = [0.5, 1]
    size = 512
    scheduler = 'CosineAnnealingLR'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs = 5
    # factor=0.2 # ReduceLROnPlateau
    # patience=4 # ReduceLROnPlateau
    # eps=1e-6 # ReduceLROnPlateau
    T_max = 5  # CosineAnnealingLR
    # T_0=5 # CosineAnnealingWarmRestarts
    lr = 5e-4  # 1e-4
    min_lr = 1e-6
    batch_size = 16  # 64
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 416
    target_size = 11
    target_cols = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                   'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                   'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                   'Swan Ganz Catheter Present']
    n_fold = 5
    trn_fold = [0]  # [0, 1, 2, 3, 4]
    train = True


if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)

# %% [code]
if CFG.device == 'TPU':
    import os

    os.system(
        'curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
    os.system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')
    os.system('export XLA_USE_BF16=1')
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

    CFG.lr = CFG.lr * CFG.nprocs
    CFG.batch_size = CFG.batch_size // CFG.nprocs

# %% [markdown] {"papermill":{"duration":0.02152,"end_time":"2020-12-14T19:52:41.693202","exception":false,"start_time":"2020-12-14T19:52:41.671682","status":"completed"},"tags":[]}
# # Library

# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:41.750878Z","iopub.status.busy":"2020-12-14T19:52:41.750245Z","iopub.status.idle":"2020-12-14T19:52:45.494184Z","shell.execute_reply":"2020-12-14T19:52:45.492665Z"},"papermill":{"duration":3.779959,"end_time":"2020-12-14T19:52:45.49431","exception":false,"start_time":"2020-12-14T19:52:41.714351","status":"completed"},"tags":[]}
# ====================================================
# Library
# ====================================================
import sys

sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

import os
import ast
import copy
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, HueSaturationValue, CoarseDropout
)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import timm

if CFG.device == 'TPU':
    import ignite.distributed as idist
elif CFG.device == 'GPU':
    from torch.cuda.amp import autocast, GradScaler

import warnings

warnings.filterwarnings('ignore')


# %% [markdown] {"papermill":{"duration":0.021243,"end_time":"2020-12-14T19:52:45.536479","exception":false,"start_time":"2020-12-14T19:52:45.515236","status":"completed"},"tags":[]}
# # Utils

# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:45.59513Z","iopub.status.busy":"2020-12-14T19:52:45.594471Z","iopub.status.idle":"2020-12-14T19:52:45.60117Z","shell.execute_reply":"2020-12-14T19:52:45.60042Z"},"papermill":{"duration":0.040687,"end_time":"2020-12-14T19:52:45.601288","exception":false,"start_time":"2020-12-14T19:52:45.560601","status":"completed"},"tags":[]}
# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:, i], y_pred[:, i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR + 'train.log'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=CFG.seed)

# %% [markdown]
# # Data Loading

# %% [code]
train = pd.read_csv('../input/ranzcr-clip-catheter-line-classification/train.csv')
folds = pd.read_csv('../input/ranzcr-folds/folds.csv')
train_annotations = pd.read_csv('../input/ranzcr-clip-catheter-line-classification/train_annotations.csv')

# %% [markdown] {"papermill":{"duration":0.023234,"end_time":"2020-12-14T19:52:45.910123","exception":false,"start_time":"2020-12-14T19:52:45.886889","status":"completed"},"tags":[]}
# # Dataset

# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:45.967747Z","iopub.status.busy":"2020-12-14T19:52:45.96573Z","iopub.status.idle":"2020-12-14T19:52:45.968559Z","shell.execute_reply":"2020-12-14T19:52:45.969115Z"},"papermill":{"duration":0.036559,"end_time":"2020-12-14T19:52:45.969232","exception":false,"start_time":"2020-12-14T19:52:45.932673","status":"completed"},"tags":[]}
# ====================================================
# Dataset
# ====================================================
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


class TrainDataset(Dataset):
    def __init__(self, df, df_annotations, use_annot=False, annot_size=50, transform=None):
        self.df = df
        self.df_annotations = df_annotations
        self.use_annot = use_annot
        self.annot_size = annot_size
        self.file_names = df['StudyInstanceUID'].values
        self.labels = df[CFG.target_cols].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TRAIN_PATH}/{file_name}.jpg'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = torch.tensor(self.labels[idx]).float()
        if self.use_annot:
            image_annot = image.copy()
            query_string = f"StudyInstanceUID == '{file_name}'"
            df = self.df_annotations.query(query_string)
            for i, row in df.iterrows():
                label = row["label"]
                data = np.array(ast.literal_eval(row["data"]))
                for d in data:
                    image_annot[d[1] - self.annot_size // 2:d[1] + self.annot_size // 2,
                    d[0] - self.annot_size // 2:d[0] + self.annot_size // 2,
                    :] = COLOR_MAP[label]
            if self.transform:
                augmented = self.transform(image=image, image_annot=image_annot)
                image = augmented['image']
                image_annot = augmented['image_annot']
            return image, image_annot, labels
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, labels


# %% [markdown] {"papermill":{"duration":0.024022,"end_time":"2020-12-14T19:52:46.01618","exception":false,"start_time":"2020-12-14T19:52:45.992158","status":"completed"},"tags":[]}
# # Transforms

# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:46.071509Z","iopub.status.busy":"2020-12-14T19:52:46.070642Z","iopub.status.idle":"2020-12-14T19:52:46.074136Z","shell.execute_reply":"2020-12-14T19:52:46.073613Z"},"papermill":{"duration":0.035507,"end_time":"2020-12-14T19:52:46.074245","exception":false,"start_time":"2020-12-14T19:52:46.038738","status":"completed"},"tags":[]}
# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    if data == 'train':
        return Compose([
            # Resize(CFG.size, CFG.size),
            RandomResizedCrop(CFG.size, CFG.size, scale=(0.85, 1.0)),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
            ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            CoarseDropout(p=0.2),
            Cutout(p=0.2, max_h_size=16, max_w_size=16, fill_value=(0., 0., 0.), num_holes=16),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ], additional_targets={'image_annot': 'image'})

    elif data == 'check':
        return Compose([
            # Resize(CFG.size, CFG.size),
            RandomResizedCrop(CFG.size, CFG.size, scale=(0.85, 1.0)),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
            ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            CoarseDropout(p=0.2),
            Cutout(p=0.2, max_h_size=16, max_w_size=16, fill_value=(0., 0., 0.), num_holes=16),
            # Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225],
            # ),
            ToTensorV2(),
        ], additional_targets={'image_annot': 'image'})

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# %% [code]
from matplotlib import pyplot as plt

train_dataset = TrainDataset(
    folds[folds['StudyInstanceUID'].isin(train_annotations['StudyInstanceUID'].unique())].reset_index(drop=True),
    train_annotations, use_annot=True, transform=get_transforms(data='check'))

for i in range(5):
    image, image_annot, label = train_dataset[i]
    plt.subplot(1, 2, 1)
    plt.imshow(image.transpose(0, 1).transpose(1, 2))
    plt.subplot(1, 2, 2)
    plt.imshow(image_annot.transpose(0, 1).transpose(1, 2))
    plt.title(f'label: {label}')
    plt.show()


# %% [markdown] {"papermill":{"duration":0.022168,"end_time":"2020-12-14T19:52:46.118843","exception":false,"start_time":"2020-12-14T19:52:46.096675","status":"completed"},"tags":[]}
# # MODEL

# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:46.17627Z","iopub.status.busy":"2020-12-14T19:52:46.170979Z","iopub.status.idle":"2020-12-14T19:52:46.18312Z","shell.execute_reply":"2020-12-14T19:52:46.183763Z"},"papermill":{"duration":0.042914,"end_time":"2020-12-14T19:52:46.183878","exception":false,"start_time":"2020-12-14T19:52:46.140964","status":"completed"},"tags":[]}
# ====================================================
# MODEL
# ====================================================
class CustomResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d_320', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        if pretrained:
            pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
            self.model.load_state_dict(torch.load(pretrained_path))
            print(f'load {model_name} pretrained model')
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return features, pooled_features, output


# %% [markdown]
# # Loss

# %% [code]
class CustomLoss(nn.Module):
    def __init__(self, weights=[1, 1]):
        super(CustomLoss, self).__init__()
        self.weights = weights

    def forward(self, teacher_features, features, y_pred, labels):
        consistency_loss = nn.MSELoss()(teacher_features.view(-1), features.view(-1))
        cls_loss = nn.BCEWithLogitsLoss()(y_pred, labels)
        loss = self.weights[0] * consistency_loss + self.weights[1] * cls_loss
        return loss


# %% [markdown] {"papermill":{"duration":0.022356,"end_time":"2020-12-14T19:52:46.228883","exception":false,"start_time":"2020-12-14T19:52:46.206527","status":"completed"},"tags":[]}
# # Helper functions

# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:46.445205Z","iopub.status.busy":"2020-12-14T19:52:46.443326Z","iopub.status.idle":"2020-12-14T19:52:46.445944Z","shell.execute_reply":"2020-12-14T19:52:46.446449Z"},"papermill":{"duration":0.064625,"end_time":"2020-12-14T19:52:46.446561","exception":false,"start_time":"2020-12-14T19:52:46.381936","status":"completed"},"tags":[]}
# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, teacher_model, model, criterion, optimizer, epoch, scheduler, device):
    if CFG.device == 'GPU':
        scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, images_annot, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
            teacher_features, _, _ = teacher_model(images_annot.to(device))
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        if CFG.device == 'GPU':
            with autocast():
                features, _, y_preds = model(images)
                loss = criterion(teacher_features, features, y_preds, labels)
                # record loss
                losses.update(loss.item(), batch_size)
                if CFG.gradient_accumulation_steps > 1:
                    loss = loss / CFG.gradient_accumulation_steps
                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                if (step + 1) % CFG.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
        elif CFG.device == 'TPU':
            features, _, y_preds = model(images)
            loss = criterion(teacher_features, features, y_preds, labels)
            # record loss
            losses.update(loss.item(), batch_size)
            if CFG.gradient_accumulation_steps > 1:
                loss = loss / CFG.gradient_accumulation_steps
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            if (step + 1) % CFG.gradient_accumulation_steps == 0:
                xm.optimizer_step(optimizer, barrier=True)
                optimizer.zero_grad()
                global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if CFG.device == 'GPU':
            if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Elapsed {remain:s} '
                      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                      'Grad: {grad_norm:.4f}  '
                    # 'LR: {lr:.6f}  '
                    .format(
                    epoch + 1, step, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    grad_norm=grad_norm,
                    # lr=scheduler.get_lr()[0],
                ))
        elif CFG.device == 'TPU':
            if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
                xm.master_print('Epoch: [{0}][{1}/{2}] '
                                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                'Elapsed {remain:s} '
                                'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                                'Grad: {grad_norm:.4f}  '
                    # 'LR: {lr:.6f}  '
                    .format(
                    epoch + 1, step, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    grad_norm=grad_norm,
                    # lr=scheduler.get_lr()[0],
                ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    trues = []
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            _, _, y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        trues.append(labels.to('cpu').numpy())
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if CFG.device == 'GPU':
            if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
                print('EVAL: [{0}/{1}] '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'Elapsed {remain:s} '
                      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    .format(
                    step, len(valid_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                ))
        elif CFG.device == 'TPU':
            if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
                xm.master_print('EVAL: [{0}/{1}] '
                                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                                'Elapsed {remain:s} '
                                'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    .format(
                    step, len(valid_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                ))
    trues = np.concatenate(trues)
    predictions = np.concatenate(preds)
    return losses.avg, predictions, trues


# %% [markdown] {"papermill":{"duration":0.022557,"end_time":"2020-12-14T19:52:46.492442","exception":false,"start_time":"2020-12-14T19:52:46.469885","status":"completed"},"tags":[]}
# # Train loop

# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:46.552529Z","iopub.status.busy":"2020-12-14T19:52:46.541988Z","iopub.status.idle":"2020-12-14T19:52:46.608359Z","shell.execute_reply":"2020-12-14T19:52:46.609161Z"},"papermill":{"duration":0.093848,"end_time":"2020-12-14T19:52:46.609353","exception":false,"start_time":"2020-12-14T19:52:46.515505","status":"completed"},"tags":[]}
# ====================================================
# Train loop
# ====================================================
def train_loop(folds, fold):
    if CFG.device == 'GPU':
        LOGGER.info(f"========== fold: {fold} training ==========")
    elif CFG.device == 'TPU':
        if CFG.nprocs == 1:
            LOGGER.info(f"========== fold: {fold} training ==========")
        elif CFG.nprocs == 8:
            xm.master_print(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_folds = train_folds[
        train_folds['StudyInstanceUID'].isin(train_annotations['StudyInstanceUID'].unique())].reset_index(drop=True)

    valid_labels = valid_folds[CFG.target_cols].values

    train_dataset = TrainDataset(train_folds, train_annotations, use_annot=True,
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, train_annotations, use_annot=False,
                                 transform=get_transforms(data='valid'))

    if CFG.device == 'GPU':
        train_loader = DataLoader(train_dataset,
                                  batch_size=CFG.batch_size,
                                  shuffle=True,
                                  num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=CFG.batch_size * 2,
                                  shuffle=False,
                                  num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    elif CFG.device == 'TPU':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=xm.xrt_world_size(),
                                                                        rank=xm.get_ordinal(),
                                                                        shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=CFG.batch_size,
                                                   sampler=train_sampler,
                                                   drop_last=True,
                                                   num_workers=CFG.num_workers)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                                        num_replicas=xm.xrt_world_size(),
                                                                        rank=xm.get_ordinal(),
                                                                        shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=CFG.batch_size * 2,
                                                   sampler=valid_sampler,
                                                   drop_last=False,
                                                   num_workers=CFG.num_workers)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True,
                                          eps=CFG.eps)
        elif CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    if CFG.device == 'TPU':
        device = xm.xla_device()
    elif CFG.device == 'GPU':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model = CustomResNet200D(CFG.model_name, pretrained=False)
    teacher_model.load_state_dict(torch.load(CFG.teacher)['model'])
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    teacher_model.to(device)

    model = CustomResNet200D(CFG.model_name, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    train_criterion = CustomLoss(weights=CFG.weights)
    valid_criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        if CFG.device == 'TPU':
            if CFG.nprocs == 1:
                avg_loss = train_fn(train_loader, teacher_model, model, train_criterion, optimizer, epoch, scheduler,
                                    device)
            elif CFG.nprocs == 8:
                para_train_loader = pl.ParallelLoader(train_loader, [device])
                avg_loss = train_fn(para_train_loader.per_device_loader(device), teacher_model, model, train_criterion,
                                    optimizer, epoch, scheduler, device)
        elif CFG.device == 'GPU':
            avg_loss = train_fn(train_loader, teacher_model, model, train_criterion, optimizer, epoch, scheduler,
                                device)

        # eval
        if CFG.device == 'TPU':
            if CFG.nprocs == 1:
                avg_val_loss, preds, _ = valid_fn(valid_loader, model, valid_criterion, device)
            elif CFG.nprocs == 8:
                para_valid_loader = pl.ParallelLoader(valid_loader, [device])
                avg_val_loss, preds, valid_labels = valid_fn(para_valid_loader.per_device_loader(device), model,
                                                             valid_criterion, device)
                preds = idist.all_gather(torch.tensor(preds)).to('cpu').numpy()
                valid_labels = idist.all_gather(torch.tensor(valid_labels)).to('cpu').numpy()
        elif CFG.device == 'GPU':
            avg_val_loss, preds, _ = valid_fn(valid_loader, model, valid_criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score, scores = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        if CFG.device == 'GPU':
            LOGGER.info(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}')
        elif CFG.device == 'TPU':
            if CFG.nprocs == 1:
                LOGGER.info(
                    f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}')
            elif CFG.nprocs == 8:
                xm.master_print(
                    f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
                xm.master_print(f'Epoch {epoch + 1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}')

        if score > best_score:
            best_score = score
            if CFG.device == 'GPU':
                LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'preds': preds},
                           OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_score.pth')
            elif CFG.device == 'TPU':
                if CFG.nprocs == 1:
                    LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
                elif CFG.nprocs == 8:
                    xm.master_print(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
                xm.save({'model': model,
                         'preds': preds},
                        OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_score.pth')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if CFG.device == 'GPU':
                LOGGER.info(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'preds': preds},
                           OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_loss.pth')
            elif CFG.device == 'TPU':
                if CFG.nprocs == 1:
                    LOGGER.info(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
                elif CFG.nprocs == 8:
                    xm.master_print(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
                xm.save({'model': model,
                         'preds': preds},
                        OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_loss.pth')

    if CFG.nprocs != 8:
        check_point = torch.load(OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_score.pth')
        for c in [f'pred_{c}' for c in CFG.target_cols]:
            valid_folds[c] = np.nan
        valid_folds[[f'pred_{c}' for c in CFG.target_cols]] = check_point['preds']

    return valid_folds


# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:46.697088Z","iopub.status.busy":"2020-12-14T19:52:46.694995Z","iopub.status.idle":"2020-12-14T19:52:46.697997Z","shell.execute_reply":"2020-12-14T19:52:46.698568Z"},"papermill":{"duration":0.043278,"end_time":"2020-12-14T19:52:46.698687","exception":false,"start_time":"2020-12-14T19:52:46.655409","status":"completed"},"tags":[]}
# ====================================================
# main
# ====================================================
def main():
    """
    Prepare: 1.train  2.folds
    """

    def get_result(result_df):
        preds = result_df[[f'pred_{c}' for c in CFG.target_cols]].values
        labels = result_df[CFG.target_cols].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}')

    if CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(folds, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                if CFG.nprocs != 8:
                    LOGGER.info(f"========== fold: {fold} result ==========")
                    get_result(_oof_df)

        if CFG.nprocs != 8:
            # CV result
            LOGGER.info(f"========== CV ==========")
            get_result(oof_df)
            # save result
            oof_df.to_csv(OUTPUT_DIR + 'oof_df.csv', index=False)


# %% [code] {"execution":{"iopub.execute_input":"2020-12-14T19:52:46.757077Z","iopub.status.busy":"2020-12-14T19:52:46.755987Z"},"papermill":{"duration":null,"end_time":null,"exception":false,"start_time":"2020-12-14T19:52:46.725364","status":"running"},"tags":[]}
if __name__ == '__main__':
    if CFG.device == 'TPU':
        def _mp_fn(rank, flags):
            torch.set_default_tensor_type('torch.FloatTensor')
            a = main()


        FLAGS = {}
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=CFG.nprocs, start_method='fork')
    elif CFG.device == 'GPU':
        main()

# %% [code]
# save as cpu
if CFG.device == 'TPU':
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            # best score
            state = torch.load(OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_score.pth')
            torch.save({'model': state['model'].to('cpu').state_dict(),
                        'preds': state['preds']},
                       OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_score_cpu.pth')
            # best loss
            state = torch.load(OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_loss.pth')
            torch.save({'model': state['model'].to('cpu').state_dict(),
                        'preds': state['preds']},
                       OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_loss_cpu.pth')