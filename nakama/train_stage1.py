# %% [markdown]
# # About this notebook
# I share an example how to use annotated images to improve score.
# If this notebook is helpful, feel free to upvote :)
# ## Training strategy
# - [1st-stage training](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-step1)
#     - teacher model training for annotated image
#         - data: annotated data
#         - pretrained weight: imagenet weight
#         - `BCEWithLogitsLoss(y_preds, labels)`
#         - `y_preds: teacher model predictions for annotated image`
# - [2nd-stage training](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-step2)
#     - student model training with teacher model features
#         - data: annotated data
#         - student model pretrained weight: imagenet weight
#         - teacher model pretrained weight: 1st-stage weight
#         - `BCEWithLogitsLoss(y_preds, labels) + w * MSELoss(student_features, teacher_features)`
#         - `y_preds: student model predictions for normal image`
#         - `student_features: student model features for normal image`
#         - `teacher_features: teacher model features for annotated image`
# - [3rd-stage training](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-step3)
#     - model training
#         - data: all data
#         - pretrained weight: 2nd-stage weight
#         - `BCEWithLogitsLoss(y_preds, labels)`
#         - `y_preds: student model predictions for normal image`
# - [inference notebook](https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-sub)

# %% [markdown] {"papermill":{"duration":0.027734,"end_time":"2020-12-23T19:04:31.666166","exception":false,"start_time":"2020-12-23T19:04:31.638432","status":"completed"},"tags":[]}
# # Directory settings

# ====================================================
# Directory settings
# ====================================================
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

OUTPUT_DIR = './stage1/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = './input/ranzcr-clip-catheter-line-classification/train/'


# %% [markdown] {"papermill":{"duration":0.027722,"end_time":"2020-12-23T19:04:31.792603","exception":false,"start_time":"2020-12-23T19:04:31.764881","status":"completed"},"tags":[]}
# # CFG

# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:04:31.864601Z","iopub.status.busy":"2020-12-23T19:04:31.863568Z","iopub.status.idle":"2020-12-23T19:04:31.867028Z","shell.execute_reply":"2020-12-23T19:04:31.866253Z"},"papermill":{"duration":0.045622,"end_time":"2020-12-23T19:04:31.867167","exception":false,"start_time":"2020-12-23T19:04:31.821545","status":"completed"},"tags":[]}
# ====================================================
# CFG
# ====================================================
class CFG:
    debug = False
    device = 'GPU'  # ['TPU', 'GPU']
    nprocs = 1  # [1, 8]
    print_freq = 100
    num_workers = 4
    model_name = 'resnet50'
    size = 256
    scheduler = 'ReduceLROnPlateau'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs = 10
    factor = 0.5 # ReduceLROnPlateau
    patience = 3 # ReduceLROnPlateau
    eps=1e-6 # ReduceLROnPlateau
    T_max = 4  # CosineAnnealingLR
    # T_0=4 # CosineAnnealingWarmRestarts
    lr = 1e-3
    min_lr = 1e-6
    batch_size = 32  # 64
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
    trn_fold = [0, 1, 2, 3, 4]
    train = True


if CFG.debug:
    CFG.epochs = 1

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

# ====================================================
# Library
# ====================================================
import sys

#sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

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


# %% [markdown] {"papermill":{"duration":0.028493,"end_time":"2020-12-23T19:06:17.598011","exception":false,"start_time":"2020-12-23T19:06:17.569518","status":"completed"},"tags":[]}
# # Utils

# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:06:17.675794Z","iopub.status.busy":"2020-12-23T19:06:17.674877Z","iopub.status.idle":"2020-12-23T19:06:17.681689Z","shell.execute_reply":"2020-12-23T19:06:17.680811Z"},"papermill":{"duration":0.055059,"end_time":"2020-12-23T19:06:17.681837","exception":false,"start_time":"2020-12-23T19:06:17.626778","status":"completed"},"tags":[]}
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
train = pd.read_csv('./input/ranzcr-clip-catheter-line-classification/train.csv')
folds = pd.read_csv('./input/ranzcr-folds/folds.csv')
train_annotations = pd.read_csv('./input/ranzcr-clip-catheter-line-classification/train_annotations.csv')

# %% [markdown] {"papermill":{"duration":0.030724,"end_time":"2020-12-23T19:06:18.847705","exception":false,"start_time":"2020-12-23T19:06:18.816981","status":"completed"},"tags":[]}
# # Dataset

# %% [code]
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
    def __init__(self, df, df_annotations, annot_size=50, transform=None):
        self.df = df
        self.df_annotations = df_annotations
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
        query_string = f"StudyInstanceUID == '{file_name}'"
        df = self.df_annotations.query(query_string)
        for i, row in df.iterrows():
            label = row["label"]
            data = np.array(ast.literal_eval(row["data"]))
            for d in data:
                image[d[1] - self.annot_size // 2:d[1] + self.annot_size // 2,
                d[0] - self.annot_size // 2:d[0] + self.annot_size // 2,
                :] = COLOR_MAP[label]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).float()
        return image, label


# %% [markdown] {"papermill":{"duration":0.031645,"end_time":"2020-12-23T19:06:19.086874","exception":false,"start_time":"2020-12-23T19:06:19.055229","status":"completed"},"tags":[]}
# # Transforms

# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:06:19.163947Z","iopub.status.busy":"2020-12-23T19:06:19.161721Z","iopub.status.idle":"2020-12-23T19:06:19.167297Z","shell.execute_reply":"2020-12-23T19:06:19.168067Z"},"papermill":{"duration":0.048275,"end_time":"2020-12-23T19:06:19.168252","exception":false,"start_time":"2020-12-23T19:06:19.119977","status":"completed"},"tags":[]}
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
        ])

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:06:19.244888Z","iopub.status.busy":"2020-12-23T19:06:19.238449Z","iopub.status.idle":"2020-12-23T19:06:20.905143Z","shell.execute_reply":"2020-12-23T19:06:20.906145Z"},"papermill":{"duration":1.705488,"end_time":"2020-12-23T19:06:20.906438","exception":false,"start_time":"2020-12-23T19:06:19.20095","status":"completed"},"tags":[]}
from matplotlib import pyplot as plt

train_dataset = TrainDataset(
    folds[folds['StudyInstanceUID'].isin(train_annotations['StudyInstanceUID'].unique())].reset_index(drop=True),
    train_annotations, transform=None)

for i in range(5):
    image, label = train_dataset[i]
    plt.imshow(image)
    plt.title(f'label: {label}')
    plt.show()


# %% [markdown] {"papermill":{"duration":0.062385,"end_time":"2020-12-23T19:06:21.079214","exception":false,"start_time":"2020-12-23T19:06:21.016829","status":"completed"},"tags":[]}
# # MODEL

# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:06:21.216578Z","iopub.status.busy":"2020-12-23T19:06:21.215376Z","iopub.status.idle":"2020-12-23T19:06:21.225711Z","shell.execute_reply":"2020-12-23T19:06:21.226511Z"},"papermill":{"duration":0.088216,"end_time":"2020-12-23T19:06:21.226699","exception":false,"start_time":"2020-12-23T19:06:21.138483","status":"completed"},"tags":[]}
# ====================================================
# MODEL
# ====================================================
class CustomResNet(nn.Module):
    def __init__(self, model_name='resnet200d_320', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
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


# %% [markdown] {"papermill":{"duration":0.063942,"end_time":"2020-12-23T19:06:21.347849","exception":false,"start_time":"2020-12-23T19:06:21.283907","status":"completed"},"tags":[]}
# # Helper functions

# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:06:21.467293Z","iopub.status.busy":"2020-12-23T19:06:21.465046Z","iopub.status.idle":"2020-12-23T19:06:21.543666Z","shell.execute_reply":"2020-12-23T19:06:21.544562Z"},"papermill":{"duration":0.141122,"end_time":"2020-12-23T19:06:21.5448","exception":false,"start_time":"2020-12-23T19:06:21.403678","status":"completed"},"tags":[]}
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


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
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
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        if CFG.device == 'GPU':
            with autocast():
                _, _, y_preds = model(images)
                loss = criterion(y_preds, labels)
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
            _, _, y_preds = model(images)
            loss = criterion(y_preds, labels)
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


# %% [markdown] {"papermill":{"duration":0.070036,"end_time":"2020-12-23T19:06:21.680816","exception":false,"start_time":"2020-12-23T19:06:21.61078","status":"completed"},"tags":[]}
# # Train loop

# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:06:21.836318Z","iopub.status.busy":"2020-12-23T19:06:21.798625Z","iopub.status.idle":"2020-12-23T19:06:21.839883Z","shell.execute_reply":"2020-12-23T19:06:21.839093Z"},"papermill":{"duration":0.115843,"end_time":"2020-12-23T19:06:21.840016","exception":false,"start_time":"2020-12-23T19:06:21.724173","status":"completed"},"tags":[]}
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
    valid_folds = valid_folds[
        valid_folds['StudyInstanceUID'].isin(train_annotations['StudyInstanceUID'].unique())].reset_index(drop=True)

    valid_labels = valid_folds[CFG.target_cols].values

    train_dataset = TrainDataset(train_folds, train_annotations,
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, train_annotations,
                                 transform=get_transforms(data='valid'))

    if CFG.device == 'GPU':
        train_loader = DataLoader(train_dataset,
                                  batch_size=CFG.batch_size,
                                  shuffle=True,
                                  num_workers=CFG.num_workers, pin_memory=False, drop_last=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=CFG.batch_size * 2,
                                  shuffle=False,
                                  num_workers=CFG.num_workers, pin_memory=False, drop_last=False)
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

    model = CustomResNet(CFG.model_name, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        if CFG.device == 'TPU':
            if CFG.nprocs == 1:
                avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
            elif CFG.nprocs == 8:
                para_train_loader = pl.ParallelLoader(train_loader, [device])
                avg_loss = train_fn(para_train_loader.per_device_loader(device), model, criterion, optimizer, epoch,
                                    scheduler, device)
        elif CFG.device == 'GPU':
            avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        if CFG.device == 'TPU':
            if CFG.nprocs == 1:
                avg_val_loss, preds, _ = valid_fn(valid_loader, model, criterion, device)
            elif CFG.nprocs == 8:
                para_valid_loader = pl.ParallelLoader(valid_loader, [device])
                avg_val_loss, preds, valid_labels = valid_fn(para_valid_loader.per_device_loader(device), model,
                                                             criterion, device)
                preds = idist.all_gather(torch.tensor(preds)).to('cpu').numpy()
                valid_labels = idist.all_gather(torch.tensor(valid_labels)).to('cpu').numpy()
        elif CFG.device == 'GPU':
            avg_val_loss, preds, _ = valid_fn(valid_loader, model, criterion, device)

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


# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:06:21.949276Z","iopub.status.busy":"2020-12-23T19:06:21.937789Z","iopub.status.idle":"2020-12-23T19:06:21.956761Z","shell.execute_reply":"2020-12-23T19:06:21.955543Z"},"papermill":{"duration":0.071384,"end_time":"2020-12-23T19:06:21.95697","exception":false,"start_time":"2020-12-23T19:06:21.885586","status":"completed"},"tags":[]}
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


# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:06:22.059092Z","iopub.status.busy":"2020-12-23T19:06:22.057932Z","iopub.status.idle":"2020-12-23T19:57:36.57877Z","shell.execute_reply":"2020-12-23T19:57:36.580062Z"},"papermill":{"duration":3074.575173,"end_time":"2020-12-23T19:57:36.580679","exception":false,"start_time":"2020-12-23T19:06:22.005506","status":"completed"},"tags":[]}
if __name__ == '__main__':
    if CFG.device == 'TPU':
        def _mp_fn(rank, flags):
            torch.set_default_tensor_type('torch.FloatTensor')
            a = main()


        FLAGS = {}
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=CFG.nprocs, start_method='fork')
    elif CFG.device == 'GPU':
        main()

# %% [code] {"execution":{"iopub.execute_input":"2020-12-23T19:57:37.175743Z","iopub.status.busy":"2020-12-23T19:57:37.174833Z","iopub.status.idle":"2020-12-23T19:57:56.066285Z","shell.execute_reply":"2020-12-23T19:57:56.065385Z"},"papermill":{"duration":19.418317,"end_time":"2020-12-23T19:57:56.066443","exception":false,"start_time":"2020-12-23T19:57:36.648126","status":"completed"},"tags":[]}
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