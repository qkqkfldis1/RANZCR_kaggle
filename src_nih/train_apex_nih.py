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
from src_nih.datasets_nih import get_transforms, RANZERDataset_nih
import pickle
from sklearn.model_selection import train_test_split

import apex
from apex import amp

from warnings import filterwarnings
filterwarnings("ignore")

device = torch.device('cuda')

import argparse

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='resnet200d')
    parser.add_argument('--exp', type=str, required=True)

    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init_lr', type=int, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--warmup_factor', type=int, default=10)
    parser.add_argument('--warmup_epo', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--early_stop', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--valid_batch_size', type=int, default=12)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_amp', action='store_true', default=True)


    parser.add_argument('--model_dir', type=str, default='./weights')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')

    args, _ = parser.parse_known_args()
    return args



class RANZCRResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        pretrained_path = './resnet200d_ra2-bdba9bf9.pth'
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

# class RANZCREffiNet(nn.Module):
#     def __init__(self, model_name='efficientnet_b0', out_dim=11, pretrained=True):
#         super().__init__()
#         self.model = timm.create_model(model_name, pretrained=pretrained)
#         n_features = self.model.classifier.in_features
#         self.model.classifier = nn.Identity()
#         self.model.global_pool = nn.Identity()
#         self.pooling = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(n_features, out_dim)

#     def forward(self, x):
#         bs = x.size(0)
#         features = self.model.forward_features(x)
#         pooled_features = self.pooling(features).view(bs, -1) # TODO Add harddrop
#         output = self.fc(pooled_features)
#         return output

class RANZCREffiNet(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)

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
    
class RANZCRViT(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, out_dim)

    def forward(self, x):
        x = self.model(x)
        return x
    

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # for faster training, but not deterministic



def macro_multilabel_auc(label, pred):
    aucs = []
    for i in range(13):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    print(np.round(aucs, 4))
    return np.mean(aucs)


def train_func(train_loader, model, optimizer, criterion):
    model.train()
    bar = tqdm(train_loader)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch_idx, (images, targets) in enumerate(bar):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        if args.use_amp:
            logits = model(images)
            loss = criterion(logits, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    loss_train = np.mean(losses)
    return loss_train


def valid_func(valid_loader, model, optimizer, criterion):
    model.eval()
    bar = tqdm(valid_loader)

    PROB = []
    TARGETS = []
    losses = []
    PREDS = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(bar):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            PREDS += [logits.sigmoid()]
            TARGETS += [targets.detach().cpu()]
            loss = criterion(logits, targets)
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])
            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    # roc_auc = roc_auc_score(TARGETS.reshape(-1), PREDS.reshape(-1))
    roc_auc = macro_multilabel_auc(TARGETS, PREDS)
    loss_valid = np.mean(losses)
    return loss_valid, roc_auc

def main():
    seed_everything(args.seed)
    logger = Logger()
    # try:
    #     os.system(f'rm {args.log_dir}/log.train_exp_{args.exp}_fold_{args.fold_id}.txt')
    # except:
    #     pass
    logger.open(f'{args.log_dir}/log.train_exp_{args.exp}_fold_{args.fold_id}.txt', mode='a')



    #data_dir = f'{args.root_dir}/src_nih/nih_train.csv'

    df_train = pd.read_csv(f'{args.root_dir}/src_nih/nih_train.csv')

    #dataset = RANZERDataset(df_train, 'train', transform=args.transforms_train)

    if 'efficientnet' in args.model_name:
        model = RANZCREffiNet(args.model_name, out_dim=13, pretrained=True)
    elif 'vit' in args.model_name:
        model = RANZCRViT(args.model_name, out_dim=13, pretrained=True)
    else:
        model = RANZCRResNet200D(args.model_name, out_dim=13, pretrained=True)
        
        
    if DP:
        model = apex.parallel.convert_syncbn_model(model)
    model = model.to(device)



    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr/args.warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min=1e-7)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10,
                                                total_epoch=args.warmup_epo,
                                                after_scheduler=scheduler_cosine)
    
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)
    
    df_train_this, df_valid_this = train_test_split(df_train, test_size=0.1, shuffle=True, random_state=42)
    df_train_this = df_train_this.reset_index(drop=True) #df_train[df_train['fold'] != args.fold_id]
    df_valid_this = df_valid_this.reset_index(drop=True)  #df_train[df_train['fold'] == args.fold_id]

    transforms_train, transforms_valid = get_transforms(args.image_size)

    dataset_train = RANZERDataset_nih(df_train_this, 'train', transform=transforms_train)
    dataset_valid = RANZERDataset_nih(df_valid_this, 'valid', transform=transforms_valid)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)

    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.valid_batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers)

    log = {}
    roc_auc_max = 0.
    loss_min = 99999
    not_improving = 0

    logger.write(f"{'#'*20} start training fold : {args.fold_id}\n")
    for epoch in range(1, args.n_epochs + 1):
        scheduler_warmup.step(epoch - 1)
        loss_train = train_func(train_loader, model, optimizer, criterion)
        loss_valid, roc_auc = valid_func(valid_loader, model, optimizer, criterion)

        log['loss_train'] = log.get('loss_train', []) + [loss_train]
        log['loss_valid'] = log.get('loss_valid', []) + [loss_valid]
        log['lr'] = log.get('lr', []) + [optimizer.param_groups[0]["lr"]]
        log['roc_auc'] = log.get('roc_auc', []) + [roc_auc]

        content = time.ctime() + ' ' + f'Fold {args.fold_id}, Epoch {epoch}, ' \
                                       f'lr: {optimizer.param_groups[0]["lr"]:.7f}, ' \
                                       f'loss_train: {loss_train:.5f}, ' \
                                       f'loss_valid: {loss_valid:.5f}, ' \
                                       f'roc_auc: {roc_auc:.6f}.\n'
        #print(content)
        logger.write(content)
        not_improving += 1

        if roc_auc > roc_auc_max:
            logger.write(f'roc_auc_max ({roc_auc_max:.6f} --> {roc_auc:.6f}). Saving model ...\n')
            torch.save(model.state_dict(), f'{args.model_dir}/{args.model_name}_fold{args.fold_id}_best_AUC_{roc_auc_max:.4f}.pth')
            roc_auc_max = roc_auc
            not_improving = 0

        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(model.state_dict(), f'{args.model_dir}/{args.model_name}_fold{args.fold_id}_best_loss_{loss_min:.4f}.pth')

        if not_improving == args.early_stop:
            logger.write('Early Stopping...')
            break

    torch.save(model.state_dict(), f'{args.model_dir}/{args.model_name}_fold{args.fold_id}_final.pth')
    with open(f'{args.log_dir}/logs.pickle', 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')
    main()