# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
import time
import pickle
import argparse
import itertools
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter

import models
from utils import create_logger, accuracy, load_state, get_result, confusion_matrix, \
    regress_accuracy_train, regress_get_result, confusion_matrix_regress, \
    save_val_result, ext_save_output, ext_get_output, focal_get_result
from datasets import SimpleDataset, Prefetcher, PUDataset, ExtDataset
from transforms import *
from loss import PULoss, FocalLoss

# config
pretrained = False # keep this to be false, we'll load weights manually
size = 448 # model input size 
original_size = 1024
prior = 0.5 # positive data ratio in unlabeled dataset.

start_epoch = 0
num_workers = 4
best_loss = float('inf')
best_acc = 0

# arg
parser = argparse.ArgumentParser(description='Chromosome Evaluation Algorithm')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--end_epoch', default=200, type=int, help='epcoh to stop training')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', default='None', help='checkpoint file path')
parser.add_argument('--root', default='/media/voyager/ssd-ext4/chromosome/', help='dataset root path, NOT USED')
parser.add_argument('--train_list', default='/home/voyager/data/chromosome/list/train_74944.txt', help='train dataset list, format: path, gt\n')
parser.add_argument('--val_list', default='/home/voyager/data/chromosome/list/val_1242.txt', help='val dataset list, format: path, gt\n')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--eval_freq', default=1, type=int, help='valide epoch')
parser.add_argument('--exp_path', default='/home/voyager/jpz/chromosome/exp/example', help='experiment path')
parser.add_argument('--model', default='resnet101', help='model type')
parser.add_argument('--num_classes', default=2, type=int, help='for classification')
parser.add_argument('--eval', action='store_true', help='used for evaluation model')
parser.add_argument('--log_file', default='./train.log', help='log file path')
parser.add_argument('--log_freq', type=int, default=50, help='batch log frequency')
parser.add_argument('--optim', type=str, default='SGD', help='Optimizer')
parser.add_argument('--lr_milestones', type=int, nargs='+', default=[20], help='lr scheduler milestones for SGD')
parser.add_argument('--preprocess', type=str, default='autolevel', help='preprocess type')
parser.add_argument('--save_val_result', type=str, default='./val_result_{:04d}_epoch.txt', help='save val dataset result to file, format: gt, pred\n')
parser.add_argument('--save_output', action='store_true', help='save model original output, format: path, gt, pred0, pred1, pred2\n')
parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='cirterion')
parser.add_argument('--dataset', type=str, default='SimpleDataset', help='dataset type, see ./dataset/__init__.py')
parser.add_argument('--train_balance', type=str, default='up', help='train dataset balance')
parser.add_argument('--val_balance', type=str, default='down', help='val dataset balance')
flags = parser.parse_args()

# logger
logger = create_logger('global_logger', flags.log_file)
logger.info('begin')


device = torch.device(flags.device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if flags.preprocess == 'autolevel':
    Auto = Identity()
elif flags.preprocess == 'automask':
    Auto = AutoMask()
else:
    raise NotImplementedError('Preprocess method {} not found'.format(flags.preprocess))

# data
trainTransform = transforms.Compose([
    transforms.Grayscale(), # this process change the channel of data to 1
    Auto, # filter cell
    AutoLevel(0.7, 0.0001), # background balance
    # transforms.Compose([
    #     transforms.CenterCrop(size=original_size),
    #     transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(1., 1.)),
    # ]),
    transforms.Compose([
        Invert(),
        transforms.RandomRotation(45, resample=Image.BILINEAR),
        Invert(),
        transforms.CenterCrop(size=original_size),
        transforms.Resize(size)
    ]),
        
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

valTransform = transforms.Compose([
    transforms.Grayscale(),
    Auto, # background balance
    AutoLevel(0.7, 0.0001),
    transforms.CenterCrop(size=original_size),
    transforms.Resize(size),
    transforms.ToTensor()
])

if flags.dataset == 'SimpleDataset':
    DataSet = SimpleDataset
elif flags.dataset == 'PUDataset':
    DataSet = PUDataset
elif flags.dataset == 'ExtDataset':
    DataSet = ExtDataset
else:
    raise NotImplementedError('No {} dataset type'.format(flags.dataset))

trainSet = DataSet(
    data_list=flags.train_list,
    balance=flags.train_balance,
    transform=trainTransform
)

trainLoader = torch.utils.data.DataLoader(
    dataset=trainSet,
    batch_size=flags.batch_size,
    shuffle=True,
    num_workers=num_workers
)

valSet = DataSet(
    data_list=flags.val_list,
    balance=flags.val_balance,
    transform=valTransform
)

valLoader = torch.utils.data.DataLoader(
    dataset=valSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=num_workers
)

batch_per_epoch = int(len(trainSet)/flags.batch_size)

logger.info('data fi')

# model
if not hasattr(models, flags.model):
    raise Exception('No such model:{}'.format(flags.model))

model = getattr(models, flags.model)(pretrained, num_classes=1 if 'regress' in flags.model else flags.num_classes)
model.to(device)
# optimizer
if not hasattr(optim, flags.optim):
    raise Exception('No such optimizer:{}'.format(flags.optim))
if flags.optim == 'SGD':
    optimizer = optim.SGD(
        model.parameters(),
        lr=flags.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    optimizer.initial_lr = flags.lr
else:
    raise NotImplementedError('Not Support Optimizer {}'.format(flags.optim))
# scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=flags.lr_milestones, gamma=0.1, last_epoch=-1)

logger.info('model fi')

if (flags.resume): # resume checkpoint
    logger.info('Loading {}'.format(flags.checkpoint))
    load_state(flags.checkpoint, model)
    checkpoint = torch.load(flags.checkpoint)
    if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler'])
    best_loss = checkpoint['loss']
    if 'acc' in checkpoint:
        best_acc += checkpoint['acc']
    elif 'perc' in checkpoint:
        best_acc += checkpoint['perc']
    else:
        best_acc = 0
    start_epoch = checkpoint['epoch']

    logger.info('Resume:'
                '\n\tBest Loss:   {:.4f}'
                '\n\tBest Acc:    {:.3f}'
                '\n\tStart Epoch: {}'.format(
            best_loss, best_acc, start_epoch))

if flags.loss == 'PULoss':
    criterion = PULoss(prior, logger)
elif flags.loss == 'FocalLoss':
    criterion = FocalLoss()
else:
    if not hasattr(nn, flags.loss):
        raise NotImplementedError('torch.nn No Such Loss {}'.format(flags.loss))
    criterion = getattr(nn, flags.loss)()

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     'min',
#     factor=0.2,
#     patience=3,
#     verbose=True
# )


writer = SummaryWriter(log_dir=os.path.join(flags.exp_path))

logger.info('CONFIG:'
            '\n\tEval:          {}'
            '\n\tNUM CLASSES:   {}'
            '\n\tInit LR:       {}'
            '\n\tLR Milestones: {}'
            '\n\tEpoch:         {}'
            '\n\tBatchSize:     {}'
            '\n\tBatchPerEpoch: {}'
            '\n\tLoss Func:     {}'
            '\n\tDevice:        {}'
            '\n\tModel:         {}'
            '\n\tLog Freq:      {}'
            '\n\tExp Path:      {}'
            '\n\tResume:        {}'
            '\n\tCheckpoint:    {}'
            '\n\tPreprocess:    {}'
            '\n\tTrain Balance: {}'
            '\n\tVal Balance:   {}'
            '\n\tTrain Data:    {}'
            '\n\tVal Data:      {}'
            '\n\tTransform:     {}'.format(
    flags.eval,flags.num_classes, flags.lr, flags.lr_milestones,
    flags.end_epoch, flags.batch_size,  
    batch_per_epoch, criterion, flags.device, flags.model, 
    flags.log_freq, flags.exp_path,
    flags.resume, flags.checkpoint,
    flags.preprocess, flags.train_balance, flags.val_balance,
    len(trainSet), len(valSet),
    trainTransform
))

# pipeline
def train(epoch):
    global writer
    model.train()
    train_loss = 0
    train_acc = 0
    start = time.time()
    batch_time = time.time()
    prefetcher = Prefetcher(trainLoader) # prefetcher必须每个epoch重新实例化
    
    paths, samples, gts = prefetcher.next()
    batch_index = 0
    batch_loss = 0
    while samples is not None and batch_index<batch_per_epoch:
        

        samples = samples.to(device)
        samples.contiguous()

        gts = gts.to(device)
        if 'regress' in flags.model:
            gts = gts.to(torch.float)
            gts = gts.unsqueeze(1)
        gts.contiguous()

        optimizer.zero_grad()

        if torch.cuda.device_count() > 1:
            output = nn.parallel.data_parallel(model, samples)
        else:
            output = model(samples)
        
        if flags.loss == 'PULoss':
            loss_val, loss_back = criterion(output, gts)
        else:
            loss_val = criterion(output, gts)
            loss_back = loss_val

        loss_back.backward()
        optimizer.step()

        train_loss += loss_val.item()
        batch_loss += loss_val.item()
        batch_index += 1
        if 'regress' in flags.model:
            if 'pu' in flags.model:
                batch_acc = regress_accuracy_train(output, gts, threshold=0.0, ispul=True)[0].item()
            else:
                batch_acc = regress_accuracy_train(output, gts)[0].item()
        else:
            batch_acc = accuracy(output, gts)[0].item()
        train_acc += batch_acc

        paths, samples, gts = prefetcher.next()
        if batch_index % flags.log_freq == 0 and batch_index!=0:
            logger.info('Epoch: {}/{}, Batch: {}/{}, Batch Loss: {:.4f}, Batch Acc: {:.3f}%, LR: {:.5f}, Elapsed Time: {}'.format(
                epoch, flags.end_epoch-1,
                batch_index, batch_per_epoch,
                batch_loss/flags.log_freq,
                batch_acc,
                scheduler.get_lr()[0],
                time.strftime('%H:%M:%S', time.gmtime(time.time()-batch_time))))
            batch_time = time.time()
            writer.add_scalar('batch_loss', batch_loss/flags.log_freq, batch_index+(epoch*batch_per_epoch))
            writer.add_scalar('batch_acc', batch_acc, batch_index+(epoch*batch_per_epoch))
            batch_loss = 0


    end = time.time()
    train_loss /= (batch_index+1)
    train_acc /= (batch_index+1)
    logger.info('Epoch: {}/{}, Epoch Loss: {:.4f}, Epoch Acc:{:.3f}, LR:{:.5f}, Elapsed Time:{}'.format(
        epoch,
        flags.end_epoch - 1,
        train_loss,
        train_acc,
        scheduler.get_lr()[0],
        time.strftime('%H:%M:%S', time.gmtime(end-start))
    ))
    
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_acc', train_acc, epoch)

def val(epoch):
    logger.info('*  VAL')

    with torch.no_grad():
        model.eval()
        val_loss = 0
        val_acc = 0
        result = [] # record result [[gt, pred], ...]
        focal_loss_result = [] # record result for focal_loss [[gt, pred0, pred1, pred2, path], ...]
        output_list = [] # save model's original output [[path, gt, pred0, pred1, pred2], ...]

        for batch_index, (paths, samples, gts) in enumerate(valLoader):
            samples = samples.to(device)
            samples.contiguous()

            gts = gts.to(device) # (batchsize, 1)
            if 'regress' in flags.model:
                gts = gts.to(torch.float)
                gts = gts.unsqueeze(1)
            gts.contiguous()
            if torch.cuda.device_count() > 1:
                output = nn.parallel.data_parallel(model, samples)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            else:
                output = model(samples)
            
            if flags.save_output:
                # save ext model's output to file
                # format path, gt, pred0, pred1, pred2
                output_list.extend(ext_get_output(paths, gts, output))

            if 'regress' in flags.model:
                result.extend(regress_get_result(gts, output))
            else:
                if 'Focal' in flags.loss:
                    focal_loss_result.extend(focal_get_result(paths, gts, output))
                result.extend(get_result(gts, output)) # for confusion matrix
                val_acc += accuracy(output, gts)[0].item() # compute prec

            if flags.loss == 'PULoss':
                loss_val, loss_back = criterion(output, gts)
            else:
                loss_val = criterion(output, gts)
            val_loss += loss_val.item()
            if batch_index % 50 == 0:
                logger.info('Val Batch: {}/{}'.format(batch_index, int(len(valSet)/flags.batch_size)))

        # save checkpoint
        global best_loss
        global best_acc
        val_loss /= len(valLoader)
        val_acc /= (batch_index+1)
        
        if 'regress' in flags.model:
            # auto select threshold for classify and compute confusion matrix
            val_acc += confusion_matrix_regress(result, logger, pul=True if flags.dataset == 'PUDataset' else False) 
        else:
            confusion_matrix(result, logger)

        if flags.save_val_result is not None and flags.save_val_result != '':
            if 'Focal' in flags.loss:
                save_path = flags.save_val_result.split('/')
                save_path[-1] = 'focal_' + save_path[-1]
                save_path = '/'.join(save_path)
                save_val_result(focal_loss_result, epoch, save_path, logger)
            else:
                save_val_result(result, epoch, flags.save_val_result, logger)
        if flags.save_output:
            ext_save_output(output_list, os.path.join(flags.exp_path, 'original_output_epoch_{:04d}.txt'.format(epoch)))
        
        logger.info('Val Loss: {:.4f}, Val Acc: {:.3f}, Best Loss: {:.4f}, Best Acc: {:.3f}'.format(
                val_loss, val_acc, best_loss, best_acc
            ))

        global writer
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        if val_acc > best_acc and not flags.eval:
            logger.info('Saving chekcpoint, best acc: {:.3f}'.format(val_acc))

            state = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': val_acc,
                'net': model.state_dict(),
                'loss': val_loss,
                'epoch': epoch+1, # NOTICE
            }
            path = os.path.join(flags.exp_path, 'checkpoint')
            if not os.path.isdir(path):
                os.mkdir(path)

            save_path = os.path.join(path,'epoch_{:04d}_acc_{:.6f}.pth'.format(
                epoch,
                float(val_acc)
            ))
            
            torch.save(state, save_path)

            best_loss = val_loss
            best_acc = val_acc
            writer.add_scalar('best_loss', best_loss, epoch)

# main loop
if __name__ == '__main__':
    if flags.eval: # just eval model
        val(start_epoch)
    else:
        for epoch in range(start_epoch, flags.end_epoch):
            train(epoch)
            if epoch%flags.eval_freq == 0 or epoch == flags.end_epoch:
                val(epoch)
            scheduler.step() # step lr
    writer.close()
