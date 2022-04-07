## 라이브러리 추가하기
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets
from model import UNet
from dataset import *
from util import *

## 트레이닝에 필요한 하이퍼파라미터 설정

lr = 1e-5
# lr0 = 1e-5
batch_size = 8
num_epoch = 5

info_dir = '/checkpoint13'

data_dir = './datasets5'
ckpt_dir = './checkpoint13'
log_dir = './log13'

## 네트워크 정보 저장
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
f = open(ckpt_dir+info_dir, 'w')
f.write("learning_rate : %f \n" %lr)
f.write("batch_size : %d \n" %batch_size)
f.write("num_epoch : %d \n" %num_epoch)
f.write("Dataset number : %s \n" %data_dir)
f.write("Transformer \n")
f.write("Normalization : False\n")
f.write("Randomflip : False\n")
f.write("ToTensor : True \n")
f.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 네트워크 학습하기
# transform = transforms.Compose([RandomFlip(), ToTensor()])
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
transform = transforms.Compose([ToTensor()])


dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)
# fn_loss = nn.BCELoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)
# optim = torch.optim.SGD(net.parameters(), lr=lr)


## 그밖에 부수적인 variables 설정하기
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tenasorboard를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


## 네트워크 학습시키기
st_epoch = 0
# net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

start = time.time()

net = nn.DataParallel(net, device_ids=[0,1])
net.cuda()

total_dice_score = 0
for_loop_counter = 0
nan_counter = 0

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []
    start = time.time()
    # lr = lr0 * np.exp(-(epoch-1)/5)

    for batch, data in enumerate(loader_train, 1):
        #forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        output2 = output.cpu()
        output2 = output2.detach().numpy()
        prediction = np.round(output2)

        label2 = label.cpu()
        ground_truth = label2.numpy()

        tk_pd = np.greater(prediction, 0)
        tk_gt = np.greater(ground_truth, 0)
        tk_dict = 2 * np.logical_and(tk_pd, tk_gt).sum() / (tk_pd.sum() + tk_gt.sum())

        if tk_pd.sum() + tk_gt.sum() > 0:
            tk_dict = 2*np.logical_and(tk_pd, tk_gt).sum()/(tk_pd.sum() + tk_gt.sum())
            print("Current Dice score : %.4f" %tk_dict)
            total_dice_score += tk_dict
            for_loop_counter += 1
            writer_train.add_scalar('Dice_score', tk_dict, for_loop_counter)

        else:
            nan_counter += 1
            print("tk_pd.sum() + tk_gt.sum() has 0 value!")


        #backward pass
        optim.zero_grad()

        loss = fn_loss(output, label)
        loss.requires_grad_(True)
        loss.backward()

        optim.step()

        # 손실함수 계산
        loss_arr += [loss.item()]
        end = time.time()
        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
              (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
        if batch % 50 == 0:
            print("TRAIN TIME for 50 Batch is : %d" %(end-start))
            start = time.time()

        # Tensorboard 저장하기
        label = fn_tonumpy(label)
        # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        input = fn_tonumpy(input)
        output = fn_tonumpy(fn_class(output))

        writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')


    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

            #Tensorborad 저장하기
            label = fn_tonumpy(label)
            # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            input = fn_tonumpy(input)
            output = fn_tonumpy(fn_class(output))

            writer_val.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

writer_train.close()
writer_val.close()


