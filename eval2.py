## 라이브러리 추가하기
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from model import UNet
from dataset import *
from util import *

## 트레이닝 파라메터 설정하기
lr = 1e-5
batch_size = 1
# batch_size = 4

num_epoch = 100

data_dir = './datasets5'
ckpt_dir = './checkpoint6'
log_dir = './log11'
result_dir = './results12'

dice_dir = '/Dice2'

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



## 네트워크 학습하기
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
transform = transforms.Compose([ToTensor()])

dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 variables 설정하기
num_data_test = len(dataset_test)

num_batch_test = np.ceil(num_data_test / batch_size)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard를 사용하기 위한 SummaryWriter 설정
writer_test = SummaryWriter(log_dir=os.path.join(log_dir, 'test'))

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # dict_model = torch.load('/home/park/PycharmProjects/002-pytorch-unet/checkpoint5/model_epoch5.pth')
    dict_model = torch.load(ckpt_dir + '/model_epoch5.pth')

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    # epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    epoch = int(ckpt_lst[-2].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

## 네트워크 학습시키기
st_epoch = 0

net = nn.DataParallel(net, device_ids=[0,1])
net.cuda()

net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
total_dice_score = 0
nan_counter = 0
for_loop_counter = 0

whole_loop_counter = 0

id=0

dice = open(result_dir+dice_dir, 'w')
dice.write("Dice score Document \n")

with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        # forward pass
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

        whole_loop_counter += 1


        if tk_pd.sum() + tk_gt.sum() > 0:
            tk_dict = 2*np.logical_and(tk_pd, tk_gt).sum()/(tk_pd.sum() + tk_gt.sum())
            print("Current Dice score : %.4f" %tk_dict)
            total_dice_score += tk_dict
            for_loop_counter += 1
            writer_test.add_scalar('Dice_score', tk_dict, for_loop_counter)
            dice.write("[%4d] DICE SCORE : %.4f\n" %(whole_loop_counter ,tk_dict))

        else:
            nan_counter += 1
            print("tk_pd.sum() + tk_gt.sum() has 0 value!")
            dice.write("[%4d] DICE SCORE : NAN\n" %whole_loop_counter)

        # 손실함수 계산하기
        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
              (batch, num_batch_test, np.mean(loss_arr)))

        # Tensorboard 저장하기
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        # input = fn_tonumpy(input)
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            # id = num_batch_test * (batch - 1) + j
            id += 1

            plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

            np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
      (batch, num_batch_test, np.mean(loss_arr)))
print("AVERAGE DICE SCORE : %04f" % ((total_dice_score+nan_counter)/(num_batch_test)))
dice.write("AVERAGE DICE SCORE : %04f" % ((total_dice_score+nan_counter)/(num_batch_test)))
dice.close()






















