# -*- coding:utf-8 -*-
'''
train an unet with the updated labels
'''
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

import argparse, sys
import numpy as np
import datetime
from unet import UNet
from data_utils import get_imdb_data, get_original_data
from loss import *

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--log_dir', type = str, help = 'dir to save result txt files', default = 'logs_75_synthetic/')
parser.add_argument('--test_log_dir', type = str, help = 'dir to save result txt files', default = '/test_1it/')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = 0.1)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--dataset', type = str, help = 'the training set', default = '\\model_noise_75_1_8\\1it\\')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=10)
parser.add_argument('--num_gpu', type=int, default=1)
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters

learning_rate = args.lr 

# load dataset    
train_data, test_data = get_imdb_data(args.num_classes )
if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule   
save_dir = args.result_dir +'/' +args.dataset

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str='seg_unet'

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


# Train the Model
def train(train_loader,epoch, model1, optimizer1):
    print('Training %s...' % model_str)
    
    train_total, train_correct = 0,0
    
    for i_batch, sample_batched in enumerate(train_loader):

        images = Variable(sample_batched[0]).cuda()
        labels = Variable(sample_batched[1]).cuda()
        w = Variable(sample_batched[2]).cuda()
        
        output1 = model1(images)
        loss_func = CombinedLoss()
        loss1 = loss_func(input = output1.float(), 
            target = labels.float(), 
            weight =  w.float(), 
            num_classes = args.num_classes)
                                                                                 
                                                                                 
        train_total+=1
        train_correct+=loss1[1].item()
        loss_1 = loss1[0]
        loss_1 = loss_1.cuda()
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        
    train_acc1 = float(train_correct)/float(train_total)
    return train_acc1

# Evaluate the Model
def evaluate(test_loader, model1):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1, total1, dice1, accuray1, loss11 = 0, 0, 0, 0, 0

    for i_batch, sample_batched in enumerate(test_loader):
        images =  Variable(sample_batched[0]).cuda()
        labels = Variable(sample_batched[1])
        w = Variable(sample_batched[2])
        output1 = model1(images)
        loss_func = CombinedLoss()
        loss1 = loss_func(input = output1.float(), target = labels.float(), 
            weight =  w.float(), num_classes = args.num_classes)

        total1 += 1        
        dice1 += loss1[1].item()
        accuray1 += loss1[3].item()
        loss11 += loss1[0].item()

    acc1 = 100*float(accuray1)/float(total1)
    dice1 = 100*float(dice1)/float(total1)
    loss_tl1 = float(loss11)/float(total1)
    return acc1, dice1, loss_tl1


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=args.batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('########################################building model...###########################################')

    cnn1 = UNet(n_channels=1, n_classes=args.num_classes)
    cnn1.cuda()
    #cnn1 = nn.DataParallel(cnn1, range(args.num_gpu))
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=args.lr)

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 \n')

    epoch=0
    train_acc1=0
    # evaluate models with random weights
    test_acc1, test_dice1, _=evaluate(test_loader, cnn1)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %%' \
        % (epoch+1, args.n_epoch, len(test_data), test_acc1))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) + " " + "\n")

    # training
    writer1 = SummaryWriter()
   
    print('#######################TRAINING ON %s training images##########################################' %(len(train_data)))
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)

        train_acc1 =train(train_loader, epoch, cnn1, optimizer1)
        # evaluate models
        test_acc1, test_dice1, loss_tl1 = evaluate(test_loader, cnn1)
        writer1.add_scalar('model1/accuray1', test_acc1, epoch)
        writer1.add_scalar('model1/dice1', test_dice1, epoch)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% , dice: Model1 %.4f %%' \
            % (epoch+1, args.n_epoch, len(test_data), test_acc1,  test_dice1))
        torch.save(cnn1.state_dict(), os.getcwd() + '\\modelunet\\seg_module.model')

        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(test_acc1) + " " + "\n")
    
    writer1.close()

if __name__=='__main__':
    main()