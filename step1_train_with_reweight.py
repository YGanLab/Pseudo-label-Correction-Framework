# -*- coding:utf-8 -*-
'''
train the initial network with the multi-level reweighting strategy
'''
import os, pdb
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
#from data.cifar import CIFAR10, CIFAR100
import argparse, sys
import numpy as np
import numpy.random as npr
import datetime

from unet import UNet
from data_utils import get_imdb_data, get_original_data
from loss import *
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
# train on 75% noise

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--start_reweight', type=int, default=30)
parser.add_argument('--n_epoch', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_classes', type=int, default=2)
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

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [args.lr] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * args.lr 
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

def shuffle_data(data):
    # shuffle the training data
    rnd_indices = list(range(data[3]))
    npr.shuffle(rnd_indices)
    im = data[0]
    lb = data[1]
    data_weight = data[2]

    im = im[rnd_indices]
    lb = lb[rnd_indices]
    data_weight = data_weight[rnd_indices]

    return [im, lb, data_weight, data[3]]
        
   
save_dir = args.result_dir +'/' +args.dataset

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str= 'seg_module'

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


# Train the Model
def train(train_data, epoch, model1, optimizer1):
    print('Training %s...' % model_str)
    
    train_images, train_label = train_data[0], train_data[1]
    train_w, num_train = train_data[2], train_data[3]
    train_total, dice1, accuray1, tr_loss = 0, 0, 0, 0

    for start, end in zip(range(0, num_train, args.batch_size), range(args.batch_size, num_train, args.batch_size)):

        images = Variable(train_images[start:end]).cuda()
        labels = Variable(train_label[start:end]).cuda()
        w = Variable(train_w[start:end]).cuda()
        
        logits1 = model1(images)    
        ce_func = nn.CrossEntropyLoss(reduce = False) # # loss for pixel level reweighting
        loss_func = CombinedLoss() # loss for image level reweighting

        # loss for image level reweighting
        loss1 = loss_func(input = logits1.float(), target = labels.float(), weight =  w.float(), num_classes = args.num_classes)

        # start to change pixel level weights
        if epoch > args.start_reweight: # 30
            celoss = ce_func(logits1.float(), labels.type(torch.LongTensor).cuda())
            weight_new = torch.ones(celoss.shape)
            celoss_nmpy = celoss.detach().cpu().numpy()
            qval = np.quantile(celoss_nmpy.reshape(-1), 0.98) # 0.95 18 all
            weight_new[celoss>qval] = 0
            weight_new = weight_new[:,None,:,:]
            weight_new = torch.cat((weight_new, weight_new), 1)
            w = Variable(weight_new).cuda()
        
        loss_1 = loss_coteaching(output1 = logits1.float(), labels = labels.float(), w =  w.float(), loss_1 = loss1[2], 
                                    num_classes = args.num_classes,
                                    forget_rate= args.forget_rate) 

        # y, dice loss+ cross rntropy; y3: dice coefficient; y4: loss used for sorting                                                                                                                            
        
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        train_total+=1
        dice1 += loss1[1].item()
        accuray1 += loss1[3].item()
        tr_loss += loss1[0].item()
        
    tr_acc =100*float(accuray1)/float(train_total)
    tr_dice = 100*float(dice1)/float(train_total)
    tr_loss = float(tr_loss)/float(train_total)
    
    return [tr_acc, tr_dice, tr_loss], [train_images, train_label, train_w, num_train]

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
        loss_func=CombinedLoss()
        loss1 = loss_func(input = output1.float(), target = labels.float(), weight =  w.float(), num_classes = args.num_classes)
        total1 += 1

        #correct1 += (pred1.cpu() == labels).sum()
        dice1 += loss1[1].item()
        accuray1 += loss1[3].item()
        loss11 += loss1[0].item()

 
    acc1 = 100*float(accuray1)/float(total1)

    dice1 = 100*float(dice1)/float(total1)

    loss_tl1 = float(loss11)/float(total1)

    return acc1, dice1, loss_tl1, 


def main():

    # Data Loader (Input Pipeline)
    print('loading dataset...')

    train_images, train_labels, weight, num_train = get_original_data('Train', args.num_classes)
    train_labels = np.squeeze(train_labels)
    train_data = [torch.from_numpy(train_images), torch.from_numpy(train_labels), torch.from_numpy(weight), num_train]

    # load test dataset

    _, test_data = get_imdb_data(args.num_classes)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=args.batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('######################################## building model... ###########################################')
    
    cnn1 = UNet(n_channels=1, n_classes=args.num_classes)
    
    RESUME = False
    if RESUME:
    	cnn1.load_state_dict(torch.load(os.getcwd() + '\\model_gt\\seg_module.model'))
    cnn1.cuda()
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=args.lr)


    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n')

    epoch=0
    train_acc1=0

    # evaluate models with random weights
    test_acc1, test_dice1, _=evaluate(test_loader, cnn1)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% ' \
        % (epoch+1, args.n_epoch, len(test_data), test_acc1))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(test_acc1) + "\n")

    # training
    test_writer = SummaryWriter(logdir=args.log_dir + args.test_log_dir)
    tr_writer = SummaryWriter(logdir=args.log_dir + '\\train\\')

    print('####################### TRAINING ON %s training images ##########################################' %(len(train_data)))
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)

        train_re, train_data=train(train_data, epoch, cnn1, optimizer1)
        # evaluate models
        train_data = shuffle_data(train_data)

        tr_writer.add_scalar('accuray', train_re[0], epoch)
        tr_writer.add_scalar('dice', train_re[1], epoch)
        tr_writer.add_scalar('loss', train_re[2], epoch)

        if epoch % 1 == 0:
            test_acc1, test_dice1, loss_tl1=evaluate(test_loader, cnn1)
            test_writer.add_scalar('accuray', test_acc1, epoch)
            test_writer.add_scalar('dice', test_dice1, epoch)
            test_writer.add_scalar('loss', loss_tl1, epoch)


        print('Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%, dice: %.4f %%, train_acc: %.4f %%, train_dice: %.4f %%' \
            % (epoch+1, args.n_epoch, len(test_data), test_acc1, test_dice1, train_re[0], train_re[1]))


        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(test_acc1) + " " + "\n")
    torch.save(cnn1.state_dict(), os.getcwd() + '\\models\\' + args.dataset + '\\seg_module.model')

if __name__=='__main__':
    main()
