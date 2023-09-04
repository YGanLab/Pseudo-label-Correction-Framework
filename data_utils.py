"""Data utility functions."""

import numpy as np
import torch
import torch.utils.data as data
import h5py
import os
from scipy import ndimage, misc
import scipy.io as scio
import matplotlib.pyplot as plt
import math
import re

def get_info(filenames, ext, root):
    images = []
    for filename in filenames :
        filepath = os.path.join(root,filename)
        if ext == '.npy':
            image = np.load(filepath)
        elif ext == '.JPG' or ext == '.tif' or ext =='.png':
            image = ndimage.imread(filepath)
        images.append(image)
    return images

def get_data(directory,ext):
    from os import listdir
    from os.path import isfile, join
    
    root_path = ""
    filenames = [f for f in listdir(directory) if isfile(join(directory, f)) and f != '.DS_Store']
    filenames = sorted(filenames)
    return filenames, get_info(filenames, ext, directory)

def get_original_data(mode, num_classes):
    cwd = os.getcwd() + '\\medical_images\\'
    if mode == 'Train':
        _, test_labels_list = get_data(cwd+'\\labels\\Train\\Train\\segmented_ids','.npy')
        filenames, test_raw_images = get_data(os.getcwd() +'\\medical_images\\oct_images\\Train\\Train','.png')
        
    elif mode == 'Test':
        filenames, test_raw_images = get_data(os.getcwd() +\
                                              '\\medical_images\\oct_images\\Test\\Test','.png')
        _, test_labels_list = get_data(cwd+
                                        '\\labels\\Test\\Test\\segmented_ids','.npy')
    test_raw_images=np.array(test_raw_images)
    test_raw_images = test_raw_images.reshape(test_raw_images.shape[0],1,256,128)
    test_raw_labels = np.zeros((len(test_raw_images), 256,128,num_classes))
    
    val = 1
    for i in range(len(test_labels_list)) :
        for lab in range(0,num_classes):
            test_raw_labels[i,:,:, lab] = test_labels_list[i] == lab

    test_indices = np.random.choice(len(test_raw_images),len(test_raw_images),replace = False)
    test_images = []
    test_labels = []

    for i in test_indices:
        test_images.append(test_raw_images[i])
        test_labels.append(test_raw_labels[i])
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    test_images = test_images.astype('float32')/255.0
    test_labels = test_labels.astype('float32')
    test_images = test_images.reshape(test_images.shape[0],1,256,128)
    test_labels = np.argmax(test_labels, -1)
    
    test_labels = test_labels.reshape(test_images.shape[0],1,256,128)
    weights = np.ones(test_images.shape)
    weights = np.tile(weights, [1, num_classes, 1, 1])
    return test_images, test_labels, weights, len(filenames)



class ImdbData(data.Dataset):
    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        weight = self.w[index]

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)
        return img, label, weight

    def __len__(self):
        return len(self.y)


def get_imdb_data(num_classes):
    
    NumClass = num_classes

    Tr_Dat, Tr_Label,_,_ = get_original_data('Train', num_classes)
    Te_Dat, Te_Label,_,_ = get_original_data('Test', num_classes)
    Tr_Label = np.squeeze(Tr_Label)  # Index from [0-(NumClass-1)]
    Tr_weights = np.ones(Tr_Dat.shape)
    Tr_weights = np.tile(Tr_weights, [1, NumClass, 1, 1])
    Te_Label = np.squeeze(Te_Label)  # Index from [0-(NumClass-1)]
    Te_weights = np.ones(Te_Dat.shape)
    Te_weights = np.tile(Te_weights, [1, NumClass, 1, 1])   

    return (ImdbData(Tr_Dat, Tr_Label, Tr_weights),
            ImdbData(Te_Dat, Te_Label, Te_weights))
