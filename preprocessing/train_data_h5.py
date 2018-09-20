# -*- coding: utf-8 -*-

import os
import random
import cv2
import numpy as np

from utils import resize, save_dict_to_hdf5

filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dealdata = filepath + '/data/dealdata/'
traindata = filepath + '/data/traindata/'

net = "ONet"

posh5 = traindata + 'pos_shuffle_%s.h5' %(net)
negh5 = traindata + 'neg_shuffle_%s.h5' %(net)
parth5 = traindata + 'part_shuffle_%s.h5' %(net)
landmarksh5 = traindata + 'landmarks_shuffle_%s.h5' %(net)

def main():

    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        print('Net type error')
        return

    net_data_dir = os.path.join(dealdata, '12')
    landmark_data_dir = os.path.join(dealdata, str(size))

    with open('%s/pos_%s.txt' % (net_data_dir, '12'), 'r') as f:
        pos = f.readlines()
    with open('%s/neg_%s.txt' % (net_data_dir, '12'), 'r') as f:
        neg = f.readlines()
    with open('%s/part_%s.txt' % (net_data_dir, '12'), 'r') as f:
        part = f.readlines()
    with open('%s/landmark_%s_aug.txt' % (landmark_data_dir, size), 'r') as f:
        landmark_anno = f.readlines()

    create_pos_dataset(pos, size, posh5)
    create_neg_dataset(neg, size, negh5)
    create_part_dataset(part, size, parth5)
    create_landmark_dataset(landmark_anno, size, landmarksh5)

def create_pos_dataset(pos, target_size, out_dir):

    ims = []
    landmarks = []
    labels = []

    for line in pos:
        words = line.strip().split()
        if '.jpg' in words[0]:
            image_file_name = dealdata + '/' + words[0]
        else:    
            image_file_name = dealdata + '/' + words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(int(words[1]))

        landmark = words[2:6]
        landmark = list(map(float, landmark))
        landmark = np.array(landmark, dtype='float32')
        landmarks.append(landmark)
        if len(ims)%500 == 0 :
            print('pos data doing, total: {}'.format(len(ims)))

    landmark_data = list(zip(labels, ims, landmarks))
    random.shuffle(landmark_data)
    labels, ims, landmarks = zip(*landmark_data)

    save_dict_to_hdf5({'labels': labels, 'ims': ims, 'bbox': landmarks}, out_dir)

    print('pos data done, total: {}'.format(len(ims)))    
    
    
def create_neg_dataset(neg, target_size, out_dir):
    ims = []
    landmarks = []
    labels = []
    for line in neg:

        words = line.strip().split()
        if '.jpg' in words[0]:
            image_file_name = dealdata + '/' + words[0]
        else:    
            image_file_name = dealdata + '/' + words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(int(words[1]))
        
        landmark = words[2:6]
        landmark = list(map(float, landmark))
        landmark = np.array(landmark, dtype='float32')
        landmarks.append(landmark)
        if len(ims)%500 == 0 :
            print('neg data doing, total: {}'.format(len(ims)))

    landmark_data = list(zip(labels, ims, landmarks))
    random.shuffle(landmark_data)
    labels, ims, landmarks = zip(*landmark_data)

    save_dict_to_hdf5({'labels': labels, 'ims': ims, 'bbox': landmarks}, out_dir)

    print('neg data done, total: {}'.format(len(ims)))    
    
    
def create_part_dataset(part, target_size, out_dir):
    ims = []
    landmarks = []
    labels = []
    for line in part:

        words = line.strip().split()
        if '.jpg' in words[0]:
            image_file_name = dealdata + '/' + words[0]
        else:    
            image_file_name = dealdata + '/' + words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(int(words[1]))

        landmark = words[2:6]
        landmark = list(map(float, landmark))
        landmark = np.array(landmark, dtype='float32')
        landmarks.append(landmark)
        if len(ims)%500 == 0 :
            print('part data doing, total: {}'.format(len(ims)))
            
    landmark_data = list(zip(labels, ims, landmarks))
    random.shuffle(landmark_data)
    labels, ims, landmarks = zip(*landmark_data)

    save_dict_to_hdf5({'labels': labels, 'ims': ims, 'bbox': landmarks}, out_dir)

    print('part data done, total: {}'.format(len(ims)))


def create_landmark_dataset(landmark_anno, target_size, out_dir):
    ims = []
    landmarks = []
    labels = []
    for line in landmark_anno:

        words = line.strip().split()
        if '.jpg' in words[0]:
            image_file_name = dealdata + '/' + words[0]
        else:    
            image_file_name = dealdata + '/' + words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(int(words[1]))

        landmark = words[2:12]
        landmark = list(map(float, landmark))
        landmark = np.array(landmark, dtype='float32')
        landmarks.append(landmark)
        if len(ims)%500 == 0 :
            print('landmarks data doing, total: {}'.format(len(ims)))

    landmark_data = list(zip(labels, ims, landmarks))
    random.shuffle(landmark_data)
    labels, ims, landmarks = zip(*landmark_data)

    save_dict_to_hdf5({'labels': labels, 'ims': ims, 'landmarks': landmarks}, out_dir)

    print('landmarks data done, total: {}'.format(len(ims)))

if __name__ == '__main__':
    main()
    
