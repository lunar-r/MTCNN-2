# -*- coding: utf-8 -*-

import os
import random

import cv2
import h5py
import numpy as np

""" 
训练集分为四种:负样本 0，正样本 1，部分样本 -1，关键点样本 -2

人脸分类任务：利用正样本和负样本进行训练

人脸边框回归任务：利用正样本和部分样本进行训练

关键点检测任务：利用关键点样本进行训练

"""
class DataGenerator():

    def __init__(self, pos_dataset_path, neg_dataset_path, part_dataset_path, landmarks_dataset_path, batch_size, im_size):
        self.im_size = im_size
        
        self.pos_file = h5py.File(pos_dataset_path, 'r')
        self.neg_file = h5py.File(neg_dataset_path, 'r')
        self.part_file = h5py.File(part_dataset_path, 'r')
        self.landmark_file = h5py.File(landmarks_dataset_path, 'r')

        self.batch_size = batch_size
        
        pos_radio = 1.0/7
        part_radio = 1.0/7
        neg_radio=3.0/7
        landmark_radio=2.0/7
        
        self.pos_batch_size = int(np.ceil(self.batch_size*pos_radio))
        self.part_batch_size = int(np.ceil(self.batch_size*part_radio))
        self.neg_batch_size = int(np.ceil(self.batch_size*neg_radio))
        self.landmark_batch_size = int(np.ceil(self.batch_size*landmark_radio))
        
        print('pos_batch_size---:',self.pos_batch_size)
        print('part_batch_size---:',self.part_batch_size)
        print('neg_batch_size---:',self.neg_batch_size)
        print('landmark_batch_size---:',self.landmark_batch_size)
        
        self.pos_start = 0
        self.part_start = 0
        self.neg_start = 0
        self.landmark_start = 0

    def _load_pos_dataset(self, end):
        im_batch = self.pos_file['ims'][self.pos_start:end]
        labels_batch = self.pos_file['labels'][self.pos_start:end]
        bboxes_batch = self.pos_file['bbox'][self.pos_start:end]
        landmarks_batch = np.zeros((self.pos_batch_size, 10), np.float32)
        return im_batch, labels_batch, bboxes_batch, landmarks_batch
    
    def _load_neg_dataset(self, end):
        im_batch = self.neg_file['ims'][self.neg_start:end]
        labels_batch = self.neg_file['labels'][self.neg_start:end]
        bboxes_batch = np.zeros((self.neg_batch_size, 4), np.float32)
        landmarks_batch = np.zeros((self.neg_batch_size, 10), np.float32)
        return im_batch, labels_batch, bboxes_batch, landmarks_batch
    
    def _load_part_dataset(self, end):
        im_batch = self.part_file['ims'][self.part_start:end]
        labels_batch = self.part_file['labels'][self.part_start:end]
        bboxes_batch = self.part_file['bbox'][self.part_start:end]
        landmarks_batch = np.zeros((self.part_batch_size, 10), np.float32)
        return im_batch, labels_batch, bboxes_batch, landmarks_batch

    def _load_landmark_dataset(self, end):
        im_batch = self.landmark_file['ims'][self.landmark_start:end]
        label_batch = self.landmark_file['labels'][self.landmark_start:end]
        
        bboxes_batch = np.zeros((self.landmark_batch_size, 4), np.float32)
        bboxes_batch = bboxes_batch + [0, 0, self.im_size - 1, self.im_size - 1]
        landmark_batch = self.landmark_file['landmarks'][self.landmark_start:end]
        
        return im_batch, label_batch, bboxes_batch, landmark_batch,

    def im_show(self, n):
        ns = random.sample(range(0, len(self.landmark_file['ims'][:])), n)
        for i in ns:
            im = self.landmark_file['ims'][i]
            cv2.imshow('{}'.format(i), im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def generate(self):
        while 1:
            
            pos_end = self.pos_start + self.pos_batch_size
            part_end = self.part_start + self.part_batch_size
            neg_end = self.neg_start + self.neg_batch_size
            landmark_end = self.landmark_start + self.landmark_batch_size
            
        
            im_batch1, labels_batch1, bboxes_batch1, landmarks_batch1 = self._load_pos_dataset(pos_end)
            im_batch2, labels_batch2, bboxes_batch2, landmarks_batch2 = self._load_part_dataset(part_end)
            im_batch3, labels_batch3, bboxes_batch3, landmarks_batch3 = self._load_neg_dataset(neg_end)
            im_batch4, labels_batch4, bboxes_batch4, landmarks_batch4 = self._load_landmark_dataset(landmark_end)

            x_batch = np.concatenate((im_batch1, im_batch2, im_batch3, im_batch4), axis=0)
            x_batch = _process_im(x_batch)

            label_batch = np.concatenate((labels_batch1, labels_batch2, labels_batch3, labels_batch4), axis=0)
            label_batch = label_batch.reshape(-1,1)

            #print('============================bboxes_batch1 shape is {}'.format(bboxes_batch1.shape))
            #print('============================bboxes_batch2 shape is {}'.format(bboxes_batch2.shape))
            #print('============================bboxes_batch3 shape is {}'.format(bboxes_batch3.shape))
            #print('============================bboxes_batch4 shape is {}'.format(bboxes_batch4.shape))
            bbox_batch = np.concatenate((bboxes_batch1, bboxes_batch2, bboxes_batch3, bboxes_batch4), axis=0)

            landmark_batch = np.concatenate((landmarks_batch1, landmarks_batch2, landmarks_batch3, landmarks_batch4), axis=0)

            #print('============================label_batch shape is {}'.format(label_batch.shape))
            #print('============================bbox_batch shape is {}'.format(bbox_batch.shape))
            #print('============================landmark_batch shape is {}'.format(landmark_batch.shape))
            label_batch_shape = label_batch.shape
            bbox_batch_shape = bbox_batch.shape
            landmark_batch_shape = landmark_batch.shape
            
            # 如果数量不一致，说明某个样本量不够了，则从头开始取数据
            if label_batch_shape[0] != bbox_batch_shape[0] or bbox_batch_shape[0] != landmark_batch_shape[0]:
                new_start = random.randrange(1, min(self.pos_batch_size,self.part_batch_size,self.neg_batch_size, self.landmark_batch_size))

                this_batch_size = max(label_batch_shape[0], bbox_batch_shape[0], landmark_batch_shape[0])
                if label_batch_shape[0] != this_batch_size:
                    self.pos_start = int(new_start)
                    self.part_start = int(new_start)
                    self.neg_start = int(new_start)
                if bbox_batch_shape[0] != this_batch_size:
                    self.pos_start = int(new_start)
                    self.part_start = int(new_start)
                if landmark_batch_shape[0] != this_batch_size:
                    self.landmark_start = int(new_start)
                continue
            
            y_batch = np.concatenate((label_batch, bbox_batch, landmark_batch), axis=1)

            yield x_batch, y_batch

            self.pos_start = pos_end
            self.part_start = part_end
            self.neg_start = neg_end
            self.landmark_start = landmark_end

    def steps_per_epoch(self):
        pos_len = len(self.pos_file['ims'][:])
        neg_len = len(self.neg_file['ims'][:])
        part_len = len(self.part_file['ims'][:])
        landmark_len = len(self.landmark_file['ims'][:])
        
        pos_total_step = int(pos_len / self.pos_batch_size)
        neg_total_step = int(neg_len / self.neg_batch_size)
        part_total_step = int(part_len / self.part_batch_size)
        landmark_total_step = int(landmark_len / self.landmark_batch_size)
        
        total_len = min(pos_total_step, neg_total_step, part_total_step, landmark_total_step)
        
        print('pos_total_step, neg_total_step, part_total_step, landmark_total_step----:',pos_total_step, neg_total_step, part_total_step, landmark_total_step)
        return total_len - 12

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pos_file.close()
        self.neg_file.close()
        self.part_file.close()
        self.landmark_file.close()


def _process_im(im):
    return (im.astype(np.float32) - 127.5) / 128
