# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import random

# 上一级目录地址
UPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据地址
RAWPATH = UPATH + "/data/rawdata/"
DEALPATH = UPATH + "/data/dealdata/"
FACEPOINT_TXT = RAWPATH + "trainImageList.txt"

net = "PNet"

LANDMARK_DIR = DEALPATH + "/train_%s_landmark_aug" %(net)
LANDMARK_TXT = DEALPATH + "landmark_%s_aug.txt" %(net)

if not os.path.exists(LANDMARK_DIR):
    os.mkdir(LANDMARK_DIR)

from utils import IoU
from deal_utils import BBox, getTxtInfo, randomShift, rotate, flip

argument = True

def main():

    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        raise 'Net type error'

    image_id = 0
    f = open(LANDMARK_TXT,'w')
   
    datalist = getTxtInfo(FACEPOINT_TXT, RAWPATH, with_landmark=True)
    idx = 0

    for im_path, bbox, landmarkGt in datalist:

        F_imgs, F_landmarks = [], []

        img = cv2.imread(im_path)
        height, width, _ = img.shape
        box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
        f_face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
        f_face = cv2.resize(f_face,(size,size))
        landmark = np.zeros((5, 2))

        #标准化，得到的是在方框里的相对位置
        for index, one in enumerate(landmarkGt):
            rv = ((one[0]-box[0])/(box[2]-box[0]), (one[1]-box[1])/(box[3]-box[1]))
            landmark[index] = rv
        
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2))

        if argument:
            idx = idx + 1
            if idx % 100 == 0:
                print(idx, "images done")

            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1        
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # 随机漂移
            for _ in range(10):

                shift_box = randomShift(width, height, x1, y1, w, h, normal=True)

                if shift_box is None:
                    continue

                bbox_size = shift_box[2] - shift_box[0]
                cropped_im = img[shift_box[1]:shift_box[3], shift_box[0]:shift_box[2], :]
                resized_im = cv2.resize(cropped_im, (size, size))

                iou = IoU(shift_box, np.expand_dims(box,0))
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    # 标准化
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0]-shift_box[0])/bbox_size, (one[1]-shift_box[1])/bbox_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1,2)

                    bbox = BBox([shift_box[0], shift_box[1], shift_box[2], shift_box[3]])                    
                    # 镜像                    
                    if random.choice([0,1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        #c*h*w
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))

                    # 逆时针旋转
                    if random.choice([0,1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 5)#逆时针旋转
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                
                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))                
                    
                    # 顺时针旋转
                    if random.choice([0,1]) > 0: 
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -5)#顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10)) 
                    
            F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
            for i in range(len(F_imgs)):

                if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                    continue

                if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                    continue

                cv2.imwrite(os.path.join(LANDMARK_DIR, "%d.jpg" %(image_id)), F_imgs[i])
                landmarks = map(str,list(F_landmarks[i]))
                f.write(os.path.join('train_%s_landmark_aug',"%d.jpg" %(net, image_id))+" -2 "+" ".join(landmarks)+"\n")
                image_id = image_id + 1
    
    f.close()

if __name__ == '__main__':
    main()
    
   
