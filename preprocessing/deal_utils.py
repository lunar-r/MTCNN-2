# -*- coding: utf-8 -*-

import numpy as np
import os

def getTxtInfo(txt, image_dir, with_landmark=True):
    """
        Generate data from txt file
        return [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]
            landmark: [(x1, y1), (x2, y2), ...]
    """
    #get dirname
    with open(txt, 'r') as fd:
        lines = fd.readlines()

    result = []
    for line in lines:
        components = line.strip().split(' ')
        img_path = os.path.join(image_dir, components[0]) # file path
        
        if not with_landmark:
            bbox = [float(x) for x in components[1:]]
            #变形(可能存在多个box)
            boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
            result.append((img_path + '.jpg', boxes))
        else:
            bbox = (components[1], components[3], components[2], components[4])
            bbox = [int(x) for x in bbox]
            landmark = np.zeros((5, 2))
            for index in range(0, 5):
                rv = (float(components[5+2*index]), float(components[5+2*index+1]))
                landmark[index] = rv
            result.append((img_path, BBox(bbox), landmark))
    return result

def randomCrop(width, height):
    """ 随机剪裁 """
    """
        args:
            width, height 原图片宽、高 
        return:
            crop box
    """
    size = np.random.randint(12, min(width, height) / 2)
    nx = np.random.randint(0, width - size)
    ny = np.random.randint(0, height - size)
    crop_box = np.array([nx, ny, nx + size, ny + size])

    return crop_box

def randomShift(width, height, x1, y1, w, h, normal=True):
    """ 随机漂移 """
    """
        args:
            width, height 原图片宽、高
            x1, y1 box的左顶点坐标
            w, h box的宽、高
            normal 漂移幅度是否正常
        return:
            shift box
    """
    if normal:
        size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
        delta_x = np.random.randint(-w * 0.2, w * 0.2)
        delta_y = np.random.randint(-h * 0.2, h * 0.2)
        nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
        ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
    else:
        size = np.random.randint(12, min(width, height) / 2)
        delta_x = np.random.randint(max(-size, -x1), w)
        delta_y = np.random.randint(max(-size, -y1), h)
        nx1 = int(max(0, x1 + delta_x))
        ny1 = int(max(0, y1 + delta_y))

    if nx1 + size > width or ny1 + size > height:
        return None
    else:
        shift_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
        return shift_box

def flip(face, landmark):
    """
        flip face
    """
    face_flipped_by_x = cv2.flip(face, 1)
    #mirror
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
    landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth
    return (face_flipped_by_x, landmark_)

def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    #whole image rotate
    #pay attention: 3rd param(col*row)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,(img.shape[1],img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    #crop face 
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return (face, landmark_)


class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]
        
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)
    #offset
    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])
    #absolute position(image (left,top))
    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])
    #landmark: 5*2
    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p
    #change to offset according to bbox
    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p
    #f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    #self.w bounding-box width
    #self.h bounding-box height
    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])