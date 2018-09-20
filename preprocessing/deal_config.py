# -*- coding: utf-8 -*-

import os

# 上一级目录地址
UPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据地址
RAWPATH = UPATH + "/data/rawdata/"
DEALPATH = UPATH + "/data/dealdata/"

# 用于face、box的数据 wider_face
# 1 标记数据 2 图片
WIDER_TRAIN_TXT = RAWPATH + "wider_face_train.txt"
WIDER_IMG_DIR = RAWPATH + "WIDER_train/images"

# 用于landmark的数据 FacePoint
FACEPOINT_TXT = RAWPATH + "trainImageList.txt"
FACEPOINT_IMG_DIR = "" # txt里面包含了目录信息

#处理后的地址
# 1 txt info
POS_TXT = DEALPATH + "/pos.txt"
PART_TXT = DEALPATH + "/part.txt"
NEG_TXT = DEALPATH + "/neg.txt"
LANDMARK_12_TXT = DEALPATH + "/landmark_12.txt"
LANDMARK_24_TXT = DEALPATH + "/landmark_24.txt"
LANDMARK_48_TXT = DEALPATH + "/landmark_48.txt"

# 2 image
POS_DIR = DEALPATH + "/positive"
PART_DIR = DEALPATH + "/part"
NEG_DIR = DEALPATH + "/negative"
LANDMARK_12_DIR = DEALPATH + "/landmark_12"
LANDMARK_24_DIR = DEALPATH + "/landmark_24"
LANDMARK_48_DIR = DEALPATH + "/landmark_48"
