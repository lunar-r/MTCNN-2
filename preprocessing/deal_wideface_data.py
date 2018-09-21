# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

UPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据地址
RAWPATH = UPATH + "/data/rawdata/"
DEALPATH = UPATH + "/data/dealdata/"

# 用于face、box的数据 wider_face
# 1 标记数据 2 图片
WIDER_TRAIN_TXT = RAWPATH + "wider_face_train.txt"
WIDER_IMG_DIR = RAWPATH + "WIDER_train/images"

#处理后的地址
# 1 txt info
POS_TXT = DEALPATH + "/pos.txt"
PART_TXT = DEALPATH + "/part.txt"
NEG_TXT = DEALPATH + "/neg.txt"
# 2 image
POS_DIR = DEALPATH + "/positive"
PART_DIR = DEALPATH + "/part"
NEG_DIR = DEALPATH + "/negative"


#生成目录
if not os.path.exists(POS_DIR):
    os.mkdir(POS_DIR)
if not os.path.exists(PART_DIR):
    os.mkdir(PART_DIR)
if not os.path.exists(NEG_DIR):
    os.mkdir(NEG_DIR)


from utils import IoU
from deal_utils import getTxtInfo, randomCrop, randomShift

# 随机生成样本量
RandomCrop_NEGNUM = 40
RandomShift_NEGNUM = 5
RandomShift_POS_PART_NUM = 20

def main():

    f_pos = open(POS_TXT, 'w')
    f_neg = open(NEG_TXT, 'w')
    f_part = open(PART_TXT, 'w')

    with open(WIDER_TRAIN_TXT, 'r') as f:
        annotations = f.readlines()

    print("%d pics in total" % len(annotations))

    datalist = getTxtInfo(WIDER_TRAIN_TXT, WIDER_IMG_DIR, with_landmark=False)

    p_idx, n_idx, d_idx, idx, box_idx = 0, 0, 0, 0, 0

    for im_path, boxes in datalist:

        #加载图片
        img = cv2.imread(im_path)
        height, width, _ = img.shape

        idx += 1
        if idx % 100 == 0:
            print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))

        # 随机生成 neg 图片 最小size为40 
        neg_num = 0
        while neg_num < RandomCrop_NEGNUM:
            crop_box = randomCrop(width, height)
            Iou = IoU(crop_box, boxes)
            
            cropped_im = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                save_file = os.path.join(NEG_DIR, "%s.jpg"%n_idx)
                f_neg.write("negative/%s.jpg"%n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        # 对box进行随机漂移
        for box in boxes:

            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # 忽略较小尺寸的box
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # 生成 neg 图片
            for _ in range(RandomShift_NEGNUM):

                shift_box = randomShift(width, height, x1, y1, w, h, normal=False)

                if shift_box is None:
                    continue

                Iou = IoU(shift_box, boxes)
                cropped_im = img[shift_box[1]:shift_box[3], shift_box[0]:shift_box[2], :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
        
                if np.max(Iou) < 0.3:
                    save_file = os.path.join(NEG_DIR, "%s.jpg" % n_idx)
                    f_neg.write("negative/%s.jpg" % n_idx + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1    
                        
            # 生成 pos 图片和 part 图片
            for _ in range(RandomShift_POS_PART_NUM):

                shift_box = randomShift(width, height, x1, y1, w, h, normal=True)

                if shift_box is None:
                    continue

                size = shift_box[2] - shift_box[0]
                offset_x1 = (x1 - shift_box[0]) / float(size)
                offset_y1 = (y1 - shift_box[1]) / float(size)
                offset_x2 = (x2 - shift_box[2]) / float(size)
                offset_y2 = (y2 - shift_box[3]) / float(size)

                cropped_im = img[shift_box[1]:shift_box[3], shift_box[0]:shift_box[2], :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(shift_box, box_) >= 0.65:
                    save_file = os.path.join(POS_DIR, "%s.jpg"%p_idx)
                    f_pos.write("positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(shift_box, box_) >= 0.4:
                    save_file = os.path.join(PART_DIR, "%s.jpg"%d_idx)
                    f_part.write("part/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
    f_pos.close()
    f_neg.close()
    f_part.close()


if __name__ == '__main__':
    main()