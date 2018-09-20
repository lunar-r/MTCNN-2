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

# 随机生成neg样本量
NEGNUM = 40

def main():

    f_pos = open(POS_TXT, 'w')
    f_neg = open(NEG_TXT, 'w')
    f_part = open(PART_TXT, 'w')

    with open(WIDER_TRAIN_TXT, 'r') as f:
        annotations = f.readlines()

    print("%d pics in total" % len(annotations))
    p_idx, n_idx, d_idx, idx, box_idx= 0, 0, 0, 0, 0

    for annotation in annotations:

        annotation = annotation.strip().split(' ')
        #图片路径
        im_path = annotation[0]

        #box 坐标
        bbox = [float(x) for x in annotation[1:]]
        #变形(可能存在多个box)
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

        #加载图片
        img = cv2.imread(os.path.join(WIDER_IMG_DIR, im_path + '.jpg'))
        height, width, channel = img.shape

        idx += 1
        if idx % 100 == 0:
            print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))

        # 随机生成 neg 图片
        neg_num = 0
        while neg_num < NEGNUM:
            #neg_num's size [40,min(width, height) / 2],min_size:40 
            size = np.random.randint(12, min(width, height) / 2)
            #top_left
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            #random crop
            crop_box = np.array([nx, ny, nx + size, ny + size])
            #计算IOU
            Iou = IoU(crop_box, boxes)
            
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            #保存图片，保存记录
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(NEG_DIR, "%s.jpg"%n_idx)
                f_neg.write("negative/%s.jpg"%n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        # 对box进行漂移产生neg图片
        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            #gt's width
            w = x2 - x1 + 1
            #gt's height
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            for i in range(5):
                size = np.random.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)
        
                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
        
                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(NEG_DIR, "%s.jpg" % n_idx)
                    f_neg.write("negative/%s.jpg" % n_idx + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1    
                        
            # 生成 pos 图片和 part 图片
            for i in range(20):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)
                #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue 
                crop_box = np.array([nx1, ny1, nx2, ny2])
                #yu gt de offset
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                #crop
                cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                #resize
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(POS_DIR, "%s.jpg"%p_idx)
                    f_pos.write("positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
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