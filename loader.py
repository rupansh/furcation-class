from glob import glob
import cv2
import numpy as np
from util import *

resize_sz = (320, 320)

def rotate_bb_90(bndbox, original_w, original_h):
    l, t, r, b = yolo_voc(bndbox, (original_w, original_h))
    nl = t
    nt = original_w - r
    nr = b
    nb = original_w - l
 
    return voc_yolo([nl, nt, nr, nb], (original_h, original_w))

def load_data(base_path, n = None, rs = resize_sz):
    x = []
    y = []
    im_cnt = 0
    for img in glob(f"{base_path}/images/*.jpg"):
        im = cv2.imread(img)

        h, w, _ = im.shape
        rot = False
        if w > h:
            rot = True
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if rs:
            im = cv2.resize(im, rs)

        im = np.asarray(im).astype(np.float32)
        x.append(im)

        label_name = img.split("/").pop()[:-4]
        with open(f"{base_path}/labels/{label_name}.txt", "r") as lab_f:
            lab_s = lab_f.readline().strip("\n")
            lab = [float(cord) for cord in lab_s.split()[1:]]

            if rot:
                lab = rotate_bb_90(lab, w, h)
            
            y.append(np.array(lab).astype(np.float32))

        im_cnt += 1
        if n and im_cnt == n:
            break 
    
    return np.array(x), np.array(y)

def load_train_data(n = None, rs = resize_sz):
    return load_data("./data/train", n, rs)

def load_test_data(n = None, rs = resize_sz):
    return load_data("./data/test", n, rs)
