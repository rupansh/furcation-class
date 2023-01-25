from glob import glob
import cv2
import numpy as np
from util import *
import json

resize_sz = (320, 320)

def load_data(base_path, n = None, rs = resize_sz):
    with open(f"{base_path}/labels.json") as labf:
        labs = json.load(labf)

    x = []
    y = []
    im_cnt = 0
    for lab in labs:
        impath = lab["file_upload"]
        im = cv2.imread(f"{base_path}/{impath}")
        if rs:
            im = cv2.resize(im, rs)
        im = np.asarray(im).astype(np.float32)
        h, w, _ = im.shape
        x.append(im)
        truth = lab["annotations"][0]["result"][0]["value"]
        #y.append(np.asarray([box["x"]/w, box["y"]/h, box["width"]/w, box["height"]/h]))
        categ = truth["rectanglelabels"][0][6:]
        grade = int(categ) if categ.isnumeric() else 0
        out = [1 if i == grade else 0 for i in range(1, 4)]
        y.append(np.asarray(out).astype(np.float32))

        im_cnt += 1
        if n and im_cnt == n:
            break 
    
    return np.array(x), np.array(y)
    

def load_train_data(n = None, rs = resize_sz):
    return load_data("./data/train/data-full", n, rs)

def load_test_data(n = None, rs = resize_sz):
    return load_data("./data/test", n, rs)
