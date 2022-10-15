from glob import glob
from PIL import Image
import numpy as np


def load_train_data():
    x = []
    y = []
    for img in glob("./data/train/*.png"):
        x.push(np.array(Image.open(img)))
        y.push(int(img.split("_")[1].replace(".png", "") == "split")) # Filename format: <image>_<split/normal>.png
    
    return np.array(x).astype(np.float32), np.array(y).astype(np.float32)