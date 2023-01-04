def yolo_voc(cord, dim):
    x, y, w, h = cord
    l = int((x - w / 2) * dim[0])
    r = int((x + w / 2) * dim[0])
    t = int((y - h / 2) * dim[1])
    b = int((y + h / 2) * dim[1])

    return (l, t, r, b)

def voc_yolo(box, dim):
    (image_w, image_h) = dim
    x1, y1, x2, y2 = box

    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]
