import numpy as np
import os
import sys



def label2yolobox(labels, info_img, maxsize, lrflip):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    
    #print("labels is ", labels)
    #labels is x1,y1,x2,y2
    label_out = torch.zeros(labels.shape(0), 13, 13, 5)
    
    x1 = labels[:, 1]* 416
    y1 = labels[:, 2]* 416
    
    w = labels[:, 3] * 416
    h = labels[:, 4] * 416
    
    xc = (x1 + w)/2.0
    yc = (y1 + h)/2.0
    
    boxes = [xc, yc, w,h]
    print("boxes",boxes)
    
    ind_box = [int(xc/(416/13)),int(yc/(416/13))]
    
    label_out[:,ind_box[0], ind_box[1], 0] = 1
    label_out[:,ind_box[0], ind_box[1], 1:5] = boxes
    print("label_out",label_out)
    
    
    
    
    
    
    """
    h, w, nh, nw, dx, dy = info_img
    
    x1 = labels[:, 1] / w
    y1 = labels[:, 2] / h
    x2 = (labels[:, 1] + labels[:, 3]) / w
    y2 = (labels[:, 2] + labels[:, 4]) / h
    
    labels[:, 1] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 2] = (((y1 + y2) / 2) * nh + dy) / maxsize
    labels[:, 3] *= nw / w / maxsize
    labels[:, 4] *= nh / h / maxsize
    if lrflip:
        labels[:, 1] = 1 - labels[:, 1]
    """
    return label_out
