import os
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2
#from pycocotools.coco import COCO


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
    
    label_out = np.zeros((1, 13, 13, 5))
    
    
    x1 = labels[:, 1]* 416
    y1 = labels[:, 2]* 416
    
    w = labels[:, 3] * 416
    h = labels[:, 4] * 416
    
    xc = x1 + w/2.0
    yc = y1 + h/2.0
    
    boxes = [xc, yc, w,h]
    #print("boxes",boxes)
    
    ind_box = [(xc/416.0*13.0).astype(np.int),(yc/416.0*13.0).astype(np.int)]
    #print("##########################: ",ind_box[0], ind_box[1])
    
    label_out[:,ind_box[1], ind_box[0], 0] = 1
    label_out[:,ind_box[1], ind_box[0], 1] = boxes[0]
    label_out[:,ind_box[1], ind_box[0], 2] = boxes[1]
    label_out[:,ind_box[1], ind_box[0], 3] = boxes[2]
    label_out[:,ind_box[1], ind_box[0], 4] = boxes[3]
    
    
    #print("label_out",label_out)
    
    
    
    
    
    
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

def preprocess(img, imgsize, jitter, random_placing=False):
    """
    Image preprocess for yolo input
    Pad the shorter side of the image and resize to (imgsize, imgsize)
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        imgsize (int): target image size after pre-processing
        jitter (float): amplitude of jitter for resizing
        random_placing (bool): if True, place the image at random position

    Returns:
        img (numpy.ndarray): input image whose shape is :math:`(C, imgsize, imgsize)`.
            Values range from 0 to 1.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    """
    h, w, _ = img.shape
    img = img[:, :, ::-1]
    assert img is not None

    if True:
        new_ar = w / h

    if new_ar < 1:
        nh = imgsize
        nw = nh * new_ar
    else:
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)

    if random_placing:
        dx = int(np.random.uniform(imgsize - nw))
        dy = int(np.random.uniform(imgsize - nh))
    else:
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

    img = cv2.resize(img, (nw, nh))
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy:dy+nh, dx:dx+nw, :] = img

    info_img = (h, w, nh, nw, dx, dy)
    return sized, info_img

class COCODataset_new(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, model_type, data_dir='../wheat_data/train/', json_file='annotations.json',
                 name='train', img_size=416,
                 augmentation=None, min_size=1, debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        
        super(COCODataset_new, self).__init__()
        
        img_label = []
        list_img = []
        img_label_txt_name = []
        self.model_type = model_type
        
        all_img_dir = os.listdir(data_dir + "images/")
        
        
        for tmp_img_name in all_img_dir:
            if ".jpg" in tmp_img_name:
                txt_name = data_dir + "labels/"+ tmp_img_name.split(".")[0] + ".txt"
                if os.path.exists(txt_name):
                    list_img.append(data_dir + "images/" + tmp_img_name )
                    #print("list_img", list_img)
                    img_label_txt_name.append(data_dir + "labels/" + tmp_img_name.split(".")[0] + ".txt")
                      
        self.img_and_label = list_img
        self.img_label_lst = img_label_txt_name
        
        self.nF = len(list_img)
        print("Total {} imgs found.".format(self.nF))
        
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        """
        self.lrflip = augmentation['LRFLIP']
        self.jitter = augmentation['JITTER']
        self.random_placing = augmentation['RANDOM_PLACING']
        self.hue = augmentation['HUE']
        self.saturation = augmentation['SATURATION']
        self.exposure = augmentation['EXPOSURE']
        self.random_distort = augmentation['RANDOM_DISTORT']
        """


    def __len__(self):
        return  self.nF

    def __getitem__(self, file_index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        
        img_dir = self.img_and_label[file_index]
        img_label_txt = self.img_label_lst[file_index]
        
        #print("img_dir is ",img_dir )
        id_ = img_dir
        img_file = img_dir

        lrflip = False
        #if np.random.rand() > 0.5 and self.lrflip == True:
        #    lrflip = True

        # load image and preprocess
        #img_file = os.path.join(self.data_dir, self.name,id_)
        #print("img_file", img_file) 
        img = cv2.imread(img_file,1)
        
        img_org_clone = img
        #print("img type is",img.shape)
        
        assert img is not None

        img, info_img = preprocess(img, self.img_size, jitter=None,
                                   random_placing=None)
        #print("img type is",img.shape, info_img)

        #if self.random_distort:
        #   img = random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))

        if lrflip:
            img = np.flip(img, axis=2).copy()

        labels = []
        #f_l = open()
        
        try:
            #print("img_label_txt",img_label_txt)
            txt = open(img_label_txt,'r')
            lines = txt.readlines()
            #print("lines", lines)
        
            
            for j in lines:
                print("HAHAHAH;",j)
                category_id=int(j.split(' ')[0])
                #category=YOLO_CATEGORIES
                x=float(j.split(' ')[1])
                y=float(j.split(' ')[2])
                w=float(j.split(' ')[3])
                h=float(j.split(' ')[4])

                area=w*h
                bbox=[x,y,w,h]
                labels.append([])
                labels[-1].append(int(category_id))
                labels[-1].extend(bbox)
                #print( "img_org_clone.shape is : ", img_org_clone.shape,x, y, x+w, y+h )
                
                #cv2.rectangle(img_org_clone, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), (255, 0, 0))
            
        except:
                labels = []
            #return

        #print("img type",id_,labels)
        padded_labels = np.zeros((self.max_labels,13,13, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img, self.img_size, lrflip)
                
            #padded_labels[range(len(labels))[:self.max_labels]
            #              ] = labels[:self.max_labels]
        padded_labels_th = torch.from_numpy(labels)
        #print("padded_labels shape is ", padded_labels.shape,labels.shape )
        
        

        return img, padded_labels_th, info_img, id_

