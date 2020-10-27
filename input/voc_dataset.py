#!/usr/bin/evn python
#coding:utf-8

import os
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2
import os
import sys

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def get_affine_transform(size1, size2): 
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    scale1 = size2[0]*1.0/size1[0]
    scale2 = size2[1]*1.0/size1[1]
    scale = min(scale1,scale2)
    # Center to Center
    src[0, :] = [size1[0]/2.0 , size1[1]/2.0]
    dst[0, :] = [size2[0]/2.0 , size2[1]/2.0]

    # Left Center to Left Center Boarder 
    src[1, :] = [0.0 , size1[1]/2.0]
    dst[1, :] = [size2[0]/2.0 - scale*size1[0]/2.0 , size2[1]/2.0]

    # Top Center to Top Center Boader
    src[2, :] = [ size1[0]/2.0, 0.0]
    dst[2, :] = [ size2[0]/2.0 , size2[1]/2.0 - scale*size1[1]/2.0 ]
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def parse_xml(file_name):
    
    base_path_name = "../../../../data/voc_person_data/"
    base_path_xml = "../../../../data/voc2012/VOCdevkit/VOC2012/Annotations/"
    # 除去末尾的换行符
    tree = ET.parse(base_path_xml + file_name)     #打开xml文档
    root = tree.getroot()         #获得root节点
    #print "*"*10
    filename = root.find('filename').text
    filename = filename[:-4]
    print(filename)
    #file_object = open(filename + ".txt", 'w') #写文件
    file_object_log = open(base_path_name + filename + ".txt", 'w') #写文件
    flag = False

    ########################################
    for size in root.findall('size'): #找到root节点下的size节点
        width = size.find('width').text   #子节点下节点width的值
        height = size.find('height').text   #子节点下节点height的值
        print(width, height)
    ########################################

    for object in root.findall('object'): #找到root节点下的所有object节点
        name = object.find('name').text   #子节点下节点name的值
        print(name)
        bndbox = object.find('bndbox')      #子节点下属性bndbox的值
        xmin = float( bndbox.find('xmin').text)
        ymin = float( bndbox.find('ymin').text)
        xmax = float( bndbox.find('xmax').text)
        ymax = float( bndbox.find('ymax').text)
        print("xmin, ymin, xmax, ymax is ", xmin, ymin, xmax, ymax)
        if name == ("person"):
            w = (xmax - xmin)#/int(width)
            h = (ymax - ymin)#/int(height)
            xc = xmin + w*1.0/2.0
            yc = ymin + h*1.0/2.0
            file_object_log.write(str(xc/int(width)) + " " + str(yc/int(height)) + " " + str(w/int(width)) + " " + str(h/int(height))  + "\n")
            flag = True
    file_object_log.close()
    print("save path is ", base_path_name + filename + ".txt")
    
    if flag == False:  #如果没有符合条件的信息，则删掉相应的txt文件以及jpg文件
        os.remove(base_path_name +  filename + ".txt")
        
        
        
class VOCDataset_Person(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, model_type, data_dir='../wheat_data/train/', json_file='annotations.json',
                 name='train', img_size=416,
                 augmentation=None, min_size=1, debug=False):
        
        super(COCODataset_Person, self).__init__()
        
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


    def __len__(self):
        return  self.nF

    def __getitem__(self, file_index):
   
        img_dir = self.img_and_label[file_index]
        img_label_txt = self.img_label_lst[file_index]
        img_file = img_dir
        img = cv2.imread(img_file,1) 
        
        assert img is not None
        # 将图像映射为41*416
        ##  -- 计算 变换矩阵， 一个用于变换图片，一个用于变换坐标
        height = img.shape[0]
        width = img.shape[1]
        
        #print(height,width)
        
        trans = get_affine_transform((width,height),(self.img_size, self.img_size))
        img = cv2.warpAffine(img,trans,(self.img_size,self.img_size))
        img_show = img.copy()
        img= img.astype(np.float32)
        img = np.transpose(img / 255., (2, 0, 1))
      
        labels = []
        #f_l = open()
        
        #try:
        padded_labels = np.zeros((1,13,13, 5),np.float32)
        if True:
            txt = open(img_label_txt,'r')
            lines = txt.readlines()
            for i in range(0,len(lines)):
                j = lines[i].strip()

                category_id=int(j.split(' ')[0])
                
                x=float(j.split(' ')[1])*width
                y=float(j.split(' ')[2])*height
                w=float(j.split(' ')[3])*width
                h=float(j.split(' ')[4])*height
                
                x1,y1,x2,y2 = x - w/2 , y - h/ 2 , x + w / 2, y + h /2
                
                x1 =np.clip(x1 , 0 , width - 1)
                y1 =np.clip(y1 , 0 , height - 1)
                x2 =np.clip(x2 , 0 , width - 1)
                y2 =np.clip(y2 , 0 , height - 1)
                
                
                
                P1 = [[x1],[y1],[1.0]]
                P2 = [[x2],[y2],[1.0]]
                
                P1_new = np.matmul(trans,P1)
                P2_new = np.matmul(trans,P2)
                
                x1 = P1_new[0][0]
                y1 = P1_new[1][0]
                
                x2 = P2_new[0][0]
                y2 = P2_new[1][0]
                
                cv2.rectangle(img_show,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                
                xc_grid = int(xc/(self.img_size*1.0)*13)
                yc_grid = int(yc/(self.img_size*1.0)*13)
                #print("xc_grid ", xc_grid,yc_grid )
                
                xc_grid_delta = xc*1.00/(self.img_size*1.00)*13.00 - xc_grid
                yc_grid_delta = yc*1.00/(self.img_size*1.00)*13.00 - yc_grid
                
                #print("xc_grid_delta", yc_grid_delta,xc_grid_delta,xc_grid, yc_grid )
                padded_labels[0, yc_grid, xc_grid, 0] = 1.0
                padded_labels[0, yc_grid, xc_grid, 1] = xc/self.img_size #xc_grid_delta
                padded_labels[0, yc_grid, xc_grid, 2] = yc/self.img_size #yc_grid_delta
                padded_labels[0, yc_grid, xc_grid, 3] = (x2 - x1)/self.img_size
                padded_labels[0, yc_grid, xc_grid, 4] = (y2 - y1)/self.img_size

#         except:
#             labels = []
        info_img = []
        id_ = ""
        cv2.imwrite("debug_"+".jpg",img_show)
        padded_labels_th = torch.from_numpy(padded_labels)
        return img, padded_labels_th, info_img, id_
    
    
# dataset = COCODataset_Person(model_type='YOLO',
#                   data_dir='../../../../data/coco_person/coco_2017_person_yolo_format/train/',
#                   img_size=416,
#                   augmentation='')

# dataset.__getitem__(9999)


base_file_path = "../../../../data/voc2012/VOCdevkit/VOC2012/Annotations/"
file_name_lst = os.listdir(base_file_path)

for file_name in file_name_lst:
    
    parse_xml(file_name)