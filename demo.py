import numpy as np
import os
import sys


import torch
import resnet_v2
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from torch.autograd import Variable
import yolo_v1
import torch.optim as optim
import cv2
import utils

from input import coco_dataset_p
batch_size = 4
torch.cuda.set_device(3)
device = torch.device("cuda:3")

def post_process(res_feature):
    fea = res_feature.cpu().detach().numpy()
    print("fea shape is ", res_feature.shape)
    #print("fea shape is ", fea)
    
    rect_lst = []
    score_lst = []
    rect_lst_nms = []
    
    print("max confidence", np.argmax(fea[0][:][:][0]))
    for i in range(fea.shape[1]):
        for j in range(fea.shape[2]):
            #print(fea[0][i][j][0])
            if fea[0][i][j][0] > 0.3:
                print(fea[0][i][j][0])
                score_lst.append(fea[0][i][j][0])
                #tmp_box = fea[0][i][j][1:]
                bbox_x = j*(416.0/13) + fea[0][i][j][1]*(416.0/13) #fea[0][i][j][1]*416 #j*(416.0/13) + fea[0][i][j][1]*(416.0/13)
                bbox_y = i*(416.0/13) + fea[0][i][j][2]*(416.0/13) #fea[0][i][j][2]*416 #i*(416.0/13) + fea[0][i][j][2]*(416.0/13)
                bbox_w =  fea[0][i][j][3]*416
                bbox_h =  fea[0][i][j][4]*416
                
                rect_lst.append([bbox_x-bbox_w/2.0,bbox_y-bbox_h/2,bbox_w,bbox_h])
                rect_lst_nms.append([bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h ])
      
    res = utils.nms(np.array(rect_lst_nms ), 0.4, np.array(score_lst) )
    print("rect_lst is ", res)
    res_nms = []
    for i in range(0,len(rect_lst_nms)):
        if i in res:
            res_nms.append(rect_lst[i])
    return res_nms
                
            
    
@torch.no_grad()
def change2eval(m):
    if type(m)== nn.BatchNorm2d :
        m.track_running_stats = False    
    
    
    
model = yolo_v1.Yolo_v1_2()  
ckpt = torch.load(sys.argv[1])
model.load_state_dict(ckpt['model'])


model = model.cuda()

cuda = torch.cuda.is_available() and True
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#state = torch.load(sys.argv[1])
print(sys.argv[1])
# print("state ", state)
#state.apply(change2eval)

#state.to(device)
model.eval()
#if 'model_state_dict' not in state.keys():
#model.load_state_dict( torch.load(sys.argv[1]))

for img_name in os.listdir("./test_data/test_voc_model")[:]:
    if ".jpg"  not in img_name:
        continue
    
    img = cv2.imread('test_data/test_voc_model/' + img_name, 1)
    print("img_name", img_name)
    
    
    img_size = 416
    
    trans = coco_dataset_p.get_affine_transform((img.shape[1],img.shape[0]),(img_size, img_size))
    img = cv2.warpAffine(img, trans, (img_size,img_size))
    
    #img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32)/255.0
    img_raw = img.astype(np.float32)/255.0
    img_raw = img_raw.transpose((2, 0, 1))
    #print(img_raw.shape,img_raw[:,200,:])
    
    #img_raw = np.zeros((1,3,416,416)).astype(np.float32)
    img_th = torch.from_numpy(img_raw).reshape(1,3, img_size, img_size).to(device)
    #print(img_th)
    
    with torch.no_grad():
    
        res_feature = model(img_th)
        #print(res_feature)
        res_feature = res_feature.sigmoid()

        rect_lst = post_process(res_feature)


        img_show = img.copy()
        for rect in rect_lst:
            bbox_x2 = rect[0] + rect[2] 
            bbox_y2 = rect[1] + rect[3] 
            
            #bbox_x2 =  rect[2] 
            #bbox_y2 =  rect[3] 
            cv2.rectangle(img_show,(int(rect[0]), int(rect[1])),(int(bbox_x2),int(bbox_y2)),(0,255,0),2)



        cv2.imwrite( "test_data/test_coco_nms_1/debug_" + str(img_name) + ".jpg", img_show )


