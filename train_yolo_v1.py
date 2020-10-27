import numpy as np
import os
import sys

import torch
import resnet
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from torch.autograd import Variable
import yolo_v1
import torch.optim as optim
import cv2
import torch.nn.functional as F

from input import coco_dataset_p
batch_size = 8
torch.cuda.set_device(1)


cuda = torch.cuda.is_available() and True
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor



debug_show = True

class yolo_v1_Loss(nn.Module):
    
    def __init__(self, S, B, l_coord, l_noobj):
        super(yolo_v1_Loss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        
    def forward(self, fea, target,imgs):
        
        
        if debug_show :
            score_pred_show = (fea[0,:,:,0].reshape(13,13)*255).cpu().detach().numpy().astype(np.uint8)
            score_gt_show = (target[0,:,:,0].reshape(13,13)*255).cpu().detach().numpy().astype(np.uint8)

            score_show = np.zeros((13,26),np.uint8)
            score_show[:,0:13]  = score_pred_show
            score_show[:,13:26] = score_gt_show
            score_show = cv2.resize(score_show,(260,130))
            cv2.imwrite("score_show.png",score_show)
            
            cx_offset_pred = (fea[0,:,:,1].reshape(13,13)).cpu().detach().numpy()
            cy_offset_pred = (fea[0,:,:,2].reshape(13,13)).cpu().detach().numpy()
            w_pred = (fea[0,:,:,3].reshape(13,13)).cpu().detach().numpy()
            h_pred = (fea[0,:,:,4].reshape(13,13)).cpu().detach().numpy()
            cx_offset_gt   = (target[0,:,:,1].reshape(13,13)).cpu().detach().numpy()
            cy_offset_gt   = (target[0,:,:,2].reshape(13,13)).cpu().detach().numpy()
            w_gt = (target[0,:,:,3].reshape(13,13)).cpu().detach().numpy()
            h_gt = (target[0,:,:,4].reshape(13,13)).cpu().detach().numpy()
            
            
            img_show_1 = (imgs[0].reshape(3,416,416)*255.0).cpu().detach().numpy().astype(np.uint8).transpose((1, 2, 0))
#             img_show = img_show.astype(np.uint8)
            img_show = img_show_1.copy().reshape(416,416,3).astype(np.uint8) #np.zeros((416,416,3),dtype=np.uint8)
            for i in range(0,13):
                for j in range(0,13):
                    if score_gt_show[i,j] > 0 :
                        cx_pred = int((cx_offset_pred[i,j] + j)*416.0/13.0)
                        cy_pred = int((cy_offset_pred[i,j] + i)*416.0/13.0)
                        cx_gt = int((cx_offset_gt[i,j] + j)*416.0/13.0)
                        cy_gt = int((cy_offset_gt[i,j] + i)*416.0/13.0)
                        img_show = cv2.circle(img_show,(cx_pred,cy_pred),4,(0,0,255),-1)
                        img_show = cv2.circle(img_show,(cx_gt,cy_gt),2,(0,255,0),-1)
                        
                        w_i_pred = int(w_pred[i,j]*416)
                        h_i_pred = int(h_pred[i,j]*416)
                        
                        w_i_gt = int(w_gt[i,j]*416)
                        h_i_gt = int(h_gt[i,j]*416)
                        
                        cv2.rectangle(img_show,(cx_pred - w_i_pred//2 , cy_pred - h_i_pred//2), (cx_pred + w_i_pred//2 , cy_pred + h_i_pred//2), (0,0,255),2)
                        cv2.rectangle(img_show,(cx_gt - w_i_gt//2 , cy_gt - h_i_gt//2), (cx_gt + w_i_gt//2 , cy_gt + h_i_gt//2),  (0,255,0),2)
                        
            cv2.imwrite("img_show.png",img_show)

        
        N = fea.shape[0]

        n_elem = target.shape[-1]
        loc_loss = []
        confidence_loss = []

        coo_mask = target[:, :, :, 0] > 0
        #print(" coo_mask.type is : ", coo_mask.dtype)

        # shape same as target
        coo_mask = coo_mask.unsqueeze(-1).expand(target.shape)
        #coo_mask = coo_mask.bool()
        obj_pred = fea[coo_mask].view(-1,n_elem)
        obj_target = target[coo_mask].view(-1,n_elem)

        #compute object confidence loss
        obj_pred_mask = torch.cuda.BoolTensor(obj_pred.size())
        obj_pred_mask.zero_()
        obj_pred_mask[:,0] = True

        obj_pred_use = obj_pred[obj_pred_mask]
        obj_target_use = obj_target[obj_pred_mask]
        loss_obj_conf = F.mse_loss(obj_pred_use, obj_target_use, size_average=False)

        #compute location loss
        #print("obj_pred_use shape",obj_pred[:,1:3].shape ,obj_target[:,3:].shape)
        loss_obj_loc_xy = F.mse_loss(obj_pred[:,1:3], obj_target[:,1:3], size_average=False)
        loss_obj_loc_wh = F.mse_loss(torch.sqrt(obj_pred[:,3:]),torch.sqrt( obj_target[:,3:]), size_average=False)
        loss_obj_loc = loss_obj_loc_xy + loss_obj_loc_wh

        # no obj confidence loss
        coo_mask_noobj = target[:, :, :, 0] == 0
        coo_mask_noobj = coo_mask_noobj.unsqueeze(-1).expand(target.shape)
        noo_pred = fea[coo_mask_noobj].view(-1,n_elem)
        noo_target = target[coo_mask_noobj].view(-1,n_elem)

        noobj_pred_mask = torch.cuda.BoolTensor(noo_pred.size())
        noobj_pred_mask.zero_()
        noobj_pred_mask[:,0] = True

        noo_pred_use = noo_pred[noobj_pred_mask]
        noo_target_use = noo_target[noobj_pred_mask]
        loss_noo_conf = F.mse_loss(noo_pred_use, noo_target_use, size_average=False)
        print("loss all ",  loss_obj_conf.item()*2, loss_noo_conf.item())
        
        return (loss_obj_loc * 5 + 2 * loss_obj_conf + loss_noo_conf) / N
        # return (  2 * loss_obj_conf + loss_noo_conf) / N




dataset =  coco_dataset_p.COCODataset_Person(model_type='YOLO',
                  data_dir='../../../../my_task/data/coco_person/coco_2017_person_yolo_format/train/images/',
                  img_size=416,
                  augmentation='')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

base_lr = 0.0001
model = yolo_v1.Yolo_v1_2()  
model = model.cuda()
model.train()
     
#optimizer = optim.Adam(params, lr=base_lr, momentum=0.9,
#                          dampening=0, weight_decay=0.0005 * batch_size)、
print("model.parameters() is ", model.parameters())
for name,p in model.named_parameters():
    
    print(name)
    print(p.requires_grad)
    #print(...)

optimizer = torch.optim.Adam(model.parameters(), lr = base_lr )
criterion = yolo_v1_Loss(7, 2, 5, 0.5)

#scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=-1)
loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
#optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
 
resume = False
start_epoch = 0
if resume:
    checkpoints_name = 'checkpoints/yolo_v1_0060_new.pth'
    base_lr = 0.000025
    if os.path.isfile(checkpoints_name):
        checkpoint = torch.load(checkpoints_name)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        
for epoch in range(start_epoch, 200):
    loss_all = 0
    for i, data in enumerate(dataloader, 0):
        
        #imgs = Variable(data[0].type(dtype))
        imgs = data[0].cuda()
        #imgs = torch.zeros((1,3,416,416)).cuda()
        #print("imgs",imgs[:,:,200,200:210] )
        #targets = Variable(data[1].type(dtype), requires_grad=False)
        targets = data[1].cuda()
        
        
        
        #print("targets ",targets )
        
        out_fea = model(imgs)
        out_fea = out_fea.sigmoid()

        #print("out_fea", out_fea)
        #print("targets", targets)
        
        targets = targets.squeeze(dim=1)
        
        loss = criterion(out_fea, targets, imgs)
        loss_all += loss.item()
        loss_tr = loss.item()
        
        optimizer.zero_grad()
        loss.backward()


        optimizer.step()
        #scheduler.step()
        
        #tr_loss += loss.item()

            
            #loss_meter.add(loss.data[0])
        if i%200==0 and i > 0:
            #print('loss', loss_all/(i+1)*1.00)
            print('loss', loss_tr*1.00,loss_all/(i+1)*1.00, i+1)
            model_save_path = "checkpoints/yolo_v1_step" + str(i).zfill(4) + "_new.pth"
            state_model = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state_model, model_save_path)
            
     
    print('epoch:{}    train_loss:{}  lr:{}'.format(
    epoch+1, loss_all/len(dataloader), optimizer.state_dict()['param_groups'][0]['lr']))


    if epoch%1 == 0 and epoch > 0:
        #model_save_path = "checkpoints/yolo_v1_" + str(epoch).zfill(4) + ".pth"
        #torch.save(model, model_save_path)
        
        model_save_path = "checkpoints/yolo_v1_" + str(epoch).zfill(4) + "_new_1.pth"
        state_model = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state_model, model_save_path)

