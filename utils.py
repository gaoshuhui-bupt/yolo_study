import numpy as np
import os
import sys


import torch
import resnet_v2
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import torchvision.models as models

def bbox_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    b1_len = boxes1.size(0)
    b2_len = boxes2.size(0)

    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions   
    
#bbox is N*4,score is N*1
#thres
def nms_1(bbox, thres, score, limit=None):
    
    #
    if len(bbox)==0:
        return
    
    order = score.argsort()[::-1]
    print("order is ", order)
    print("bbox[order] is ", bbox[order])
    
    area = np.prod(bbox[:,2:] - bbox[:,:2], axis=1)
    print("area is ", area )
    select = np.zeros(len(area), dtype=bool)
    print("select is ",  select,bbox[select])
    
    
    for i,b in enumerate(bbox):
        
        min_xy = np.maximum(b[:2], bbox[select,:2])
        max_xy = np.minimum(b[2:], bbox[select,2:])

        area_tmp = np.prod(max_xy - min_xy, axis=1) * ( max_xy > min_xy ).all(axis=1)
        iou = area_tmp / ( area[i] + area[select] - area_tmp )

        print("area is",i, ",",  area[select],iou )  
        if (iou > thres).any():
            continue
            
        select[i] = True
        
        if limit is not None and np.sum(select!=False)==limit:
            break
           
    #print("selec ... ", select)
    selec = np.where(select)#[0]
    #print("selec is selec ", selec)
    if score is not None:
        select_res = order[selec]
    #print("select_res", select_res)
    return select_res
    
    
    
def nms(bbox, thresh, score=None, limit=None): 
    """Suppress bounding boxes according to their IoUs and confidence scores.
    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K\leq R`.
    """
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)
    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        #print("selec[i]", selec,i,b,b[:2],  bbox[selec, :2])
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)   
        #print("iou", area,iou, thresh )
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break
            
    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    计算AP值，若use_07_metric=true,则用11个点采样的方法，将rec从0-1分成11个点，这些点prec值求平均近似表示AP
    若use_07_metric=false,则采用更为精确的逐点积分方法
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
 
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
 
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
 
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


#compute precision and recall
def voc_eval(BB, confidence, BBGT):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: 产生的txt文件，里面是一张图片的各个检测框结果。
    annopath: xml 文件与对应的图像相呼应。
    imagesetfile: 一个txt文件，里面是每个图片的地址，每行一个地址。
    classname: 种类的名字，即类别。
    cachedir: 缓存标注的目录。
    [ovthresh]: IOU阈值，默认为0.5，即mAP50。
    [use_07_metric]: 是否使用2007的计算AP的方法，默认为Fasle
    
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # 首先加载Ground Truth标注信息。
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    # 即将新建文件的路径
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # 读取文本里的所有图片路径
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    # 获取文件名，strip用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    imagenames = [x.strip() for x in lines]
    #如果cachefile文件不存在，则写入
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            #annopath.format(imagename): label的xml文件所在的路径
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            #写入cPickle文件里面。写入的是一个字典，左侧为xml文件名，右侧为文件里面个各个参数。
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # 对每张图片的xml获取函数指定类的bbox等
    class_recs = {}# 保存的是 Ground Truth的数据
    npos = 0
    for imagename in imagenames:
        # 获取Ground Truth每个文件中某种类别的物体
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        #  different基本都为0/False
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult) #自增，~difficult取反,统计样本个数
        # # 记录Ground Truth的内容
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets 读取某类别预测输出
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines] # 图片ID
    confidence = np.array([float(x[1]) for x in splitlines]) # IOU值
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) # bounding box数值

    """
    
    ovthresh  = 0.5
    
    
    
    #BB id predict box, confidence is score, BBGT is groundtruth,
    
    # 对confidence的index根据值大小进行降序排列。
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    #重排bbox，由大概率到小概率。
    BB = BB[sorted_ind, :]
    # 图片重排，由大概率到小概率。
    #image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(BB)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    flag_gt = [False  for i in range(0,len(BBGT))]
    
    for d in range(nd):
        bb = BB[d, :].astype(float) 


        if len(BBGT) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if True:
                if flag_gt[jmax] == False:
                    tp[d] = 1.
                    flag_gt[jmax] = True
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    print("np.cumsum(fp)", np.cumsum(fp), fp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(BBGT.shape[0]) #)float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    use_07_metric = False
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap#, ap

#compute  average precision, recall
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')



#bbox = [[1,2,3,4],[3,4,5,6],[2,2,3,4]]
#thres = 0.3
#score = [0.3, 0.8, 0.9]
#nms_1(np.array(bbox), thres, np.array(score), limit=None)
a = [ [0,0,5,5], [5,5,10,10] ]
gt = [ [0,0,5,5],[5,5,10,10],[20,20,50,50] ]
confidence = [0.9,0.6]

c = voc_eval(np.array(a), np.array(confidence), np.array(gt))
print("c", c)