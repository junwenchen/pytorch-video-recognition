import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import time
import pdb
import os

class _graphFront(nn.Module):
    # def __init__(self,rois):
    #     # super(_fasterRCNN, self).__init__()
    #     self.rois = rois

    def IoU(self, boxa, boxb):
        inter_x1 = max(boxa[0],boxb[0])
        inter_x2 = min(boxa[2],boxb[2])
        inter_y1 = max(boxa[1],boxb[1])
        inter_y2 = min(boxa[3],boxb[3])

        inter_area = max((inter_x2-inter_x1),0) * max((inter_y2-inter_y1),0)
        boxa_area = (boxa[2]-boxa[0]+1)*(boxa[3]-boxa[1]+1)
        boxb_area = (boxb[2]-boxb[0]+1)*(boxb[2]-boxb[1]+1)
        iou = inter_area/(boxa_area + boxb_area - inter_area)
        return iou

    def build_graph(self, rois):
        num_frames = len(rois)
        adjacent_matrix = torch.zeros((num_frames*20, num_frames*20))
        for t in range(num_frames-1):
            roi_t = rois[t]
            roi_t1 = rois[t+1]
            for i in range(len(roi_t)):
                for j in range(len(roi_t+1)):
                    adjacent_matrix[20*(t-1)+i][20*(t-1)+j] = self.IoU(roi_t[i], roi_t1[j])
        return adjacent_matrix.view(1,num_frames*20,num_frames*20)

    def gcn(self, rois):
        num_frames = len(rois)
        adjacent_matrix = torch.zeros((num_frames*20,num_frames*20))
        for t in range(num_frames-1):
            roi_t = rois[t]
            roi_t1 = rois[t+1]
            for i in range(len(roi_t)):
                for j in range(len(roi_t+1)):
                    adjacent_matrix[20*(t-1)+i][20*(t-1)+j] = self.IoU(roi_t[i], roi_t1[j])
        return adjacent_matrix

# class _graphFront(nn.Module):
#     # def __init__(self):
#     #     # super(_fasterRCNN, self).__init__()
#     #     self.iou = self.IOU()
#     # def forward(self, box1, box2):
#     def IOU(self, boxa, boxb):
#         inter_x1 = max(boxa[0],boxb[0])
#         inter_x2 = min(boxa[2],boxb[2])
#         inter_y1 = max(boxa[1],boxb[1])
#         inter_y2 = min(boxa[3],boxb[3])
#
#         inter_area = max((inter_x2-inter_x1),0) * max((inter_y2-inter_y1),0)
#         boxa_area = (boxa[2]-boxa[0]+1)*(boxa[3]-boxa[1]+1)
#         boxb_area = (boxb[2]-boxb[0]+1)*(boxb[2]-boxb[1]+1)
#         iou = inter_area/(boxa_area + boxb_area - inter_area)
#         return iou

# s = _graphFront()
# datadir = "/data/dataset/something-somthing-v2/result/2058/"
# num_frames = len(os.listdir(datadir))
# node = np.zeros((num_frames*20,num_frames*20))
#
# for t in range(1,num_frames):
#     print(os.path.join(datadir,"%06d" % t + '_det.txt'))
#     with open(os.path.join(datadir,"%06d" % t + '_det.txt')) as fi:
#         objects_t = fi.readlines()
#         # print(objects_t)
#         # objects_t = objects_t.strip().split('\n')
#         with open(os.path.join(datadir,"%06d" % (t+1) + '_det.txt')) as fj:
#              objects_t1 = fj.readlines()
#              # objects_t1 = objects_t1.strip().split('\n')
#              for i in range(len(objects_t)):
#                  itemt = [float(x) for x in objects_t[i].strip().split(' ')]
#                  for j in range(len(objects_t1)):
#                      itemt1 = [float(x) for x in objects_t1[j].strip().split(' ')]
#                      # print(itemt)
#                      # print(itemt1)
#                      node[20*(t-1)+i][20*(t-1)+j] = s.IOU(itemt,itemt1)
#
# print(node)
# print(node.shape)
