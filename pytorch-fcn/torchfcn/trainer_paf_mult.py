import datetime
import math
import os
import os.path as osp
import shutil
import scipy.io
import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import torch.nn as nn
import torchfcn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from scipy.ndimage.filters import gaussian_filter
import collections
import os
import os.path as osp
import scipy
import cv2
import socket

lossTEST = nn.MSELoss().cuda()

def visualize_( lbl_pred, lbl_true, img):
    # lbl_pred = np.dstack([lbl_pred,lbl_pred,lbl_pred])
    # lbl_true = np.dstack([lbl_true,lbl_true,lbl_true])
    lbl_pred = cv2.applyColorMap(lbl_pred, cv2.COLORMAP_JET)
    lbl_pred = cv2.cvtColor(lbl_pred,cv2.COLOR_BGR2RGB)
    lbl_true = cv2.applyColorMap(lbl_true, cv2.COLORMAP_JET)
    lbl_true = cv2.cvtColor(lbl_true,cv2.COLOR_BGR2RGB)
    # vis = np.hstack([img,(lbl_pred*255).astype("uint8"),(lbl_true*255).astype("uint8")])
    vis = np.hstack([img,lbl_pred,lbl_true])
    return vis

def pose_metric(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def paf_pose_metric(input_pose,input_paf,matpath):
    # return 0.
    mat = scipy.io.loadmat(matpath)
    input_pose = np.transpose(input_pose,(1,2,0))
    input_paf = np.transpose(input_paf,(1,2,0))
    input_pose = cv2.resize(input_pose,(0,0),fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
    input_paf = cv2.resize(input_paf,(0,0),fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
    input_pose = np.transpose(input_pose,(2,0,1))
    input_paf = np.transpose(input_paf,(2,0,1))
    # print(input_pose.shape,input_paf.shape)
    def fun1(arr):
        count = 0
        for ind in range(len(arr)):
            d = arr[ind]
            if d>0.001:
                count = count + 1
        return count
    
    stft_p = input_pose
    c, h, w = stft_p.shape
    peak_counter = 0
    all_peaks = []
    for part in range(c):
        x_list = []
        y_list = []
        map_ori = stft_p[part,:,:]
        map_ori = map_ori/map_ori.max()
        thr1 = 0.3
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > 2*thr1))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    # print(len(all_peaks))
    limbSeq =  [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]

    connection_all = []
    special_k = []
    mid_num = 10
    for k in range(len(limbSeq)):
        # print("k")
        score_mid = input_paf[[2*k,2*k-1],:,:]
        candA = all_peaks[limbSeq[k][0]]
        candB = all_peaks[limbSeq[k][1]]

        nA = len(candA)
        nB = len(candB)

        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                # print(nB)
                for j in range(nB):
                    # print("j")
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                               np.linspace(candA[i][1], candB[j][1], num=mid_num))

                    vec_x = np.array([score_mid[0,int(round(startend[I][1])), int(round(startend[I][0]))] \
                              for I in range(len(startend))])
                    vec_y = np.array([score_mid[1,int(round(startend[I][1])), int(round(startend[I][0]))] \
                              for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])

                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*h/norm-1, 0)

                    criterion1 = fun1(score_midpts) > 0.8 * len(score_midpts)

                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                # print("c")
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # print(len(connection_all))
    subset = -1 * np.ones((0, 16))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    
    for k in range(len(limbSeq)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k])
            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    #print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

            # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])	

    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    # print(len(subset))
    pck = 0
    count = 1
    frc = 0.00001
    
    for n in range(len(subset)):
        indexes = subset[n][:-2]
        min_dist = float("inf")
        x_pred = candidate[indexes.astype(int),1]
        y_pred = candidate[indexes.astype(int),0]
        for human in mat['joints'][0]:
            poselist = np.around(human[:,:-1]).astype(np.int64)
            
            dist = ((x_pred - poselist[:,0])**2 + (y_pred - poselist[:,1])**2)**0.5
            xd = []
            yd = []
            if np.sum(dist)<np.sum(min_dist):
                min_dist = dist
                closest_human = poselist
                min_limb = float("inf")
            
                for i, limb in enumerate(limbSeq):
                    
                    p1 = limb[0]
                    p2 = limb[1]
                    x1,y1 = poselist[p1,:]
                    x2,y2 = poselist[p2,:]
                    ll = ((x1-x2)**2 + (y1-y2)**2)**0.5
                    if ll<min_limb and ll!=0:
                        min_limb = ll
                PCK_t = 2*min_limb

        for i in range(14):
            index = subset[n][i]
            x_p = candidate[index.astype(int),1]
            y_p = candidate[index.astype(int),0]
            x_gt = closest_human[i,0]
            y_gt = closest_human[i,1]
            dist = ((x_p - x_gt)**2 + (y_p - y_gt)**2)**0.5
            if dist < PCK_t:
                pck = float((pck*count + dist + frc)/count)
                count = count + 1
            # print("dist:"+str(dist)+", PCK_T:"+str(PCK_t))
    return(pck) 

def vishuman(input_pose,input_paf):
    input_pose = np.transpose(input_pose,(1,2,0))
    input_paf = np.transpose(input_paf,(1,2,0))
    input_pose = cv2.resize(input_pose,(0,0),fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
    input_paf = cv2.resize(input_paf,(0,0),fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
    input_pose = np.transpose(input_pose,(2,0,1))
    input_paf = np.transpose(input_paf,(2,0,1))
    def fun1(arr):
        count = 0
        for ind in range(len(arr)):
            d = arr[ind]
            if d>0.001:
                count = count + 1
        return count
    stft_p = input_pose
    c, h, w = stft_p.shape
    peak_counter = 0
    all_peaks = []
    for part in range(c):
        x_list = []
        y_list = []
        map_ori = stft_p[part,:,:]
        map_ori = map_ori/map_ori.max()
        thr1 = 0.3
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > 2*thr1))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)


    limbSeq =  [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]

    connection_all = []
    special_k = []
    mid_num = 10
    for k in range(len(limbSeq)):
        # print("k")
        score_mid = input_paf[[2*k,2*k-1],:,:]
        candA = all_peaks[limbSeq[k][0]]
        candB = all_peaks[limbSeq[k][1]]

        nA = len(candA)
        nB = len(candB)

        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                # print(nB)
                for j in range(nB):
                    # print("j")
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                               np.linspace(candA[i][1], candB[j][1], num=mid_num))

                    vec_x = np.array([score_mid[0,int(round(startend[I][1])), int(round(startend[I][0]))] \
                              for I in range(len(startend))])
                    vec_y = np.array([score_mid[1,int(round(startend[I][1])), int(round(startend[I][0]))] \
                              for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])

                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*h/norm-1, 0)

                    criterion1 = fun1(score_midpts) > 0.8 * len(score_midpts)

                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                # print("c")
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 16))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(limbSeq)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k])
            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    #print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

            # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    outim = np.zeros((input_pose.shape[1],input_pose.shape[2],3))
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for n in range(len(subset)):
        for i in range(14):
            index = subset[n][i]
            x_p = candidate[index.astype(int),1]
            y_p = candidate[index.astype(int),0]
            cv2.circle(outim, (int(x_p),int(y_p)), 4, colors[i])
        for i,limb in enumerate(limbSeq):
            index1 = subset[n][limb[0]]
            index2 = subset[n][limb[1]]
            x_1 = candidate[index1.astype(int),1]
            y_1 = candidate[index1.astype(int),0]
            x_2 = candidate[index2.astype(int),1]
            y_2 = candidate[index2.astype(int),0]
            length = ((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2) ** 0.5
            angle = math.degrees(math.atan2(x_1 - x_2, y_1 - y_2))
            polygon = cv2.ellipse2Poly((int((y_1+y_2)/2),int((x_1+x_2)/2)), (int(length/2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(outim, polygon, colors[i])
    return outim

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss
    
def crop_pred(input, target):
    n, c, h, w = input.size()
    n_t , h_t, w_t = target.size()
    
    pad_w1 = (w-w_t)/2
    # pad_w2 = (w-w_t)/2 if (((w-wt)%2)==0) else (w-w_t+1)/2
    pad_h1 = (h-h_t)
    # pad_h2 = (h-h_t)/2 if (((h-h_t)%2)==0) else ((h-h_t)+1)/2
    
    input = input[:, :, pad_h1:pad_h1 + target.size()[1], pad_w1:pad_w1 + target.size()[2]].contiguous()
    return input

def cross_entropy2d_wo_crop(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    
    #changing the padding of the prediction based on the target 
    
#     n_t , h_t, w_t = target.size()
    
#     pad_w1 = (w-w_t)/2
#     # pad_w2 = (w-w_t)/2 if (((w-wt)%2)==0) else (w-w_t+1)/2
#     pad_h1 = (h-h_t)
#     # pad_h2 = (h-h_t)/2 if (((h-h_t)%2)==0) else ((h-h_t)+1)/2
    
#     input = input[:, :, pad_h1:pad_h1 + target.size()[1], pad_w1:pad_w1 + target.size()[2]].contiguous()
    
    #     padd = (pad_w1,pad_w2,pad_h1,pad_h2)
    #     padder = nn.ZeroPad2d(padd)
    #     target = padder(target.view(n_t,1,h_t,w_t))
    
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss,pad_h1,pad_w1

def pose_loss(input, target, weight=None, size_average=False):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    mask = target > 0.001
    # input = torch.masked_select(input, mask)
    # target = torch.masked_select(target, mask)
    mask = mask.type(torch.cuda.FloatTensor)
    input = input*mask
    target = target*mask
    loss = lossTEST(input, target)
    if size_average:
        loss /= mask.data.sum()
    return loss

def paf_loss(input, target, weight=None, size_average=False):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    mask = target != 0.
    # input = torch.masked_select(input, mask)
    # target = torch.masked_select(target, mask)
    mask = mask.type(torch.cuda.FloatTensor)
    input = input*mask
    target = target*mask
    loss = lossTEST(input, target)
    if size_average:
        loss /= mask.data.sum()
    return loss

def pose_loss_test(input, target, weight=None, size_average=False):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    mask = target > 0.001
#     mask_ = mask==0
    mask = mask.type(torch.cuda.FloatTensor)
#     mask_ = mask_.type(torch.cuda.FloatTensor)
    
    input = input*mask
    target = target*mask
    loss = torch.sum((target - input)**2)
    # number = torch.nonzero(mask.data).size(0)
    # loss/=number
    # # mask_ = 1. - mask
    # loss2 = torch.sum(((input*mask_)**2))
    # # number = torch.nonzero(mask_.data).size(0)
    # loss_= loss + loss2#/number
    return loss

def paf_loss_test(input, target, weight=None, size_average=False):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    mask = target != 0.
    # mask_ = mask==0
    mask = mask.type(torch.cuda.FloatTensor)
    # mask_ = mask_.type(torch.cuda.FloatTensor)
    
    input = input*mask
    target = target*mask
    loss = torch.sum((target - input)**2)
    # number = torch.nonzero(mask.data).size(0)
    # loss/=number
    # mask_ = 1. - mask
    # loss2 = torch.sum(((input*mask_)**2))
    # loss_= loss + loss2#/number
    return loss

class TrainerPAF_mult(object):

    def __init__(self, cuda, model, optimizer,scheduler,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        self.cuda = cuda
        tboarddir = os.path.join('/new_data/gpu/ayushya/tblogs/PAF-mult-new', datetime.datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
        self.writer = SummaryWriter(tboarddir)
        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        # self.gamma = gamma
        # self.lr = lr
        

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/loss_pose',
            'train/loss_paf',
            'train/met',
            'valid/loss',
            'valid/loss_pose',
            'valid/loss_paf',
            'valid/met',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    #resizing ground truth
    
    def ground_resize(self,x):
        m = nn.AvgPool2d((2, 2), stride=(2, 2))
        x = m(m(m(x)))
        return x
    
    
    """-----------------------------VALIDATION EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------VALIDATION EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------VALIDATION EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------VALIDATION EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------VALIDATION EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------VALIDATION EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------VALIDATION EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------VALIDATION EPOCH CODE FOR PAF LOADER---------------------"""
    
        
    def validate(self):
        self.model.eval()

        n_class = len(self.val_loader.dataset.pose_names)

        val_loss, pose_loss, paf_loss = 0, 0, 0
        valid_vis_param = 4
        vis_pose, vis_paf = [], []
        metls = []
        label_trues, label_preds = [], []
        for batch_idx, (data, targetpose, targetpaf, targetpath) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            
            if self.cuda:
                data, targetpose, targetpaf = data.cuda(), targetpose.cuda(), targetpaf.cuda()
            
            data, targetpose, targetpaf = Variable(data, volatile=True), Variable(targetpose), Variable(targetpaf)
            
            m = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)
            mp = nn.ZeroPad2d(99)

            targetpose = m(m(m(mp(targetpose))))
            targetpaf = m(m(m(mp(targetpaf))))
            # targetpose = m(m(m(targetpose)))
            # targetpaf = m(m(m(targetpaf)))
            
            scorepose1, scorepaf1, scorepose2, scorepaf2 = self.model(data)
            
            losspose1 = pose_loss_test(scorepose1, targetpose,
                                   size_average=self.size_average)
            losspaf1 = paf_loss_test(scorepaf1, targetpaf,
                                   size_average=self.size_average)
            losspose2 = pose_loss_test(scorepose2, targetpose,
                                   size_average=self.size_average)
            losspaf2 = paf_loss_test(scorepaf2, targetpaf,
                                   size_average=self.size_average)
            
            
            # losspose = pose_loss_test(scorepose, targetpose,
                                   # size_average=self.size_average)
            # losspaf = paf_loss_test(scorepaf, targetpaf,
                                   # size_average=self.size_average)
            loss = losspose1+losspaf1+losspose2+losspaf2
            losspose = losspose1+losspose2
            losspaf = losspaf1+losspaf2
            
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            
            val_loss += float(loss.data[0]) / len(data)
            pose_loss += float(losspose.data[0]) / len(data)
            paf_loss += float(losspaf.data[0]) / len(data)

            # val_loss /= len(self.val_loader)

            
            #____________________________Visualise_______________________
            
            if(len(vis_paf)<valid_vis_param):
                imgs = data.data.cpu()
                pose_pred = scorepose2.data.cpu().numpy()
                paf_pred = scorepaf2.data.cpu().numpy()
                pose_lab = targetpose.data.cpu().numpy()
                paf_lab = targetpaf.data.cpu().numpy()

                for img, lt_po, lp_po ,lt_pf ,lp_pf in zip(imgs, pose_lab, pose_pred, paf_lab, paf_pred):
                    img = self.val_loader.dataset.untransform(img)
                    lt_po = np.max(lt_po,axis=0)
                    lp_po = np.max(lp_po,axis=0)
                    lt_pf = np.max(np.abs(lt_pf),axis=0)
                    lp_pf = np.max(np.abs(lp_pf),axis=0)
                    lt_po = scipy.misc.imresize(lt_po, (img.shape[0],img.shape[1]))
                    lt_pf = scipy.misc.imresize(lt_pf, (img.shape[0],img.shape[1]))
                    lp_po = scipy.misc.imresize(lp_po, (img.shape[0],img.shape[1]))
                    lp_pf = scipy.misc.imresize(lp_pf, (img.shape[0],img.shape[1]))
                    # mask_po = lt_po > 0.02
                    # lp_po = lp_po*mask_po
                    viz = visualize_(
                        lbl_pred=lp_po, lbl_true=lt_po, img=img)
                    vis_pose.append(viz)

                    # mask_pf = lt_pf != 0
                    # lp_pf = lp_pf*mask_pf
                    viz = visualize_(
                        lbl_pred=lp_pf, lbl_true=lt_pf, img=img)
                    vis_paf.append(viz)

            
            #________________Metric________________________
            metrics = []
            pose_pred = scorepose2.data.cpu().numpy()
            paf_pred = scorepaf2.data.cpu().numpy()
            
            for posep, pafp, patht in zip(pose_pred,paf_pred,targetpath):
                metout = paf_pose_metric(posep, pafp, patht)
            # metout = pose_metric(scorepose.data, targetpose.data)
                print(metout)
                metrics.append((metout,))
            metls.append(np.mean(metrics, axis=0))
            # metls.append(metout)
        # """Out of the forward loop"""
        val_loss /= len(self.val_loader)
        
        
            
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - \
                self.timestamp_start
            log = [self.epoch, self.iteration] + ['']*4 + \
                  [val_loss] + [pose_loss] + [paf_loss] + [np.mean(metls, axis=0)] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        
        self.writer.add_image('Image_pose', fcn.utils.get_tile_image(vis_pose), self.iteration)
        self.writer.add_image('Image_paf', fcn.utils.get_tile_image(vis_paf), self.iteration)
        self.writer.add_scalars('valid', {"loss": val_loss,
                                          "loss_pose": pose_loss,
                                          "loss_paf": paf_loss,
                                         "met": np.mean(metls, axis=0)}, self.iteration)

        mean_iu = np.mean(metls,axis = 0)
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))
    
    
    
    """-----------------------------TRAINING EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------TRAINING EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------TRAINING EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------TRAINING EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------TRAINING EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------TRAINING EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------TRAINING EPOCH CODE FOR PAF LOADER---------------------"""
    """-----------------------------TRAINING EPOCH CODE FOR PAF LOADER---------------------"""
            
    def train_epoch(self):
        self.model.train()
        train_vis_param = 4

        n_class = len(self.train_loader.dataset.pose_names)
        vis_pose, vis_paf = [],[]
        ravg, poseavg, pafavg = [], [], []
        rmet = []
        for batch_idx, (data, targetpose, targetpaf, targetpath) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            # if self.iteration % self.interval_validate == 0 and self.iteration>1:
            #     self.validate()

            if self.cuda:
                data, targetpose, targetpaf = data.cuda(), targetpose.cuda(), targetpaf.cuda()
            data, targetpose, targetpaf = Variable(data), Variable(targetpose), Variable(targetpaf)
            
            
            #resizing groundtruth
            
            # targetpose = ground_resize(targetpose)
            # targetpaf = ground_resize(targetpaf)
            
            # print targetpose.size()
            m = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)
            mp = nn.ZeroPad2d(99)
            targetpose = m(m(m(mp(targetpose))))
            targetpaf = m(m(m(mp(targetpaf))))
            # targetpose = m(m(m(targetpose)))
            # targetpaf = m(m(m(targetpaf)))

            # print(targetpose.size())
            
            #optimizer
            self.optim.zero_grad()
            
            scorepose1, scorepaf1, scorepose2, scorepaf2 = self.model(data)
            
            # print(scorepose1.size(),scorepose2.size())
            
            losspose1 = pose_loss(scorepose1, targetpose,
                                   size_average=self.size_average)
            losspaf1 = paf_loss(scorepaf1, targetpaf,
                                   size_average=self.size_average)
            losspose2 = pose_loss(scorepose2, targetpose,
                                   size_average=self.size_average)
            losspaf2 = paf_loss(scorepaf2, targetpaf,
                                   size_average=self.size_average)
            
            
            # losspose = pose_loss_test(scorepose, targetpose,
                                   # size_average=self.size_average)
            # losspaf = paf_loss_test(scorepaf, targetpaf,
                                   # size_average=self.size_average)
            # loss = losspose1+losspaf1+losspose2+losspaf2
            wtpose = 14*scorepose2.size()[2]*scorepose2.size()[3]/2 
            wtpaf = 26*scorepaf2.size()[2]*scorepaf2.size()[3]/2
            losspose = losspose1+losspose2
            losspaf = losspaf1+losspaf2
            loss = losspose*wtpose+losspaf*wtpaf
        
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            loss /= len(data)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            ravg.append(loss.data[0])
            poseavg.append(losspose.data[0])
            pafavg.append(losspaf.data[0])
            self.writer.add_scalars('train_loss', {"tot" : loss.data[0],
                                                  "pose" : losspose.data[0],
                                                  "paf" : losspaf.data[0],}, self.iteration)

            # print(loss.data[0])
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            #____________________________Visualise_______________________
            if len(vis_pose) < train_vis_param:
                imgs = data.data.cpu()
                pose_pred = scorepose2.data.cpu().numpy()
                paf_pred = scorepaf2.data.cpu().numpy()
                pose_lab = targetpose.data.cpu().numpy()
                paf_lab = targetpaf.data.cpu().numpy()

                for img, lt_po, lp_po ,lt_pf ,lp_pf in zip(imgs, pose_lab, pose_pred, paf_lab, paf_pred):
                    img = self.val_loader.dataset.untransform(img)
                    lt_po = np.max(lt_po,axis=0)
                    lp_po = np.max(lp_po,axis=0)
                    lt_pf = np.max(np.abs(lt_pf),axis=0)
                    lp_pf = np.max(np.abs(lp_pf),axis=0)
                    lt_po = scipy.misc.imresize(lt_po, (img.shape[0],img.shape[1]))
                    lt_pf = scipy.misc.imresize(lt_pf, (img.shape[0],img.shape[1]))
                    lp_po = scipy.misc.imresize(lp_po, (img.shape[0],img.shape[1]))
                    lp_pf = scipy.misc.imresize(lp_pf, (img.shape[0],img.shape[1]))
                    # mask_po = lt_po > 0.02
                    # lp_po = lp_po*mask_po
                    viz = visualize_(
                        lbl_pred=lp_po, lbl_true=lt_po, img=img)
                    vis_pose.append(viz)

                    # mask_pf = lt_pf != 0
                    # lp_pf = lp_pf*mask_pf
                    viz = visualize_(
                        lbl_pred=lp_pf, lbl_true=lt_pf, img=img)
                    vis_paf.append(viz)

    
            
#             if len(vis_paf) < train_vis_param:
#                 for img, lt, lp in zip(imgs, paf_lab, paf_pred):
#                     img, lt = self.val_loader.dataset.untransform(img, lt)
#                     img = scipy.misc.imresize(img, (lt.shape[0],lt.shape[1]))

#                     if len(vis_paf) < train_vis_param:
#                         viz = fcn.utils.visualize_segmentation(
#                             lbl_pred=lp, lbl_true=lt, img=img, n_class=26)
#                         vis_paf.append(viz)
            
            #_________________ Logging and metric calculation ______________
            if (self.iteration % 10)==0:
                metrics = []
                pose_pred = scorepose2.data.cpu().numpy()
                paf_pred = scorepaf2.data.cpu().numpy()

                for posep, pafp, patht in zip(pose_pred,paf_pred,targetpath):
                    # metout = paf_pose_metric(posep, pafp, patht)
                    metout = 0
                    # print(metout)
                    metrics.append(metout)
                metrics = np.mean(metrics)
                # metrics = pose_metric(scorepose.data, targetpose.data)
                self.writer.add_scalar('train_metric', metrics, self.iteration)
                rmet.append(metrics)
            # else:
            #     metrics = ''
            
                with open(osp.join(self.out, 'log.csv'), 'a') as f:
                    elapsed_time = (
                        datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                        self.timestamp_start).total_seconds()
                    log = [self.epoch, self.iteration] + [loss.data[0]] + [losspose.data[0]] + [losspaf.data[0]] + [metrics] + ['']*4 + [elapsed_time]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')
            
            #_________________ Tensorboard writing ______________
            
            if (self.iteration % 100)==0:
                if len(vis_pose)==train_vis_param:
                    self.writer.add_image('Image_pose_train', fcn.utils.get_tile_image(vis_pose), self.iteration)
                    self.writer.add_image('Image_paf_train', fcn.utils.get_tile_image(vis_paf), self.iteration)
                    # vis_pose, viz_paf = [], []
                    # self.writer.add_scalars('train', {"avg_met": np.mean(rmet),
                    #                                  "avg_loss": np.mean(ravg)}, self.iteration)
                self.writer.add_scalar('train_avg_met', np.mean(rmet), self.iteration)
                self.writer.add_scalars('train_avg_loss', {"total" : np.mean(ravg),
                                                          "pose" : np.mean(poseavg),
                                                          "paf" : np.mean(pafavg),}, self.iteration)
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.iteration)
                    # self.writer.add_histogram(name+"/grad", param.grad.clone().cpu().data.numpy(), self.iteration)
                # ravg = []
                # rmet = []
                # self.writer.add_scalar('train_metric', metrics, self.iteration)
            
            if self.iteration >= self.max_iter:
                break
                

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            # adjust_lr(self.optim,self.epoch,self.gamma,self.lr)
            self.train_epoch()
            self.scheduler.step()
            if self.iteration >= self.max_iter:
                break
