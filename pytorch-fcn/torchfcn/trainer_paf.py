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

def paf_pose_metric(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    return 0
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
    log_p = F.log_softmax(input)
    
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
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
    mask = target > 0.02
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
    mask = target > 0.02
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
    
    input  = input*mask
    target  = target*mask
    loss = torch.sum((target - input)**2)
    # number = torch.nonzero(mask.data).size(0)
    # loss/=number
    # mask_ = 1. - mask
    # loss2 = torch.sum(((input*mask_)**2))
    # loss_= loss + loss2#/number
    return loss

class TrainerPAF_T(object):

    def __init__(self, cuda, model, optimizer,scheduler,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        self.cuda = cuda
        tboarddir = os.path.join('/new_data/gpu/ayushya/tblogs/PAF_test/adam_loss', datetime.datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
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
            'train/met',
            'valid/loss',
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

        val_loss = 0
        valid_vis_param = 10
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
            
            scorepose, scorepaf = self.model(data)
            
            losspose = pose_loss(scorepose,targetpose)
            losspaf = paf_loss(scorepaf,targetpaf)

            # losspose = pose_loss_test(scorepose, targetpose,
                                   # size_average=self.size_average)
            # losspaf = paf_loss_test(scorepaf, targetpaf,
                                   # size_average=self.size_average)
            loss = losspose+losspaf

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            
            val_loss += float(loss.data[0]) / len(data)

            # val_loss /= len(self.val_loader)

            
            #_______________Visualise_______________________
            if len(vis_pose) < valid_vis_param:
                imgs = data.data.cpu()
                pose_pred = scorepose.data.cpu().numpy()
                paf_pred = scorepaf.data.cpu().numpy()
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
#             metrics = []
#             # pose_pred = scorepose.data.cpu().numpy()
#             # paf_pred = scorepaf.data.cpu().numpy()
            
#             # for posep, pafp, patht in zip(pose_pred,paf_pred,targetpath):
#                 # metout = paf_pose_metric(posep, pafp, patht)
#             metout = pose_metric(scorepose.data, targetpose.data)
#                 # print(metout)
#                 # metrics.append((metout,))
#             # metls.append(np.mean(metrics, axis=0))
#             metls.append(metout)
#         # """Out of the forward loop"""
        
            if (self.iteration % 10)==0:
                metrics = []
                pose_pred = scorepose.data.cpu().numpy()
                paf_pred = scorepaf.data.cpu().numpy()

                for posep, pafp, patht in zip(pose_pred,paf_pred,targetpath):
                    metout = paf_pose_metric(posep, pafp, patht)
                    # print(metout)
                    metrics.append(metout)
                metrics = np.mean(metrics)
                # metrics = pose_metric(scorepose.data, targetpose.data)
                self.writer.add_scalar('train_metric', metrics, self.iteration)
                rmet.append(metrics)
        
            
            
            
            #___________________ mteric and logging ______________________
            # else:
            #     metrics = ''
            
                # with open(osp.join(self.out, 'log.csv'), 'a') as f:
                #     elapsed_time = (
                #         datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                #         self.timestamp_start).total_seconds()
                #     log = [self.epoch, self.iteration] + [loss.data[0]] + [metrics] + ['']*2 + [elapsed_time]
                #     log = map(str, log)
                #     f.write(','.join(log) + '\n')
            
        val_loss /= len(self.val_loader)
        
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - \
                self.timestamp_start
            log = [self.epoch, self.iteration] + ['']*2 + \
                  [val_loss] + [np.mean(metls, axis=0)] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        
        self.writer.add_image('Image_pose', fcn.utils.get_tile_image(vis_pose), self.iteration)
        self.writer.add_image('Image_paf', fcn.utils.get_tile_image(vis_paf), self.iteration)
        self.writer.add_scalars('valid', {"loss": val_loss,
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
        ravg = []
        rmet = []
        for batch_idx, (data, targetpose, targetpaf, targetpath) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0 and self.iteration>1:
                self.validate()

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
            # print targetpose.size()
            
            #optimizer
            self.optim.zero_grad()
            scorepose, scorepaf = self.model(data)

            # print scorepose.size(),targetpose.size()
            # print scorepaf.size(),targetpaf.size()
            
            losspose = pose_loss(scorepose, targetpose,
                                   size_average=self.size_average)
            losspaf = paf_loss(scorepaf, targetpaf,
                                   size_average=self.size_average)
            
            # losspose = lossTEST(scorepose, targetpose)
            # losspaf = lossTEST(scorepaf, targetpaf)
            loss = losspose+losspaf
            loss /= len(data)
            
            ravg.append(loss.data[0])

            self.writer.add_scalars('train_loss',{"total_loss":loss.data[0],
                                                 "paf_loss": losspaf.data[0],
                                                 "pos_loss": losspose.data[0]} , self.iteration)
            # self.writer.add_scalars('train', {"avg_met": np.mean(rmet),
                    #                                  "avg_loss": np.mean(ravg)}, self.iteration)
                
            # print(loss.data[0])
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            #____________________________Visualise_______________________
            if len(vis_pose) < train_vis_param:
                imgs = data.data.cpu()
                pose_pred = scorepose.data.cpu().numpy()
                paf_pred = scorepaf.data.cpu().numpy()
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
                pose_pred = scorepose.data.cpu().numpy()
                paf_pred = scorepaf.data.cpu().numpy()

                for posep, pafp, patht in zip(pose_pred,paf_pred,targetpath):
                    metout = paf_pose_metric(posep, pafp, patht)
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
                    log = [self.epoch, self.iteration] + [loss.data[0]] + [metrics] + ['']*2 + [elapsed_time]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')
            
            #_________________ Tensorboard writing ______________
            
            if (self.iteration % 100)==0:
                if len(vis_pose)==train_vis_param:
                    self.writer.add_image('Image_pose_train', fcn.utils.get_tile_image(vis_pose), self.iteration)
                    self.writer.add_image('Image_paf_train', fcn.utils.get_tile_image(vis_paf), self.iteration)
                    vis_pose, vis_paf = [], []
                    # self.writer.add_scalars('train', {"avg_met": np.mean(rmet),
                    #                                  "avg_loss": np.mean(ravg)}, self.iteration)
                self.writer.add_scalar('train_avg_met', np.mean(rmet), self.iteration)
                self.writer.add_scalar('train_avg_loss', np.mean(ravg), self.iteration)
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.iteration)
                    self.writer.add_histogram(name+"/grad", param.grad.clone().cpu().data.numpy(), self.iteration)
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
            self.scheduler.step()
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
