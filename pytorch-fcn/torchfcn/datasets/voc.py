#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data


class VOCClassSegBase(data.Dataset):

    class_names = np.array([
        'background',
        'head',
        'torso',
        'upper-arm',
        'lower-arm',
        'upper-leg',
        'lower-leg',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    pose_names = np.array(['head',
        'neck',
        'lshou',
        'lelbow',
        'lwrist',
        'lhip',
        'lknee',
        'lankle',
        'rshoul',
        'relb',
        'rwrist',
        'rhip',
        'rknee',
        'rankle',])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class VOC2011ClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', transform=False):
        super(VOC2011ClassSeg, self).__init__(
            root, split=split, transform=transform)
        pkg_root = osp.join(osp.dirname(osp.realpath(__file__)), '..')
        imgsets_file = osp.join(
            pkg_root, 'ext/fcn.berkeleyvision.org',
            'data/pascal/seg11valid.txt')
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})


class VOC2012ClassSeg(VOCClassSegBase):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False):
        super(VOC2012ClassSeg, self).__init__( +
            root, split=split, transform=transform)


class SBDClassSeg(VOCClassSegBase):
    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA
    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(self.root, 'pascal_data/pascal_data/%s_seg.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(self.root, 'VOCdevkit/VOC2010/JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(self.root, 'pascal_data/pascal_data/SegmentationPart/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })


    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = PIL.Image.open(lbl_file)
        lbl =  np.array(mat, dtype=np.uint8)
        # if self._transform:
        return self.transform(img, lbl)
        # else:
        #     return img, lbl

class Seg_test(VOCClassSegBase):

    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        self.test_par = 1.
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(self.root, 'pascal_data/pascal_data/%s_id.txt' % split)
            np.random.seed(1)
            img_id_list = [did.strip() for did in open(imgsets_file)]
            np.random.shuffle(img_id_list)
            img_id_list = img_id_list[:np.ceil(self.test_par*len(img_id_list)).astype(np.int32)]
            for did in img_id_list:
                img_file = osp.join(self.root, 'VOCdevkit/VOC2010/JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(self.root, 'pascal_data/pascal_data/SegmentationPart/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = PIL.Image.open(lbl_file)
        lbl =  np.array(mat, dtype=np.uint8)
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
        
        
class PAF_loader_test(VOCClassSegBase):

    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, root, split='train', transform=False,test_par = 0.1):
        self.root = root
        self.split = split
        self._transform = transform
        self.test_par = test_par
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(self.root, 'pascal_data/pascal_data/%s_idnew.txt' % split)
            np.random.seed(3)
            img_id_list = [did.strip() for did in open(imgsets_file)]
            np.random.shuffle(img_id_list)
            img_id_list = img_id_list[:np.ceil(self.test_par*len(img_id_list)).astype(np.int32)]
            for did in img_id_list:
                img_file = osp.join(self.root, 'VOCdevkit/VOC2010/JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(self.root, 'pascal_data/pascal_data/PersonJoints/%s.mat' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })
    
    def genmat(self,path,image):
        H,W,_ = image.shape
        # Hn = np.ceil(H/8).astype(np.int64)
        # Wn = np.ceil(W/8).astype(np.int64)
        mat = scipy.io.loadmat(path)
        limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]
        out = np.zeros((14,H,W))
        paf = np.zeros((26,H,W))
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        for human in mat['joints'][0]:
            poselist = np.around(human[:,:-1]).astype(np.int64)
            # poselist[:,0] = poselist[:,0]*Hn/H
            # poselist[:,1] = poselist[:,1]*Wn/W
            vis = human[:,2]
            #PAF GT
            for (i,limb) in enumerate(limbs):
                p1 = poselist[limb[0],:]
                p2 = poselist[limb[1],:]
                if not (vis[limb[0]]==0 or vis[limb[1]]==0):
    #             if (np.all(p1>0) and np.all(p2>0)):
                    #APPROX RECON
                    if(np.linalg.norm(p2-p1)==0):
                        # raise ValueError('['+str(limb[0])+','+str(limb[1])+']'+'['+str(human[limb[0]][0])+','+str(human[limb[0]][1])+']'+'['+str(human[limb[1]][0])+','+str(human[limb[1]][1])+']')
                        continue
                    dvec = (p2-p1)/np.linalg.norm(p2-p1)
                    vecx = x - p1[0]
                    vecy = y - p1[1]
                    dot = vecx*dvec[0] + vecy*dvec[1]
                    perp2 = vecx**2+vecy**2-dot**2
                    boolmat = (dot>0) & (dot<np.linalg.norm(p2-p1)) & (perp2<np.linalg.norm(p2-p1)*0.3) #sigma^2
                    paf[2*i][boolmat] = dvec[0]
                    paf[2*i+1][boolmat] = dvec[1]
            #POSE GT
            for (i,pose) in enumerate(poselist):
                if not (vis[i]==0):
                    tmp = np.exp(-((x-pose[0])**2 + (y-pose[1])**2)/(2.0*10.0))
                    out[i] = np.maximum(out[i],tmp)
        return out,paf

    
    def untransform(self, img):#, lbl):
        img = img.numpy()
        # print (img.shape)
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        # lbl = lbl.numpy()
        return img#, lbl

    def transform(self, img, pose, paf, lbl_file):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        pose = torch.from_numpy(pose).float()
        paf = torch.from_numpy(paf).float()
        return img, pose, paf, lbl_file

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        pose, paf = self.genmat(lbl_file,img)
        if self._transform:
            return self.transform(img, pose, paf, lbl_file)
        else:
            return img, pose, paf, lbl_file
        
class Combined_test(VOCClassSegBase):

    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, root, split='train', transform=False,test_par = 0.1):
        self.root = root
        self.split = split
        self._transform = transform
        self.test_par = test_par
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(self.root, 'pascal_data/pascal_data/%s_idnew.txt' % split)
            np.random.seed(3)
            img_id_list = [did.strip() for did in open(imgsets_file)]
            np.random.shuffle(img_id_list)
            img_id_list = img_id_list[:np.ceil(self.test_par*len(img_id_list)).astype(np.int32)]
            for did in img_id_list:
                img_file = osp.join(self.root, 'VOCdevkit/VOC2010/JPEGImages/%s.jpg' % did)
                seg_file = osp.join(self.root, 'pascal_data/pascal_data/SegmentationPart/%s.png' % did)
                pos_file = osp.join(self.root, 'pascal_data/pascal_data/PersonJoints/%s.mat' % did)
                self.files[split].append({
                    'img': img_file,
                    'pos': pos_file,
                    'seg': seg_file,
                })
    
    def genmat(self,path,image):
        H,W,_ = image.shape
        # Hn = np.ceil(H/8).astype(np.int64)
        # Wn = np.ceil(W/8).astype(np.int64)
        mat = scipy.io.loadmat(path)
        limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]
        out = np.zeros((14,H,W))
        paf = np.zeros((26,H,W))
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        for human in mat['joints'][0]:
            poselist = np.around(human[:,:-1]).astype(np.int64)
            # poselist[:,0] = poselist[:,0]*Hn/H
            # poselist[:,1] = poselist[:,1]*Wn/W
            vis = human[:,2]
            #PAF GT
            for (i,limb) in enumerate(limbs):
                p1 = poselist[limb[0],:]
                p2 = poselist[limb[1],:]
                if not (vis[limb[0]]==0 or vis[limb[1]]==0):
    #             if (np.all(p1>0) and np.all(p2>0)):
                    #APPROX RECON
                    if(np.linalg.norm(p2-p1)==0):
                        # raise ValueError('['+str(limb[0])+','+str(limb[1])+']'+'['+str(human[limb[0]][0])+','+str(human[limb[0]][1])+']'+'['+str(human[limb[1]][0])+','+str(human[limb[1]][1])+']')
                        continue
                    dvec = (p2-p1)/np.linalg.norm(p2-p1)
                    vecx = x - p1[0]
                    vecy = y - p1[1]
                    dot = vecx*dvec[0] + vecy*dvec[1]
                    perp2 = vecx**2+vecy**2-dot**2
                    boolmat = (dot>0) & (dot<np.linalg.norm(p2-p1)) & (perp2<np.linalg.norm(p2-p1)*0.3) #sigma^2
                    paf[2*i][boolmat] = dvec[0]
                    paf[2*i+1][boolmat] = dvec[1]
            #POSE GT
            for (i,pose) in enumerate(poselist):
                if not (vis[i]==0):
                    tmp = np.exp(-((x-pose[0])**2 + (y-pose[1])**2)/(2.0*10.0))
                    out[i] = np.maximum(out[i],tmp)
        return out,paf

    
    def untransform(self, img):#, lbl):
        img = img.numpy()
        # print (img.shape)
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        # lbl = lbl.numpy()
        return img#, lbl

    def transform(self, img, pose, paf,seg, lbl_file):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        pose = torch.from_numpy(pose).float()
        paf = torch.from_numpy(paf).float()
        seg = torch.from_numpy(seg).long()
        return img, pose, paf,seg, lbl_file

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        pos_file = data_file['pos']
        pose, paf = self.genmat(pos_file,img)
        
        seg_file = data_file['seg']
        mat = PIL.Image.open(seg_file)
        seg =  np.array(mat, dtype=np.uint8)
        
        if self._transform:
            return self.transform(img, pose, paf,seg, pos_file)
        else:
            return img, pose, paf,seg, pos_file
   
        
        
class PAFloader(VOCClassSegBase):

    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        self.test_par = 0.1

        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(self.root, 'pascal_data/pascal_data/%s_idnew.txt' % split)
            np.random.seed(2)
            img_id_list = [did.strip() for did in open(imgsets_file)]
            np.random.shuffle(img_id_list)
            img_id_list = img_id_list[:np.ceil(self.test_par*len(img_id_list)).astype(np.int32)]
            for did in img_id_list:
                img_file = osp.join(self.root, 'VOCdevkit/VOC2010/JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(self.root, 'pascal_data/pascal_data/PersonJoints/%s.mat' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })
    
    def genmat(self,path,image):
        H,W,_ = image.shape
        # Hn = np.ceil(H/8).astype(np.int64)
        # Wn = np.ceil(W/8).astype(np.int64)
        mat = scipy.io.loadmat(path)
        limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]
        out = np.zeros((14,H,W))
        paf = np.zeros((26,H,W))
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        for human in mat['joints'][0]:
            poselist = np.around(human[:,:-1]).astype(np.int64)
            # poselist[:,0] = poselist[:,0]*Hn/H
            # poselist[:,1] = poselist[:,1]*Wn/W
            vis = human[:,2]
            #PAF GT
            for (i,limb) in enumerate(limbs):
                p1 = poselist[limb[0],:]
                p2 = poselist[limb[1],:]
                if not (vis[limb[0]]==0 or vis[limb[1]]==0):
    #             if (np.all(p1>0) and np.all(p2>0)):
                    #APPROX RECON
                    if(np.linalg.norm(p2-p1)==0):
                        # raise ValueError('['+str(limb[0])+','+str(limb[1])+']'+'['+str(human[limb[0]][0])+','+str(human[limb[0]][1])+']'+'['+str(human[limb[1]][0])+','+str(human[limb[1]][1])+']')
                        continue
                    dvec = (p2-p1)/np.linalg.norm(p2-p1)
                    vecx = x - p1[0]
                    vecy = y - p1[1]
                    dot = vecx*dvec[0] + vecy*dvec[1]
                    perp2 = vecx**2+vecy**2-dot**2
                    boolmat = (dot>0) & (dot<np.linalg.norm(p2-p1)) & (perp2<np.linalg.norm(p2-p1)*0.3) #sigma^2
                    paf[2*i][boolmat] = dvec[0]
                    paf[2*i+1][boolmat] = dvec[1]
            #POSE GT
            for (i,pose) in enumerate(poselist):
                if not (vis[i]==0):
                    tmp = np.exp(-((x-pose[0])**2 + (y-pose[1])**2)/(2.0*10.0))
                    out[i] = np.maximum(out[i],tmp)
        return out,paf

    
    def transform(self, img, pose, paf, lbl_file):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        pose = torch.from_numpy(pose).float()
        paf = torch.from_numpy(paf).float()
        return img, pose, paf, lbl_file

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        pose, paf = self.genmat(lbl_file,img)
        if self._transform:
            return self.transform(img, pose, paf, lbl_file)
        else:
            return img, pose, paf, lbl_file
