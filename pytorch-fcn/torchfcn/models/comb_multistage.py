import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
            
            
class COMB_MULT(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn32s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vM2oya3k0Zlgtekk',
            path=cls.pretrained_model,
            md5='8acf386d722dc3484625964cbe2aba49',
        )

    def __init__(self, n_seg_class=7,n_pos_class = 14,n_paf_class = 26):
        super(COMB_MULT, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        #pose
        
        self.poseconv1 = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.poserelu1 = nn.ReLU(True)
        self.poseconv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.poserelu2 = nn.ReLU(True)
        self.poseconv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.poserelu3 = nn.ReLU(True)
        self.poseconv4 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.poserelu4 = nn.ReLU(True)
        self.poseconv5 = nn.Conv2d(512, n_pos_class, kernel_size=(1, 1), stride=(1, 1))

        #paf
        self.pafconv1 = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pafrelu1 = nn.ReLU(True)
        self.pafconv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pafrelu2 = nn.ReLU(True)
        self.pafconv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pafrelu3 = nn.ReLU(True)
        self.pafconv4 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.pafrelu4 = nn.ReLU(True)
        self.pafconv5 = nn.Conv2d(512, n_paf_class, kernel_size=(1, 1), stride=(1, 1))


        #pose st2
        
        self.poseconv1_st2 = nn.Conv2d(512+n_pos_class+n_paf_class+n_seg_class, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.poserelu1_st2 = nn.ReLU(True)
        self.poseconv2_st2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.poserelu2_st2 = nn.ReLU(True)
        self.poseconv3_st2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.poserelu3_st2 = nn.ReLU(True)
        self.poseconv4_st2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.poserelu4_st2 = nn.ReLU(True)
        self.poseconv5_st2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.poserelu5_st2 = nn.ReLU(True)
        self.poseconv6_st2 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.poserelu6_st2 = nn.ReLU(True)
        self.poseconv7_st2 = nn.Conv2d(512, n_pos_class, kernel_size=(1, 1), stride=(1, 1))

        #paf st2
        self.pafconv1_st2 = nn.Conv2d(512+n_pos_class+n_paf_class+n_seg_class, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.pafrelu1_st2 = nn.ReLU(True)
        self.pafconv2_st2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.pafrelu2_st2 = nn.ReLU(True)
        self.pafconv3_st2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.pafrelu3_st2 = nn.ReLU(True)
        self.pafconv4_st2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.pafrelu4_st2 = nn.ReLU(True)
        self.pafconv5_st2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.pafrelu5_st2 = nn.ReLU(True)
        self.pafconv6_st2 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.pafrelu6_st2 = nn.ReLU(True)
        self.pafconv7_st2 = nn.Conv2d(512, n_paf_class, kernel_size=(1, 1), stride=(1, 1))

        #conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_seg_class, 1)
        
       
        #seg_st2
        self.segconv1_st2 = nn.Conv2d(512+n_pos_class+n_paf_class+n_seg_class, 512, 3, padding=1)
        self.segrelu1_st2 = nn.ReLU(inplace=True)
        self.segconv2_st2 = nn.Conv2d(512, 512, 3, padding=1)
        self.segrelu2_st2 = nn.ReLU(inplace=True)
        self.segconv3_st2 = nn.Conv2d(512, 512, 3, padding=1)
        self.segrelu3_st2 = nn.ReLU(inplace=True)
        self.segpool_st2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        self.segfc1_st2 = nn.Conv2d(512, 512, 7)
        self.segrelufc1_st2 = nn.ReLU(inplace=True)
        self.segdrop1_st2 = nn.Dropout2d()
        self.segfc2_st2 = nn.Conv2d(512, 512, 1)
        self.segrelufc2_st2 = nn.ReLU(inplace=True)
        self.segdrop2_st2 = nn.Dropout2d()
        self.segscore_fr = nn.Conv2d(512, n_seg_class, 1)
        
        self.upscore2 = nn.ConvTranspose2d(
            n_seg_class, n_seg_class, 8, stride=4, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_seg_class, n_seg_class, 16, stride=8, bias=False)
        self.upscore = nn.ConvTranspose2d(n_seg_class, n_seg_class, 64, stride=32,
                                          bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight)
                if m.bias is not None:
                    init.constant(m.bias, 0.1)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def crop_and_concat(self, upsampled, bypass, crop=True):
        if crop:
            c1 = (bypass.size()[2] - upsampled.size()[2]) // 2
            c2 = (bypass.size()[3] - upsampled.size()[3]) // 2
            c3 = (bypass.size()[2] - upsampled.size()[2] - c1)
            c4 = (bypass.size()[3] - upsampled.size()[3] - c2)
            bypass = F.pad(bypass, (-c2, -c4, -c1, -c3))
        # print(upsampled.size(),bypass.size())
        return torch.cat((upsampled, bypass), 1)            
    
    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        
        pose = self.poserelu1(self.poseconv1(h))
        pose = self.poserelu2(self.poseconv2(pose))
        pose = self.poserelu3(self.poseconv3(pose))
        pose = self.poserelu4(self.poseconv4(pose))
        pose = self.poseconv5(pose)
        # pose = pose[:, :, 9:9 + np.ceil(x.size()[2]/8).astype(np.int64), 9:9 + np.ceil(x.size()[3]/8).astype(np.int64)].contiguous()

        paf = self.pafrelu1(self.pafconv1(h))
        paf = self.pafrelu2(self.pafconv2(paf))
        paf = self.pafrelu3(self.pafconv3(paf))
        paf = self.pafrelu4(self.pafconv4(paf))
        paf = self.pafconv5(paf)
        # paf = paf[:, :, 9:9 + np.ceil(x.size()[2]/8).astype(np.int64), 9:9 + np.ceil(x.size()[3]/8).astype(np.int64)].contiguous()

        seg = self.relu5_1(self.conv5_1(h))
        seg = self.relu5_2(self.conv5_2(seg))
        seg = self.relu5_3(self.conv5_3(seg))
        seg = self.pool5(seg)

        seg = self.relu6(self.fc6(seg))
        seg = self.drop6(seg)

        seg = self.relu7(self.fc7(seg))
        seg = self.drop7(seg)

        seg = self.score_fr(seg)
        segmid = self.upscore2(seg)
        seg = self.upscore8(segmid)
        seg = seg[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        
        h1 = torch.cat([h, pose, paf], 1)
        h1 = self.crop_and_concat(h1,segmid)
        pose2 = self.poserelu1_st2(self.poseconv1_st2(h1))
        pose2 = self.poserelu2_st2(self.poseconv2_st2(pose2))
        pose2 = self.poserelu3_st2(self.poseconv3_st2(pose2))
        pose2 = self.poserelu4_st2(self.poseconv4_st2(pose2))
        pose2 = self.poserelu5_st2(self.poseconv5_st2(pose2))
        pose2 = self.poserelu6_st2(self.poseconv6_st2(pose2))
        pose2 = self.poseconv7_st2(pose2)
        # pose = pose[:, :, 9:9 + np.ceil(x.size()[2]/8).astype(np.int64), 9:9 + np.ceil(x.size()[3]/8).astype(np.int64)].contiguous()

        paf2 = self.pafrelu1_st2(self.pafconv1_st2(h1))
        paf2 = self.pafrelu2_st2(self.pafconv2_st2(paf2))
        paf2 = self.pafrelu3_st2(self.pafconv3_st2(paf2))
        paf2 = self.pafrelu4_st2(self.pafconv4_st2(paf2))
        paf2 = self.pafrelu5_st2(self.pafconv5_st2(paf2))
        paf2 = self.pafrelu6_st2(self.pafconv6_st2(paf2))
        paf2 = self.pafconv7_st2(paf2)
        # paf = paf[:, :, 9:9 + np.ceil(x.size()[2]/8).astype(np.int64), 9:9 + np.ceil(x.size()[3]/8).astype(np.int64)].contiguous()

        seg2 = self.segrelu1_st2(self.segconv1_st2(h1))
        seg2 = self.segrelu2_st2(self.segconv2_st2(seg2))
        seg2 = self.segrelu3_st2(self.segconv3_st2(seg2))
        seg2 = self.segpool_st2(seg2)

        seg2 = self.segrelufc1_st2(self.segfc1_st2(seg2))
        seg2 = self.segdrop1_st2(seg2)

        seg2 = self.segrelufc2_st2(self.segfc2_st2(seg2))
        seg2 = self.segdrop2_st2(seg2)

        seg2 = self.segscore_fr(seg2)
        seg2 = self.upscore(seg2)
        seg2 = seg2[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        
        
        return pose, paf, pose2, paf2, seg, seg2

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
            
