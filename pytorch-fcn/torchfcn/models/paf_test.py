import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
            
class PAF_test(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn32s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vM2oya3k0Zlgtekk',
            path=cls.pretrained_model,
            md5='8acf386d722dc3484625964cbe2aba49',
        )

    def __init__(self, n_class=21):
        super(PAF_test, self).__init__()
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
        # self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        #pose
        
        self.poseconv1 = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.poserelu1 = nn.ReLU(True)
        self.poseconv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.poserelu2 = nn.ReLU(True)
        self.poseconv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.poserelu3 = nn.ReLU(True)
        self.poseconv4 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.poserelu4 = nn.ReLU(True)
        self.poseconv5 = nn.Conv2d(512, 14, kernel_size=(1, 1), stride=(1, 1))

        #paf
        self.pafconv1 = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pafrelu1 = nn.ReLU(True)
        self.pafconv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pafrelu2 = nn.ReLU(True)
        self.pafconv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pafrelu3 = nn.ReLU(True)
        self.pafconv4 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
        self.pafrelu4 = nn.ReLU(True)
        self.pafconv5 = nn.Conv2d(512, 26, kernel_size=(1, 1), stride=(1, 1))

        
#         #conv5
#         self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
#         self.relu5_1 = nn.ReLU(inplace=True)
#         self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
#         self.relu5_2 = nn.ReLU(inplace=True)
#         self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
#         self.relu5_3 = nn.ReLU(inplace=True)
#         self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32


        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    init.constant(m.bias, 0.1)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

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

        
#         h = self.relu6(self.fc6(h))
#         h = self.drop6(h)

#         h = self.relu7(self.fc7(h))
#         h = self.drop7(h)

#         h = self.score_fr(h)

#         h = self.upscore(h)
#         h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return pose, paf

    def copy_params_from_vgg16(self, vgg16):
        # return
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
            # self.pool4,        
            # self.conv5_1, self.relu5_1,
            # self.conv5_2, self.relu5_2,
            # self.conv5_3, self.relu5_3,
            # self.pool5,
        ]
        for i in xrange(23):
            l1 = vgg16.features[i]
            l2 = features[i]
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
