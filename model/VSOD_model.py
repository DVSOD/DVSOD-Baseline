import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from torch.nn.parameter import Parameter
from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from model.fusion import fusion



class SODModel(nn.Module):
    def __init__(self, MedChannel=32, isTrain=True, is_ResNet=True):
        super(SODModel, self).__init__()
        self.training = isTrain
        self.ResNet = is_ResNet

        if self.ResNet:
            self.model_rgb = CPD_ResNet(channel=MedChannel)
            self.model_depth = CPD_ResNet(channel=MedChannel)
            self.model_fusion = fusion()
        else:
            self.model_rgb = CPD_VGG(channel=MedChannel)
            self.model_depth = CPD_VGG(channel=MedChannel)
            print('Loading VGG Error')
            exit(2)
        self.conv_fuse = nn.Conv2d(MedChannel * 4, MedChannel * 2, kernel_size=1)
        self.conv_fuse_out = nn.Conv2d(MedChannel * 2, 1, kernel_size=1)

        if self.training:
            self._init_weight()

    def forward(self, rgb, depth):
        x = rgb
        dx = depth[:, :1, ...]
        dx = torch.cat((dx, dx, dx), dim=1)

        # RGB Stream
        Att_r, Out_r, x3_r, x4_r, x5_r = self.model_rgb(x)
        # Depth Stream
        Att_d, Out_d, x3_d, x4_d, x5_d = self.model_depth(dx)
        # RGB-D Fusion
        Att, Pred, fea = self.model_fusion(x3_r, x4_r, x5_r, x3_d, x4_d, x5_d)
        MedOut = (Att_r, Out_r, Att_d, Out_d)

        return MedOut, Att, Pred, fea

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SimpleFusion(nn.Module):
    def __init__(self, stm_val_dim):
        super(SimpleFusion, self).__init__()

        self.conv_layer_1 = nn.Conv2d(stm_val_dim,  stm_val_dim * 2, kernel_size=1)
        self.conv_layer_2 = nn.Conv2d(stm_val_dim,  stm_val_dim * 2, kernel_size=1)
        self.conv_layer_3 = nn.Conv2d(stm_val_dim * 2, stm_val_dim * 2, kernel_size=1)
        self.conv_layer_4 = nn.Conv2d(stm_val_dim * 2, stm_val_dim * 2, kernel_size=1)

        self.bn = nn.BatchNorm2d(stm_val_dim * 2)

    def forward(self, mem, vQ):
        in_1 = vQ
        in_2 = mem

        fused_mem = torch.cat([in_1, in_2], dim=1)

        att_1 = self.conv_layer_3(fused_mem)
        att_2 = self.conv_layer_4(fused_mem)

        out_1 = self.conv_layer_1(in_1)
        out_2 = self.conv_layer_2(in_2)

        out_1 *= torch.sigmoid(att_1)
        out_2 *= torch.sigmoid(att_2)

        fused_mem = out_1 + out_2
        fused_mem = F.relu(self.bn(fused_mem), inplace=True)

        return fused_mem

class Memory(nn.Module):
    def __init__(self, args):
        super(Memory, self).__init__()
        self.learnable_constant = True
        if self.learnable_constant:
            self.const = nn.Parameter(torch.zeros(1))

    def forward(self, m_in, m_out, q_in):  # m_in: o,c,t,h,w
        # o = batch of objects = num objects.
        # d is the dimension, number of channels, t is time
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.view(B, D_e, H*W)  # b, emb, HW

        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        if self.learnable_constant:
            p = torch.cat([p, self.const.view(1, 1, 1).expand(B, -1, H*W)], dim=1)
        p = F.softmax(p, dim=1) # b, THW, HW
        if self.learnable_constant:
            p = p[:, :-1, :]
        # For visualization later
        p_volume = None
        # p_volume = p.view(B, T, H, W, H, W)

        mo = m_out.view(B, D_o, T*H*W)
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        return mem, p, p_volume

class MemoryQueue():

    def __init__(self, args):
        self.queue_size = args.stm_queue_size
        self.queue_keys = []
        self.queue_vals = []
        self.queue_idxs = []

    def reset(self):
        self.queue_keys = []
        self.queue_vals = []
        self.queue_idxs = []

    def current_size(self):
        return len(self.queue_keys)

    def update(self, key, val, idx):
        self.queue_keys.append(key)
        self.queue_vals.append(val)
        self.queue_idxs.append(idx)

        if len(self.queue_keys) > self.queue_size:
            self.queue_keys.pop(0)
            self.queue_vals.pop(0)
            self.queue_idxs.pop(0)

    def get_indices(self):
        return self.queue_idxs

    def get_keys(self):
        return torch.stack(self.queue_keys, dim=2)

    def get_vals(self):
        return torch.stack(self.queue_vals, dim=2)

    def get_fea(self):
        return self.queue_vals

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim, val_pass):
        super(KeyValue, self).__init__()
        self.Key = nn.Sequential(
            nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1),
            nn.BatchNorm2d(keydim),
            nn.ReLU(inplace=True))
        self.val_pass = val_pass
        if not self.val_pass:
            self.Value = nn.Sequential(
                nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1),
                nn.BatchNorm2d(valdim),
                nn.ReLU(inplace=True))

    def forward(self, x):
        val = F.relu(x) if self.val_pass else self.Value(x)
        return self.Key(x), val

class Adaptive_layer(nn.Module):
    def __init__(self, in_dim):
        super(Adaptive_layer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True))

    def forward(self, f_fusion):
        return self.conv_layer(f_fusion)

class RGBDVSODModel(nn.Module):
    def __init__(self, args, MedChannel=32, isTrain=True, is_ResNet=True):
        super(RGBDVSODModel, self).__init__()
        self.training = isTrain
        self.ResNet = is_ResNet
        self.baseline_mode = args.baseline_mode
        self.stm_queue_size = args.stm_queue_size
        self.sample_rate = args.sample_rate

        self.SaliencyModel = SODModel(MedChannel=MedChannel, isTrain=self.training, is_ResNet=self.ResNet)

        fea_dim = MedChannel*2
        stm_val_dim = MedChannel
        stm_key_dim = MedChannel // 2
        val_pass = False

        self.kv_M_r4_fusion = KeyValue(fea_dim, keydim=stm_key_dim, valdim=stm_val_dim, val_pass=val_pass)
        self.kv_Q_r4_fusion = KeyValue(fea_dim, keydim=stm_key_dim, valdim=stm_val_dim, val_pass=val_pass)

        self.memory_module_F = Memory(args)
        self.memory_queue_fusion = MemoryQueue(args)
        self.memory_fusion_F = SimpleFusion(stm_val_dim)
        self.aggregate_F = Adaptive_layer(fea_dim)

        self.knum = 1
        self.predict_Sal = nn.Sequential(
            nn.Conv2d(fea_dim * self.knum, fea_dim, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(fea_dim, 1, kernel_size=1))
        self.Up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        if self.training:
            self._init_weight()

    def memory_range(self, seq_len, flag='all'):
        if flag == "all":
            ret_range = range(seq_len)
        if flag == "random":
            ret_range = random.sample(range(seq_len - 1), self.stm_queue_size - 1)
            ret_range.append(seq_len - 1)  # ensure the last frame is label
            ret_range.sort()

        assert (seq_len - 1 in ret_range)
        return ret_range



    def forward(self, rgb, depth):
        # Input (rgb) has to be of size (batch_size, seq_len, channels, h, w)
        seq_len = rgb.size(1)
        memory_range = self.memory_range(seq_len)


        for t in memory_range:  # [0, 1, 2, 3]
            input_size = rgb[:, t, :, :, :].size()
            h = int(input_size[2])
            w = int(input_size[3])

            if self.baseline_mode:
                if not (t == seq_len -1):
                    continue
                MedOut, Att, Pred, Finer_fusedFea = self.SaliencyModel(rgb[:, t, :, :, :], depth[:, t, :, :, :])
                fea_F = Finer_fusedFea
                continue

            # Multimodal Fusion
            MedOut, Att, Pred, Finer_fusedFea = self.SaliencyModel(rgb[:, t, :, :, :], depth[:, t, :, :, :])

            if not (t == seq_len -1):       # Memory Load except Last Frame
                kM_fusion, vM_fusion = self.kv_M_r4_fusion.forward(Finer_fusedFea)
                idx = t if seq_len != 1 else exit(1)
                self.memory_queue_fusion.update(kM_fusion, vM_fusion, idx)
            else:
                if self.memory_queue_fusion.current_size() == 0:
                    exit(1)
                else:
                    # Temporal Fusion
                    kQ_fusion, vQ_fusion = self.kv_Q_r4_fusion.forward(Finer_fusedFea)
                    mem_f_f, _, _ = self.memory_module_F.forward(
                        self.memory_queue_fusion.get_keys(), self.memory_queue_fusion.get_vals(), kQ_fusion)
                    fused_mem_f_f = self.memory_fusion_F(mem_f_f, vQ_fusion)
                    fea_F = self.aggregate_F(fused_mem_f_f)

        Outputs = self.Up(self.predict_Sal(fea_F))
        (Att_r, Out_r, Att_d, Out_d) = MedOut
        return (Att_r, Out_r, Att_d, Out_d), (Att, Pred), Outputs

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()