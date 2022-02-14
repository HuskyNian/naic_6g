import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import timm
import torchvision
from collections import OrderedDict
import gc
import os
import random
mean = 0.5
std = 0.01
# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class SEBlock(nn.Module):
 
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True, padding_mode='circular')
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True, padding_mode='circular')
 
    def forward(self, inputs):
        x = F.avg_pool2d(inputs, 2)
        x = self.down(x)
        x = F.leaky_relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.repeat(1, 1, inputs.size(2), inputs.size(3))
        return inputs * x
    
class CRBlock(nn.Module):
    def __init__(self,ch_nums,norm=True):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, ch_nums, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(ch_nums, ch_nums, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(ch_nums, ch_nums, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, ch_nums, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(ch_nums, ch_nums, [5, 1])),
        ]))
        self.identity = nn.Identity()
        self.norm = norm
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True) 
        if norm:
            self.out_layer = nn.Sequential(
                nn.BatchNorm2d(2),
                nn.LeakyReLU(negative_slope=0.3, inplace=True),
            )
            self.conv1x1 = ConvBN(ch_nums * 2, 2, 1)
            
        else:
            self.conv1x1 = conv3x3(ch_nums*2,2)
            self.out_layer = nn.Sigmoid()
        

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)
        if self.norm:
            out = self.out_layer(out + identity)
        else:
            out = self.out_layer(out)
        return out
    
class MRFBlock(nn.Module):
    def __init__(self,ch_nums=64):
        super(MRFBlock, self).__init__()
        self.path1 = nn.Sequential(
            ConvBN(ch_nums,ch_nums,5),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        self.path2 = nn.Sequential(
            ConvBN(ch_nums,ch_nums,7),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        self.path3 = nn.Sequential(
            ConvBN(ch_nums,ch_nums,9),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        self.conv1x1 = nn.Sequential(
            ConvBN(ch_nums*3,ch_nums,1),
            nn.BatchNorm2d(ch_nums)
        )
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True) 
        
        

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out = torch.cat((out1, out2,out3), dim=1)
        out = self.conv1x1(out)
        out = self.relu(out+identity)
        return out
    
class Decoder(nn.Module):
    B = 2

    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        
        self.decoder = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5 )),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock(32)),
            ("CRBlock8", CRBlock(32,norm=False)),
        ]))
        
        self.fc = nn.Linear(int(feedback_bits // self.B), 32256)
        self.bn2d = nn.BatchNorm2d(2)
        self.sig = nn.Sigmoid()
        
        '''self.mask = nn.Sequential(
            MRFBlock(16),
            MRFBlock(16),
            MRFBlock(16),
        )'''
        
        self.ende_refinement = nn.Sequential(
            nn.Linear(int(self.feedback_bits / self.B), int(self.feedback_bits / self.B)),
            nn.BatchNorm1d(int(self.feedback_bits / self.B)),
            nn.ReLU(True),
            nn.Linear(int(self.feedback_bits / self.B), int(self.feedback_bits / self.B), bias=False),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.dequantize(x)
        out = out.view(-1, int(self.feedback_bits // self.B))
        
        out_error = self.ende_refinement(out)
        out = out + out_error - 0.5
        
        out = self.sig(self.fc(out))
        out = out.view(-1, 2, 126, 128)
        #out = self.bn2d(out)
        #out = (out-mean)/std
        out = self.decoder(out)
        #out = out*std+mean
        return out
    
def get_efficientnet_ns(model_name='tf_efficientnet_b0_ns', pretrained=True):
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.classifier.in_features

    return net, n_features

class Encoder(nn.Module):
    B = 2

    def __init__(self, feedback_bits, efn_name='tf_efficientnet_b0_ns'):
        super(Encoder, self).__init__()
        #self.efn_name = efn_name
        #self.efn, self.feat_dim = get_efficientnet_ns(model_name=efn_name)
        #self.efn.conv_stem.in_channels = 2
        #self.efn.conv_stem.weight = torch.nn.Parameter(self.efn.conv_stem.weight[:, 0:2:, :, :])
        
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(self.feat_dim, int(feedback_bits // self.B))
        self.fc = nn.Linear(32256, int(feedback_bits // self.B))
        self.bn = nn.BatchNorm1d(int(feedback_bits // self.B))
        self.encoder = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5 )),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock(32)),
        ]))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = (x-mean)/std
        #x = self.efn.forward_features(x)
        #x = self.avgpool(x)
        
        x = self.encoder(x)
        
        
        out = torch.flatten(x, 1)

        out = self.bn(self.fc(out))
        out = self.sig(out)
        out = self.quantize(out)

        return out

# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 512 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits, efn_name='tf_efficientnet_b0_ns'):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits, efn_name=efn_name)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out



def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x), -1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x), -1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real ** 2 + x_imag ** 2, axis=1)
    mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
    nmse = mse / power
    return nmse
 
class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction
 
    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score




class DatasetFolder(Dataset):
    def __init__(self, matData, phase='val'):
        self.matdata = matData
        self.phase = phase
        self.data_shape = matData[0].shape
 
    def __getitem__(self, index):
        y = self.matdata[index]
        '''if self.phase == 'train' and random.random() < 0.5:
            y = y[:, :, ::-1].copy()
        if self.phase == 'train' and random.random() < 0.5:
            y = 1 - self.matdata[index]  # 数据中存在类似正交的关系
        if self.phase == 'train' and random.random() < 0.5:
            y =  y = y[::-1, :, :].copy() # 不同时刻数据实虚存在部分相等的情况'''
        return y
 
    def __len__(self):
        return self.matdata.shape[0]
# dataLoader