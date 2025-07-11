import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def createPDCFunc(PDC_type): 
    assert PDC_type in ['cv', 'cd1', 'ad1', 'rd1', 'cd2', 'ad2', 'rd2'], 'unknown PDC type: %s' % str(PDC_type)
    if PDC_type == 'cv':  
        return F.conv2d

    if PDC_type == 'cd1':  
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2' 
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
   
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)

            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
           
            return y - yc
            

        return func

    elif PDC_type == 'cd2': 
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'  
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = 2 * weights.sum(dim=[2, 3], keepdim=True)
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)  
            weights_conv = (
                    weights[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8]] + weights[:, :, [8, 7, 6, 5, 4, 3, 2, 1, 0]]).view(
                shape) 
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

            return y - yc

        return func

    elif PDC_type == 'ad1': 
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)  
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    elif PDC_type == 'ad2': 
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'
 
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1) 
            weights_conv = (weights[:, :, [1, 0, 5, 6, 4, 2, 3, 8, 7]] + weights[:, :, [3, 2, 1, 0, 4, 8, 7, 6,
                                                                                        5]] - 2 * weights).view(
                shape)  
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    elif PDC_type == 'rd1':  
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)  
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
      
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
 
            buffer[:, :, 12] = 0  
            buffer = buffer.view(shape[0], shape[1], 5, 5)  
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    elif PDC_type == 'rd2': 
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 5 and weights.size(3) == 5, 'kernel size for rd_conv should be 3x3'
            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)  
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, [6, 7, 8, 11, 13, 16, 17, 18]]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -2 * weights[:, :, [6, 7, 8, 11, 13, 16, 17, 18]]
            kernel_c = weights[:, :, [6, 7, 8, 11, 13, 16, 17, 18]].sum()
            buffer[:, :, 12] = kernel_c
            buffer = buffer.view(shape[0], shape[1], 5, 5)  
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    else:
        print('unknown PDC type: %s' % str(PDC_type)) 
        return None


class Conv2d(nn.Module):  
    def __init__(self, pdc_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        """
        :param pdc_func: 卷积函数
        """
        super(Conv2d, self).__init__()
        if in_channels % groups != 0: 
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation 
        self.groups = groups
        #
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc_func = pdc_func

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.pdc_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv33 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=dilation,
                                dilation=dilation, bias=False, groups=in_channels)
        self.relu = nn.ELU()
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                                bias=False)
        self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        o = self.conv33(x)
        o = self.relu(o)
        o = self.conv11(o)
        if self.in_channels != self.out_channels:
            x = self.shortcut(x)
        o = o + x
        return o


class PDC_Block(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type):
        super(PDC_Block, self).__init__()
        self.conv_type = conv_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.mul_weight = mul_weight
        if conv_type[0] == 'cv':
            pdc_func_1 = createPDCFunc('cv')
            self.mul_weight = 0.5
        elif conv_type[0] == 'cd1':
            pdc_func_1 = createPDCFunc('cd1')
        elif conv_type[0] == 'ad1':
            pdc_func_1 = createPDCFunc('ad1')
        elif conv_type[0] == 'rd1':
            pdc_func_1 = createPDCFunc('rd1')
        else:
            print('warning!')
        if conv_type[1] == 'cv':
            pdc_func_2 = createPDCFunc('cv')
        elif conv_type[1] == 'cd2':
            pdc_func_2 = createPDCFunc('cd2')
        elif conv_type[1] == 'ad2':
            pdc_func_2 = createPDCFunc('ad2')
        elif conv_type[1] == 'rd2':
            pdc_func_2 = createPDCFunc('rd2')
        else:
            print('warning!')

        kernel_size = 3
        padding = 1
        channels = int(in_channels / 2)
        o_channels = in_channels if in_channels == out_channels else out_channels
        self.relu1 = nn.ReLU(True)
        self.conv11_1 = nn.Conv2d(in_channels=channels, out_channels=o_channels, kernel_size=1, padding=0,
                                  bias=False)
        self.relu2 = nn.ReLU(True)
        self.conv11_2 = nn.Conv2d(in_channels=channels, out_channels=o_channels, kernel_size=1, padding=0,
                                  bias=False)
        self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=o_channels, kernel_size=1, padding=0,
                                   bias=False)
        self.reset_parameters()

        self.PDC_1 = Conv2d(pdc_func_1, in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                            padding=1, groups=channels, bias=False)
        if conv_type[1] == 'rd2':
            kernel_size = 5
            padding = 2
        self.PDC_2 = Conv2d(pdc_func_2, in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                            padding=padding, groups=channels, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.cal_Weight = cal_weights(out_channels)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x_ = x
        mul_weight = self.cal_Weight(x)[0]
        x = torch.chunk(x, 2, dim=1)

        o1 = self.PDC_1(x[0])
        o1 = self.relu1(o1)
        o1 = self.norm1(o1)
        o1 = self.conv11_1(o1)
        # o1 = o1 + self.shortcut1(x[0])

        o2 = self.PDC_2(x[1])
        o2 = self.relu2(o2)
        o2 = self.norm2(o2)
        o2 = self.conv11_2(o2)
        # o2 = o2 + self.shortcut2(x[1])

        out = mul_weight * o1 + (1 - mul_weight) * o2
        out = out + self.shortcut(x_)


        return out


class make_mask(nn.Module):
    def __init__(self, channels):
        super(make_mask, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return y


class pre_encoder(nn.Module):
    def __init__(self, channels):
        super(pre_encoder, self).__init__()
        self.stage4_up = nn.Conv2d(in_channels=channels * 8, out_channels=channels * 2, kernel_size=1, padding=0,
                                   bias=False)

        # self.stage3_change = nn.Conv2d(in_channels=channels * 4, out_channels=channels * 2, kernel_size=1, padding=0,
        #                                bias=False)
        self.stage3_up = nn.Conv2d(in_channels=channels * 6, out_channels=channels * 2, kernel_size=1, padding=0,
                                   bias=False)

        # self.stage2_change = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, padding=0,
        #                                bias=False)
        self.stage2_up = nn.Conv2d(in_channels=channels * 4, out_channels=channels, kernel_size=1, padding=0,
                                   bias=False)

        # self.stage1_change = nn.Conv2d(in_channels=channels, out_channels=int(channels / 2), kernel_size=1, padding=0,
        #                                bias=False)

        self.reset_parameters()
        self.stage1 = nn.Sequential(ConvBlock(channels, channels),
                                    ConvBlock(channels, channels),
                                    ConvBlock(channels, channels, dilation=2))
        self.stage2 = nn.Sequential(ConvBlock(channels, channels * 2),
                                    ConvBlock(channels * 2, channels * 2),
                                    ConvBlock(channels * 2, channels * 2, dilation=2))
        self.stage3 = nn.Sequential(ConvBlock(channels * 2, channels * 4),
                                    ConvBlock(channels * 4, channels * 4),
                                    ConvBlock(channels * 4, channels * 4, dilation=2))
        self.stage4 = nn.Sequential(ConvBlock(channels * 4, channels * 8),
                                    ConvBlock(channels * 8, channels * 8),
                                    ConvBlock(channels * 8, channels * 8, dilation=2))
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.make_mask1 = make_mask(channels*2)
        self.make_mask2 = make_mask(channels*4)
        self.make_mask3 = make_mask(channels*6)
        # self.make_mask4 = make_mask(channels * 2)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        _, _, h1, w1 = x.shape
        stage1 = self.stage1(x)
        s1 = self.pool1(stage1)
        stage2 = self.stage2(s1)
        _, _, h2, w2 = stage2.shape
        s2 = self.pool2(stage2)
        stage3 = self.stage3(s2)
        _, _, h3, w3 = stage3.shape
        s3 = self.pool3(stage3)
        stage4 = self.stage4(s3)
        # mask4 = self.make_mask4(stage4)

        stage4_up = F.interpolate(stage4, [h3, w3], mode="bilinear", align_corners=False)
        stage4_up = self.stage4_up(stage4_up)
        stage3 = torch.cat([stage3, stage4_up], dim=1)
        mask3 = self.make_mask3(stage3)

        stage3_up = F.interpolate(stage3, [h2, w2], mode="bilinear", align_corners=False)
        stage3_up = self.stage3_up(stage3_up)
        stage2 = torch.cat([stage2, stage3_up], dim=1)
        mask2 = self.make_mask2(stage2)

        stage2_up = F.interpolate(stage2, [h1, w1], mode="bilinear", align_corners=False)
        stage2_up = self.stage2_up(stage2_up)

        mask1 = self.make_mask1(torch.cat([stage1, stage2_up], dim=1))

        return mask1, mask2, mask3#, mask4


class cal_weights(nn.Module):
    def __init__(self, channels):
        super(cal_weights, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        weight = F.softmax(x, dim=1).view(-1)
        return weight


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        channels = 40
        self.tomask = 12
        toencode = channels
        self.input = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1, bias=False, groups=1)
        self._initialize_weights()
        self.preencoder = pre_encoder(self.tomask)
        self.stage1 = nn.Sequential(ConvBlock(toencode, toencode),
                                    PDC_Block(toencode, toencode, ['cd1', 'cd2']),
                                    # ConvBlock(toencode, toencode, dilation=2),
                                    # PDC_Block(toencode, toencode, ['ad1', 'ad2']),
                                    ConvBlock(toencode, toencode, dilation=2),
                                    PDC_Block(toencode, toencode, ['rd1', 'rd2']))

        self.stage2 = nn.Sequential(ConvBlock(toencode, toencode * 2, dilation=2),
                                    PDC_Block(toencode * 2, toencode * 2, ['cd1', 'cd2']),
                                    ConvBlock(toencode * 2, toencode * 2),
                                    # PDC_Block(toencode * 2, toencode * 2, ['ad1', 'ad2']),
                                    # ConvBlock(toencode * 2, toencode * 2),
                                    PDC_Block(toencode * 2, toencode * 2, ['rd1', 'rd2']))

        self.stage3 = nn.Sequential(ConvBlock(toencode * 2, toencode * 4, dilation=2),
                                    PDC_Block(toencode * 4, toencode * 4, ['cd1', 'cd2']),
                                    ConvBlock(toencode * 4, toencode * 4, dilation=2),
                                    # PDC_Block(toencode * 4, toencode * 4, ['ad1', 'ad2']),
                                    # ConvBlock(toencode * 4, toencode * 4, dilation=2),
                                    PDC_Block(toencode * 4, toencode * 4, ['rd1', 'rd2']))

        self.stage4 = nn.Sequential(ConvBlock(toencode * 4, toencode * 4, dilation=2),
                                    PDC_Block(toencode * 4, toencode * 4, ['cd1', 'cd2']),
                                    ConvBlock(toencode * 4, toencode * 4, dilation=2),
                                    # PDC_Block(toencode * 4, toencode * 4, ['ad1', 'ad2']),
                                    # ConvBlock(toencode * 4, toencode * 4, dilation=2),
                                    PDC_Block(toencode * 4, toencode * 4, ['rd1', 'rd2']))

        # self.stage5 = nn.Sequential(ConvBlock(toencode * 6, toencode * 4, dilation=2),
        #                             PDC_Block(toencode * 4, toencode * 4, ['cd1', 'cd2'], mul_weight=0.95),
        #                             # ConvBlock(toencode * 8, toencode * 8),
        #                             PDC_Block(toencode * 4, toencode * 4, ['ad1', 'ad2'], mul_weight=0.95),
        #                             # ConvBlock(toencode * 8, toencode * 8),
        #                             PDC_Block(toencode * 4, toencode * 4, ['rd1', 'rd2'], mul_weight=0.95),
        #                             ConvBlock(toencode * 4, toencode * 4, dilation=4))

        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        # self.maxpool4 = nn.MaxPool2d(2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.input(x)
        mask = self.preencoder(x[:, :self.tomask, :, :])
        #
        input = x * mask[0]
        stage1 = self.stage1(input)

        stage1_ = self.maxpool1(stage1)
        # stage1_ = torch.chunk(stage1_, 2, dim=1)
        _stage1_ = stage1_ * mask[1]
        # stage1_ = torch.cat([_stage1_, stage1_[1]], dim=1)
        # stage1_ = channel_shuffle(stage1_, groups=2)
        stage2 = self.stage2(_stage1_)

        stage2_ = self.maxpool2(stage2)
        # stage2_ = torch.chunk(stage2_, 4, dim=1)
        _stage2_ = stage2_ * mask[2]
        # stage2_ = torch.cat([_stage2_, stage2_[1], stage2_[2], stage2_[3]], dim=1)
        # stage2_ = channel_shuffle(stage2_, groups=4)
        stage3 = self.stage3(_stage2_)

        stage3_ = self.maxpool3(stage3)
        # stage3_ = stage3_ * mask[3]
        stage4 = self.stage4(stage3_)

        # stage4_ = self.maxpool4(stage4)
        # stage5 = self.stage5(stage4_)

        return stage1, stage2, stage3, stage4#, stage5


class CRFEM(nn.Module):
    def __init__(self):
        super(CRFEM, self).__init__()
        # self.conv13 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1), bias=False)
        # self.conv31 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv53 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 3), padding=(4, 2), dilation=2,
                                bias=False)
        self.conv35 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 5), padding=(2, 4), dilation=2,
                                bias=False)
        self.conv_dil1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv_dil2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=4, dilation=4, bias=False)
        self.conv11 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, padding=0)
        self.conv33 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1, bias=False)
        self._initialize_weights()
        self.relu = nn.ELU()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 4:
                    nn.init.constant_(m.weight, 0.25)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # o1 = self.conv31(x)
        # o2 = self.conv13(x)
        o3 = self.conv_dil1(x)
        o4 = self.conv_dil2(x)
        o5 = self.conv53(x)
        o6 = self.conv35(x)
        o = o3 + o4 + o5 + o6 + x
        # o = o1 + o2 + o3 + o4 + o5 + o6 + x
        mask = self.relu(o)
        mask = self.conv11(mask)
        mask = self.conv33(mask)
        mask = mask.sigmoid()

        o = mask * o

        return o


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        channels = 40
        self.layer1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(in_channels=channels * 2, out_channels=8, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(in_channels=channels * 4, out_channels=8, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(in_channels=channels * 4, out_channels=8, kernel_size=3, padding=1)
        # self.layer5 = nn.Conv2d(in_channels=channels * 4, out_channels=8, kernel_size=3, padding=1, bias=False)
        self.stage1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.stage2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.stage3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.stage4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        # self.stage5 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.fusion = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self._initialize_weights()
        self.crfem1 = CRFEM()
        self.crfem2 = CRFEM()
        self.crfem3 = CRFEM()
        self.crfem4 = CRFEM()
        # self.crfem5 = CRFEM()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 4:
                    nn.init.constant_(m.weight, 0.25)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, *input):
        _, _, h, w = input[0].shape
        layer1 = self.relu1(self.layer1(input[0]))
        layer2 = self.relu2(self.layer2(input[1]))
        layer3 = self.relu3(self.layer3(input[2]))
        layer4 = self.relu4(self.layer4(input[3]))
        # layer5 = self.layer5(input[4])

        stage2 = F.interpolate(layer2, [h, w], mode="bilinear", align_corners=False)
        stage3 = F.interpolate(layer3, [h, w], mode="bilinear", align_corners=False)
        stage4 = F.interpolate(layer4, [h, w], mode="bilinear", align_corners=False)
        # stage5 = F.interpolate(layer5, [h, w], mode="bilinear", align_corners=False)

        stage1 = self.stage1(layer1)
        stage2 = self.stage2(stage2)
        stage3 = self.stage3(stage3)
        stage4 = self.stage4(stage4)
        # stage5 = self.stage5(stage5)

        out = self.fusion(torch.cat([stage1, stage2, stage3, stage4], dim=1))
        return out.sigmoid(), stage1.sigmoid(), stage2.sigmoid(), stage3.sigmoid(), stage4.sigmoid()


class Contour_Detection(nn.Module):
    def __init__(self):
        super(Contour_Detection, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x):
        side_outputs = self.encoder(x)
        outputs = self.decoder(*side_outputs)
        return outputs[0], outputs[1:]




from thop import profile


def cal_param(net):
    # model = torch.nn.DataParallel(net)
    inputs = torch.randn([1, 3, 321, 481]).cuda()
    flop, para = profile(net, inputs=(inputs,), verbose=False)
    return 'Flops：' + str(2 * flop / 1000 ** 3) + 'G', 'Params：' + str(para / 1000 ** 2) + 'M'


if __name__ == "__main__":
    # net = pre_encoder(8).cuda()
    net = Contour_Detection().cuda()
    print(cal_param(net))
    # inputs = torch.randn([1, 8, 320, 320])
    # mask = cal_weights(8)
    # print(mask(inputs)[0])
