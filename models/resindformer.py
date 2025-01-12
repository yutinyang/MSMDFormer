import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
import sys
sys.path.append('/root/data2/Projects/ECCV2024/CopyforGCN/Gabor_CNN_PyTorch-master/')
from gcn.layers import GConv
import sys
sys.path.append('/root/data2/Projects/ECCV2024/CopyforGCN/Gabor_CNN_PyTorch-master/demo/')
from net_factoryinsertdformer import Block


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class Mlp(nn.Module):
    "Implementation of MLP"

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Attention(nn.Module):
    def __init__(self, in_places, scale=6, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x, x1):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x1).view(batch_size, -1, width * height)
        K = self.key_conv(x1).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()


class Attention1(nn.Module):
    def __init__(self, out_chan):
        super(Attention1, self).__init__()
        # self.convblk = ConvBnRelu(in_chan, out_chan, ksize=1, stride=1, pad=0)
        # self.convblk1 = ConvBnRelu(12, 32, ksize=1, stride=1, pad=0)
        self.conv_atten = Attention(out_chan)
        # self.conv_atten1 = Attention(12)

    def forward(self, s5, s4):
        # fcat = torch.cat([s5, s4, s3, s2], dim=1)
        # feat = self.convblk(fcat)
        atten = self.conv_atten(s5, s4)
        feat_out = atten + s5
        return feat_out


class ResNetCIFAR(nn.Module):
    """This is a variation of ResNet for CIFAR database.
    Indeed, the network defined in Sec 4.2 performs poorly on CIFAR-100.
    This network is similar to table 1 without the stride and max pooling
    to avoid to reduce too much the input size.

    This modification have been inspired by DenseNet implementation
    for CIFAR databases.
    """

    def __init__(self, layers, num_classes=1000, levels=4):
        block = BasicBlock
        self.inplanes = 64
        super(ResNetCIFAR, self).__init__()

        self.levels = levels
        if (self.levels != 4 and self.levels != 3):
            raise "Impossible to use this number of levels"

        # Same as Densetnet
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0])  # ori=64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        channel = 4
        self.adptivepooling32 = nn.AdaptiveAvgPool2d(32)
        # self.indformer1 = nn.Sequential(
        #     GConv(256, 10, 5, padding=2, stride=1, M=channel, nScale=1, bias=False, expand=True),
        #     nn.BatchNorm2d(10 * channel),
        #     nn.ReLU(inplace=True),
        #     Block(oridim=1, oputdim=120, dim=120, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
        #           attn_drop=0.,
        #           drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratios=[1, 2, 4, 7])
        #
        # )
        self.model1 = nn.Sequential(
            GConv(3, 10, 5, padding=2, stride=1, M=channel, nScale=1, bias=False, expand=True),
            nn.BatchNorm2d(10 * channel),
            nn.ReLU(inplace=True),
            #
            # GConv(10, 20, 5, padding=2, stride=1, M=channel, nScale=2, bias=False),
            # nn.BatchNorm2d(20 * channel),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            #
            # GConv(20, 40, 5, padding=0, stride=1, M=channel, nScale=3, bias=False),
            # nn.BatchNorm2d(40 * channel),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            #
            # GConv(40, 80, 5, padding=0, stride=1, M=channel, nScale=4, bias=False),
            # nn.BatchNorm2d(80 * channel),
            # nn.ReLU(inplace=True),
        )
        self.model2 = nn.Sequential(
            # GConv(1, 10, 5, padding=2, stride=1, M=channel, nScale=1, bias=False, expand=True),
            # nn.BatchNorm2d(10 * channel),
            # nn.ReLU(inplace=True),
            #
            GConv(10, 20, 5, padding=2, stride=1, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(20 * channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # GConv(20, 40, 5, padding=0, stride=1, M=channel, nScale=3, bias=False),
            # nn.BatchNorm2d(40*channel),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2,2),

            # GConv(40, 80, 5, padding=0, stride=1, M=channel, nScale=4, bias=False),
            # nn.BatchNorm2d(80*channel),
            # nn.ReLU(inplace=True),
        )
        self.model3 = nn.Sequential(
            # GConv(1, 10, 5, padding=2, stride=1, M=channel, nScale=1, bias=False, expand=True),
            # nn.BatchNorm2d(10 * channel),
            # nn.ReLU(inplace=True),
            #
            # GConv(10, 20, 5, padding=2, stride=1, M=channel, nScale=2, bias=False),
            # nn.BatchNorm2d(20 * channel),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),

            GConv(20, 40, 5, padding=2, stride=1, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(40 * channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # GConv(40, 80, 5, padding=0, stride=1, M=channel, nScale=4, bias=False),
            # nn.BatchNorm2d(80*channel),
            # nn.ReLU(inplace=True),
        )
        self.model4 = nn.Sequential(
            # GConv(1, 10, 5, padding=2, stride=1, M=channel, nScale=1, bias=False, expand=True),
            # nn.BatchNorm2d(10 * channel),
            # nn.ReLU(inplace=True),
            #
            # GConv(10, 20, 5, padding=2, stride=1, M=channel, nScale=2, bias=False),
            # nn.BatchNorm2d(20 * channel),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            #
            # GConv(20, 40, 5, padding=0, stride=1, M=channel, nScale=3, bias=False),
            # nn.BatchNorm2d(40 * channel),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            #
            GConv(40, 80, 5, padding=2, stride=1, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(80 * channel),
            nn.ReLU(inplace=True),
        )
        self.block1 = Block(oridim=3, oputdim=40, dim=40, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                            drop=0., attn_drop=0.,
                            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratios=[1, 2, 4, 7])
        # self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
        #                  num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
        #                  attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 9, 3], **kwargs)
        self.block2 = Block(oridim=40, oputdim=80, dim=80, num_heads=2, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                            drop=0., attn_drop=0.,
                            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratios=[1, 2, 4, 8])
        self.block3 = Block(oridim=80, oputdim=160, dim=160, num_heads=2, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                            drop=0., attn_drop=0.,
                            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratios=[1, 2])
        self.attentionwave1 = Attention1(512)
        self.convmid1 = nn.Conv2d(80, 512, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        self.convmid11 = nn.Conv2d(160, 512, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.convmid2 = nn.Conv2d(256, 512, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        self.convmid21 = nn.Conv2d(256, 512, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.adaptivepool = nn.AdaptiveAvgPool2d(2)
        inter_channels =160
        out_channels = 1024
        norm_layer = nn.BatchNorm2d
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                                         norm_layer(inter_channels // 2),
                                         nn.ReLU(True),
                                         nn.Dropout2d(0.1, False),
                                         nn.Conv2d(inter_channels // 2, out_channels, 1))
        if self.levels == 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
            self.avgpool4 = nn.AdaptiveAvgPool2d(8)
            # self.fc = nn.Linear(512 * block.expansion, num_classes)
            hidden_dim = 256
            self.fc1 = nn.Linear(512, hidden_dim)
            self.fc = nn.Linear(hidden_dim, num_classes)
            self.mlp1 = Mlp(2048, 512, num_classes)

        else:
            # 3 levels
            # self.avgpool = nn.AvgPool2d(8, stride=1)
            # self.fc = nn.Linear(256 * block.expansion, num_classes)
            hidden_dim = 256
            self.fc1 = nn.Linear(512, hidden_dim)
            self.fc = nn.Linear(hidden_dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(=xx)
        x=self.adptivepooling32(x)
        # xorimark = x
        xori = x
        x = self.model1(x)
        # print(x.shape)
        B, C, H1, W1 = x.shape
        x = x.view(B, H1, W1, C)
        # xori = x.view(B, H1, W1, C)
        # x = x.view(128, 28,28,40)
        # x = self.block1(x, 28, 28)
        x = self.block1(xori, x, 32, 32)
        x = x.view(B, C, H1, W1)
        xori2 = x
        x = self.model2(x)
        B2, C2, H2, W2 = x.shape
        x = x.view(B2, H2, W2, C2)
        x = self.block2(xori2, x, 16, 16)
        x = x.view(B2, C2, H2, W2)

        xori3 = x
        x = self.model3(x)
        B3, C3, H3, W3 = x.shape
        x = x.view(B3, H3, W3, C3)
        x = self.block3(xori3, x, 8, 8)
        xour = x.view(B3, C3, H3, W3)

        # xorimark = self.conv1(xorimark)
        # xorimark = self.layer1(xorimark)
        #
        # xorimark = self.layer2(xorimark)
        # xorimark = self.layer3(xorimark)
        #
        # # x = self.indformer1(x,8,8)
        # if self.levels == 4:
        #     xorimark = self.layer4(xorimark)
        #     # xorimark = self.avgpool4 (xorimark)
        #
        # xour = self.convmid11(xour)
        # # xour = self.convmid21(xour) + self.convmid2(xour)
        # xorimark = self.attentionwave1(xorimark, xour)
        #
        # # xorimark= torch.cat([xour,xorimark],dim=1)
        # # if self.levels == 4:
        # #     xorimark = self.layer4(xorimark)
        # # xorimark = self.conv21(xorimark) + self.conv2(xorimark)
        # # xorimark = self.conv31(xorimark) + self.conv3(xorimark)
        # xorimark = self.adaptivepool(xorimark)
        #
        # xorimark = xorimark.view(x.size(0), -1)
        # # x = self.fc(x)
        # # xorimark = self.fc(torch.relu(self.fc1(xorimark)))
        # xorimark = self.mlp1(xorimark)
        xour = self.score_layer(xour)

        return xour
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # # x = self.fc(x)
        # x = self.fc(torch.relu(self.fc1(x)))
        #
        # return x


class ResNetCIFARNormal(nn.Module):

    def __init__(self, layers, num_classes=1000):
        block = BasicBlock
        self.inplanes = 16
        super(ResNetCIFARNormal, self).__init__()
        raise "It is not possible to use this network"
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Same as Densetnet
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    # model = A2FPN(3).cuda()
    # input = torch.rand(2, 3, 512, 512).cuda()
    # output = model(input)
    # print(output.size())
    from thop import clever_format
    from thop import profile

    # class YourModule(nn.Module):
    #     model = A2FPN(3, class_num=30).cuda()
    # def count_your_model(model, x, y):
    #     # your rule here
    # model = A2FPN(3, class_num=30).cuda()
    model = ResNetCIFAR([2, 2, 2, 2], num_classes=1000, levels=4).cuda()
    input = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)
