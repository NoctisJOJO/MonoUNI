import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BatchNorm = nn.BatchNorm2d


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return os.path.join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ Bottleneck类是一个典型的残差块（Residual Block），通常用于深度残差网络（ResNet）中。
    这个块的设计重点在于通过瓶颈结构（即使用较小的通道数进行卷积）来减少计算量，同时保留足够的表达能力。
    它的核心思想是将网络中的通道数压缩再恢复，使得网络更加高效。"""
    # expansion是一个常量，通常用于扩展卷积后输出通道的数量，在ResNet等架构中，瓶颈结构通常是将输入的通道数压缩到较小的值，再通过一个卷积恢复回去
    # expansion = 2 意味着， planes （输入层）的值是瓶颈层（即中间层）通道数的两倍。所以，如果 planes = 256，瓶颈层通道数会是 128
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    """ Bottleneck类是一个典型的残差块（Residual Block），通常用于深度残差网络（ResNet）中。
    这个块的设计重点在于通过瓶颈结构（即使用较小的通道数进行卷积）来减少计算量，同时保留足够的表达能力。
    它的核心思想是将网络中的通道数压缩再恢复，使得网络更加高效。
    
    BottleneckX 是一个改进版的瓶颈模块，基于经典的残差块（ResNet Bottleneck），引入了 cardinality（分组卷积）和一些修改的计算方式，
    使得模型更高效并具备更好的表达能力。cardinality 是一个关键参数，表示分组卷积中的组数。
    通过增加组数，模型能够有效地捕捉更多的特征模式，而不会显著增加计算复杂度。"""
    expansion = 2
    cardinality = 32 # 表示组卷积的组数（group convolution）。这里设置为 32，意味着在第二层卷积中将会使用 32 个组来进行卷积操作。

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        # 与传统的瓶颈层不同，BottleneckX 使用了 cardinality 来动态调整瓶颈层通道数。原始的 planes 会根据 cardinality 被缩放，从而形成 bottle_planes。
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        # 使用 groups=cardinality 来实现分组卷积。分组卷积是将输入通道分成多个组，并在每个组上进行独立的卷积操作，极大减少了计算量。
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)  # padding填充是为了保证卷积之后输出的空间尺寸和输入一致（stride步幅=1配合）
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):  # 这里是 *x，即传入指针，表示接受不定数量的输入参数，通常是多个张量
        children = x
        x = self.conv(torch.cat(x, 1))  # 将所有张量沿第一维（通道维度）拼接在一起
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    """ 递归结构：
        Tree 是一个递归模块，可以动态地构建树状神经网络。
        每一层通过两棵子树分支、残差连接以及根节点融合特征。
        
        特征融合：
        通过 Root 模块，在叶节点融合来自不同分支的特征。
        支持残差连接，增强特征的传递效率。
        
        降采样与通道调整：
        提供降采样（MaxPool2d）和通道调整（1x1 Conv），用于处理输入分辨率和通道数不匹配的问题。
        
        灵活性：
        支持通过 block 自定义树的基本构建单元（如残差块）。
        参数 levels 决定树的深度，stride 控制降采样，dilation 调整感受野。
        
        应用场景
        深度特征融合：适用于复杂模型（如 DLA）中的层级聚合。
        特征图降采样：在特征提取阶段逐步降低分辨率，聚合多尺度信息。
        递归网络：用于构建高效且可扩展的递归神经网络模块。
        
        level: 树的层数，每减少1层，子树会进一步展开
        当 levels == 1 时，树的叶节点由两个 block 模块组成，且会引入 Root 模块进行特征融合
        level_root 标志当前层是否是子树的根节点（是否汇集多个路径的特征
        root_dim 根节点特征拼接后的通道数，默认为 2 * out_channels，若 level_root=True，会额外加上 in_channels
        """
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:  # 默认根节点特征维度为 2 * outchannels
            root_dim = 2 * out_channels
        if level_root:      # 如果当前层为根节点，特征维度额外加上输入通道数in_channels
            root_dim += in_channels
        if levels == 1:  # 单层时，直接构建两层 block 模块
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:  # 多层时，递归调用 Tree 构建子树结构
            # 仅第一颗子树使用 stride 处理输入  第二颗子树的 root_dim 包含来自子节点的输出通道
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:  # 同上单层时，引入 Root 模块
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1: # 步幅大于一，降采样，最大池化
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x  # 使用最大池化降采样，再存入 bottom
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)  # 递归调用第一颗子树，将输入x和残差传入
        if self.levels == 1: # 单层时，调用第二棵子树 tree2，使用 Root 模块融合特征，包括：来自 tree1 的输出 x1，来自 tree2 的输出 x2，当前层的子节点特征（*children）
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:  # 多层时，将 tree1 的输出 x1 存入 children，传递给第二棵子树
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    #  inplanes:输入通道数; planes:输出通道数; convs:这一层堆叠的卷积层数量; stride:卷积操作的步幅（仅对第一层有效）; dilation: 膨胀率（扩张卷积的参数）
    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []  # 用于存放该层的卷积模块（卷积层 + 批归一化层 + 激活函数）
        for i in range(convs):  # 循环 convs 次，每次构建一个卷积块，一个卷积块包括卷积层、批归一化层和ReLU激活函数
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        # 多层级特征提取：该方法能够对每个 level 进行处理，提取不同层级的特征，并且可以根据需要返回这些特征。
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x) # x = self.level{0}(x)、1、2、3、4、5 调用每一层的处理函数
            y.append(x)
        # 通过 return_levels 参数控制是否返回所有中间层的输出，适用于需要逐层分析特征的任务（如特征可视化或多尺度分析）
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            # 使用 view 将输出展平，x.size(0) 是批量大小，-1 表示自动推断维度。
            # 展平操作是为了将多维张量转化为一维向量，以便送入全连接层（通常用于分类
            x = x.view(x.size(0), -1)

            return x

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        self.fc = fc


def dla34(pretrained=False, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


def dla46_c(pretrained=False, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla46_c')
    return model


def dla46x_c(pretrained=False, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla46x_c')
    return model


def dla60x_c(pretrained=False, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60x_c', hash='b870c45c')
    return model


def dla60(pretrained=False, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla60')
    return model


def dla60x(pretrained=False, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla60x')
    return model


def dla102(pretrained=False, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla102')
    return model


def dla102x(pretrained=False, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla102x')
    return model


def dla102x2(pretrained=False, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla102x2')
    return model


def dla169(pretrained=False, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_pretrained_model(pretrained, 'dla169')
    return model


if __name__ == '__main__':
    net = dla34(pretrained=False)
    print("net:\n",net)