import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

from network.model.graph_front.graphFront import _graphFront
from network.model.tgcn.gcn import ConvTemporalGraphical
from network.model.roi_align.modules.roi_align import RoIAlignAvg


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        # kernel_size = _triple(kernel_size)

        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.bn(self.temporal_spatial_conv(x))
        x = self.relu(x)
        return x

class SpatioTemporalResBlocka(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, downsample=False):
        super(SpatioTemporalResBlocka, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        if self.downsample:
            self.conv1a = SpatioTemporalConv(in_channels, out_channels[0], kernel_size=(1,1,1), padding=(0,0,0))
            self.conv1b = SpatioTemporalConv(out_channels[0], out_channels[0], kernel_size=(3,3,3), padding=(1,1,1), stride=(1,2,2))
            self.conv1c = SpatioTemporalConv(out_channels[0], out_channels[1], kernel_size=(1,1,1), padding=(0,0,0))
            self.conv2 = SpatioTemporalConv(in_channels, out_channels[1], kernel_size=(1,1,1), padding=(0,0,0),stride=(1,2,2))
        else:
            self.conv1a = SpatioTemporalConv(in_channels, out_channels[0], kernel_size=(1,1,1), padding=(0,0,0))
            self.conv1b = SpatioTemporalConv(out_channels[0], out_channels[0], kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
            self.conv1c = SpatioTemporalConv(out_channels[0], out_channels[1], kernel_size=(1,1,1), padding=(0,0,0))
            self.conv2 = SpatioTemporalConv(in_channels, out_channels[1], kernel_size=(1,1,1), padding=(0,0,0),stride=(1,1,1))
        # self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        if self.downsample:
            res = self.conv2(x)
            x = self.conv1c(self.conv1b(self.conv1a(x)))
        else:
            res = self.conv2(x)
            x = self.conv1c(self.conv1b(self.conv1a(x)))

        return self.outrelu(x + res)

class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        # padding = kernel_size // 2
        if self.downsample:
            # downsample with stride =2 the input x
            # self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            # self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2 when producing the residual
            self.conv1a = SpatioTemporalConv(in_channels, out_channels[0], kernel_size=(1,1,1), padding=(0,0,0), stride=(1,2,2))
            self.conv1b = SpatioTemporalConv(out_channels[0], out_channels[0], kernel_size=(3,3,3), padding=(1,1,1))
            self.conv1c = SpatioTemporalConv(out_channels[0], out_channels[1], kernel_size=(1,1,1), padding=(0,0,0))
        else:
            self.conv1a = SpatioTemporalConv(out_channels[1], out_channels[0], kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
            self.conv1b = SpatioTemporalConv(out_channels[0], out_channels[0], kernel_size=(3,3,3), padding=(1,1,1))
            self.conv1c = SpatioTemporalConv(out_channels[0], out_channels[1], kernel_size=(1,1,1), padding=(0,0,0))

        self.conv2 = SpatioTemporalConv(in_channels, out_channels[1], kernel_size=(1,1,1), padding=(0,0,0),stride=(1,2,2))
        self.outrelu = nn.ReLU()

    def forward(self, x):
        # print("x is",x.shape)
        if self.downsample:
            res = self.conv2(x)
            x = self.conv1c(self.conv1b(self.conv1a(x)))
        else:
            res = x
            # print("res is",res.shape)
            x = self.conv1c(self.conv1b(self.conv1a(x)))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = SpatioTemporalResBlocka(in_channels, out_channels, downsample=downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels[1], out_channels, downsample=False)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x


class R3DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R3DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [5, 7, 7], stride=[1, 2, 2], padding=[2, 3, 3])
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=[0, 1, 1])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3

        # print('layer_sizes',layer_sizes)
        self.conv2 = SpatioTemporalResLayer(in_channels=64, out_channels=(64, 256), layer_size=layer_sizes[0], block_type=block_type, downsample=False)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=[1, 0, 0])
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(in_channels=256, out_channels=(128, 512), layer_size=layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(in_channels=512, out_channels=(256, 1024), layer_size=layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(in_channels=1024, out_channels=(512, 2048), layer_size=layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(2560, num_classes)

        self.RCNN_roi_align = RoIAlignAvg(7, 7, 1.0/16.0)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.graph = _graphFront()
        self.gcn1 = ConvTemporalGraphical(2048, 512, 1)
        self.gcn2 = ConvTemporalGraphical(512, 512, 1)

    def forward(self, x, bbox):
        print("R3Dx", x.device)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x_avg = self.pool(x).squeeze(2).squeeze(2).squeeze(2)

        print(bbox.shape)

        [N,C,T,H,W] = x.shape
        x = x.permute(0,2,1,3,4).contiguous().view(-1,C,H,W)

        video_pooled_feat = torch.zeros((N*T,20,2048,7,7))
        bbox = bbox[:,:8,:,:]
        bbox = bbox.view(-1,20,5)[:32,:,:]

        # grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
        # grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        # cropped_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
        # global_pooled_feat = F.adaptive_max_pool2d(cropped_feat, (1, 1))
        # global_pooled_feat = global_pooled_feat.squeeze(2).squeeze(2)
        print("bbox.shape", bbox.shape)
        print("R3Dx", x.device)
        for i in range(bbox.shape[0]):
           video_pooled_feat[i] = self.RCNN_roi_align(x[i].view(1,C,H,W), bbox[i]) #(N*T,100,d,7,7)
        video_pooled_feat = video_pooled_feat.cuda()
        print("video_pooled_feat", video_pooled_feat.device)

        #nkctv,kvw->nctw
        #[32, 100, 2048, 7, 7]
        video_pooled_feat = F.adaptive_avg_pool2d(video_pooled_feat.view(-1,2048,7,7), (1, 1)).squeeze(2).squeeze(2).view(T,20,C)  #[1600, 2048, 1, 1]
        print("video_pooled_feat", type(video_pooled_feat))

        adjacent_matrix = torch.randn(N, T*20,T*20)

        adjacent_matrix[0] = self.graph.build_graph(bbox)
        adjacent_matrix = adjacent_matrix.cuda()

        print('adjacent',adjacent_matrix)  #(N,800,800)
        video_pooled_feat = video_pooled_feat.view(N, 2048, -1)
        print("video_pooled_feat",video_pooled_feat)

        node,A = self.gcn1(video_pooled_feat, adjacent_matrix)  #
        node,A = self.gcn2(node, A)

        node = node.squeeze(0).view(512,-1).permute(1,0)

        node = torch.mean(node,0).view(1,-1)

        feat = torch.cat((x_avg,node),dim=1)

        logits = self.linear(feat)

        return logits



class R3DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock, pretrained=False):
        super(R3DClassifier, self).__init__()

        self.res3d = R3DNet(num_classes, layer_sizes, block_type)

        # self.RCNN_roi_align = RoIAlignAvg(7, 7, 1.0/16.0)

        # self.linear = nn.Linear(2048, num_classes)

        # self.pool = nn.AdaptiveAvgPool3d(1)

        # self.graph = _graphFront()

        # self.gcn1 = ConvTemporalGraphical(2048, 512, 1)
        # self.gcn2 = ConvTemporalGraphical(512, 512, 1)
        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x, bbox):
        x = self.res3d(x, bbox)
        print(x.shape)  #torch.Size([1, 2048, 16, 14, 14]) 4, 2048, 8, 14, 14
        # x_avgpool = self.pool(x).squeeze(2).squeeze(2).squeeze(2)
        # print("x_avgpool",x_avgpool.shape) #([4, 2048])

        # [N,C,T,H,W] = x.shape
        # print(N,C,T,H,W)
        # print(x.permute(0,2,1,3,4).shape)
        # x = x.permute(0,2,1,3,4).contiguous().view(-1,C,H,W)
        # #print(x.view(-1,C,H,W))
        # #print('permute',x.permute(0,2,1,3,4).view(-1,C,H,W).shape)  #[16, 2048, 14, 14]
        #
        # print(bbox.view(-1,100,4).shape)  #[16, 100, 4]
        # #video_pooled_feat = self.RCNN_roi_align(x.permute(0,2,1,3,4).view(-1,C,H,W), bbox.view(-1,4)) #(N*T,100,d,7,7)
        #
        # video_pooled_feat = torch.zeros((N*T,100,2048,7,7))
        # print("video",video_pooled_feat.shape)
        # print(bbox.shape)
        # bbox = bbox[:,:8,:,:]
        # bbox = bbox.view(-1,100,4)[:32,:,:]
        #
        # for i in range(bbox.shape[0]):
        #    video_pooled_feat[i] = self.RCNN_roi_align(x[i].view(1,C,H,W), bbox[i]) #(N*T,100,d,7,7)
        #
        # print("video",video_pooled_feat.shape)  #[1600, 2048, 7, 7]
        #
        # #nkctv,kvw->nctw
        # #[32, 100, 2048, 7, 7]
        # video_pooled_feat = F.adaptive_avg_pool2d(video_pooled_feat.view(-1,2048,7,7), (1, 1)).squeeze(2).squeeze(2).view(T,100,C)  #[1600, 2048, 1, 1]
        # print(video_pooled_feat.shape)
        #
        # adjacent_matrix = torch.randn(N, T*100,T*100)
        # print("bbox.shape",bbox.shape)
        # #for i in range(bbox.shape[0]):
        # adjacent_matrix[0] = self.graph.build_graph(bbox)
        #
        # print('adjacent',adjacent_matrix.shape)  #(N,800,800)
        # video_pooled_feat = video_pooled_feat.view(N,2048,1,-1)
        # #video_pooled_feat = video_pooled_feat.view(N,T,C,-1).permute(0,2,1,3)  #1, 1, 16, 2048, 100
        # print("video_pooled_feat",video_pooled_feat.shape)
        #
        # node,A = self.gcn1(video_pooled_feat, adjacent_matrix)  #
        # node,A = self.gcn2(node, A)
        #
        # node = node.squeeze(0).view(512,-1).permute(1,0)
        # print("node",node.shape)  #[1, 512, 16, 100]
        #
        # node = torch.mean(node,0).view(1,-1)
        # print("node",node.shape)
        #
        # feat = torch.cat((x_avgpool,node),dim=1)
        #
        # logits = self.linear(feat)
        # logits = self.linear(x_avgpool)

        return x

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res3d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    import torch
    inputs = torch.rand(1, 3, 32, 224, 224)
    net = R3DClassifier(101, (3, 4, 6, 3), pretrained=True)
    bbox = torch.rand(1, 16, 100, 4)
    outputs = net.forward(inputs, bbox)
    print(outputs.size())
