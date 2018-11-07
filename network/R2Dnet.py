import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
from torch.autograd import Variable

# from network.model.graph_front.graphFront import _graphFront
# from network.model.tgcn.gcn import ConvTemporalGraphical
# from network.model.roi_align.modules.roi_align import RoIAlignAvg
from mypath import Path

class SpatioConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(1,1), stride=(1,1), \
    padding=(0,0), bias=False):
        super(SpatioConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.relu(x)
        return x

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.blocks = nn.ModuleList([])

        self.blocks += [SpatioConv(3, 96, (3,3), (2,2))]
        self.blocks += [nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))]
        self.blocks += [SpatioConv(96, 256, (3,3), (2,2))]
        self.blocks += [nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))]
        self.blocks += [SpatioConv(256, 384, (3,3), (1,1))]
        self.blocks += [SpatioConv(384, 384, (3,3), (1,1))]
        self.blocks += [SpatioConv(384, 256, (3,3), (1,1))]
        self.blocks += [nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))]
        # self.linear1 = nn.Linear(4096,4096)
        # self.linear2 = nn.Linear(4096,4096)
        self.linear1 = nn.Linear(768, 256)
        # self.linear2 = nn.Linear(256,256)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        # x = self.linear2(x)
        return x

class VGG(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, \
        'M', 512, 512, 512, 512, 'M']
        self.features = self.make_layers()
        self.fc6 = nn.Linear(12288, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        # self.classifier = nn.Linear(2048, num_classes)
        # self._initialize_weights()

    def forward(self, x):
        # print("input", x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print("VGG", np.where(x>100))
        # x = self.fc6(x)
        # x = self.fc7(x)
        # x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()

    def make_layers(self, batch_norm=False):
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

class R2DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, group_num_classes, hidden_dim, embedding_dim):
        super(R2DNet, self).__init__()
        # self.base_model = BaseModel()
        self.base_model = VGG(num_classes=group_num_classes)
        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1da = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.conv1db = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.convLinear = nn.Conv1d(in_channels=512, out_channels=group_num_classes, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)

        # self.hidden = (torch.autograd.Variable(torch.zeros(1,21,self.hidden_dim)).cuda(), \
        # torch.autograd.Variable(torch.zeros(1,21,self.hidden_dim)).cuda())
        self.linear = nn.Linear(256, group_num_classes)

    def forward(self, x, dist, dist_num):
        [N, T, M, C, H, W] = x.shape
        # base_out = self.base_model(x.view(-1, C, H, W)).view(N*T, M, -1)

        # print(np.where(base_out>100))
        # base_out_c.detach()
        # base_out_c.no_grad()
        # print("base_out", base_out[0][0])
        # print("base_out", base_out[0][1])
        # x = x.view(-1, C, H, W)

        # dist = torch.zeros(N*T, M, M)
        # for i in np.arange(N*T):
        #     for j in np.arange(M):
        #         for k in np.arange(j+1,M):
        #             # print("base_out", base_out[i][j])
        #             # print("base_out", base_out[i][k])
        #             dist[i,j,k] = F.pairwise_distance(base_out[i][j][:].unsqueeze(0), \
        #             base_out[i][k][:].unsqueeze(0))
        #             dist[i,k,j] = dist[i,j,k]
        #
        #     dist_des, dist_index = dist[i].sort(1,descending=True)
        #     for l in np.arange(M):
        #         dist[i][l][dist_index[:, 3:][l]] = 0
        #         dist[i][l][dist_index[:, :3][l]] = 0.25
        #
        #     # print("dist",dist[i])
        #     dist[i] += torch.eye(M)*0.25
        #     dist[i] = self.normalize_digraph(dist[i].unsqueeze(0).cuda())
        # # print("dist",dist[0])

        # with torch.no_grad():
        #     base_out = Variable(base_out)

        # print("base_out", base_out.shape)
        gcn_out = torch.zeros(N*T, 8).cuda()
        dist_num = dist_num.view(-1)
        # print("dist_num", dist_num)
        dist = dist.view(-1, 12, 12)
        x = x.view(-1, M, C, H, W)
        for i in range(N*T):
            # print("dist_num", dist_num.view(-1)[i])
            base_out = self.base_model(x[i, :dist_num[i]])
            with torch.no_grad():
                base_out = Variable(base_out)
            # print(base_out[i][:dist_num.view(-1)[i]].unsqueeze(0).shape)
            node1 = self.conv1da(base_out[:dist_num.view(-1)[i]].unsqueeze(0).permute(0,2,1).contiguous())
            # print("before", node1)
            # print(dist[i,:dist_num[i],:dist_num[i]].unsqueeze(0).shape)
            # print(node1.permute(0,2,1).contiguous().shape)
            node1 = torch.bmm(dist[i, :dist_num[i], :dist_num[i]].unsqueeze(0).float(), \
            node1.permute(0,2,1).contiguous())
            # print("bmm", node1)
            node1 = F.relu(node1)
            nodeLinear = self.convLinear(node1.permute(0,2,1).contiguous())
            # print("nodeLinear", nodeLinear)
            pooled_feat = self.pool(nodeLinear).squeeze(2)
            # print("pooled_feat", pooled_feat.shape)
            gcn_out[i] = pooled_feat

        group_out = self.avg_pool(gcn_out.view(N, -1, T))

        #normalize
        # node1 = self.conv1da(base_out.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        # node1 = torch.bmm(dist.squeeze(0)[::2,:,:].float(), node1).permute(0,2,1).contiguous()
        # node1 = F.relu(node1)
        # node2 = self.conv1db(node1).permute(0,2,1).contiguous()
        # node2 = torch.bmm(dist.squeeze(0)[::2,:,:].float(), node2).permute(0,2,1).contiguous()
        # node2 = F.relu(node2)
        #
        # nodeLinear = self.convLinear(node2)
        #
        # pooled_feat = self.pool(nodeLinear).squeeze(2).view(N, T, -1)
        # group_out = self.avg_pool(pooled_feat.view(N,-1,T))

        # print("gcn", gcn_out.shape)
        # group_out, _ = self.lstm(gcn_out.view(N, T, -1))
        # # print("group_out", group_out.squeeze(2).squeeze(0))
        # # group_cls_out = self.linear(group_out[:,-1,:])
        # # group_cls_out = self.linear(group_out.squeeze(2))
        # print(group_out.shape)
        # return group_out[:,-1,:]
        return group_out.squeeze(2)

    def normalize_digraph(self, A):
        Dl = torch.sum(A, 2)
        num_node = A.shape[2]
        Dn = torch.zeros(A.shape[0], num_node, num_node).cuda()
        for i in range(A.shape[0]):
            for j in range(num_node):
                if Dl[i][j] > 0:
                     Dn[i][j][j] = Dl[i][j]**(-1)
        AD = torch.bmm(Dn,A)

        return AD

class R2DClassifier(nn.Module):

    def __init__(self, group_num_classes,  hidden_dim=8, embedding_dim=8, pretrained=False):
        super(R2DClassifier, self).__init__()

        self.res2d = R2DNet(group_num_classes, hidden_dim, embedding_dim)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x, dist, dist_num):
        x = self.res2d(x, dist, dist_num)
        return x

    def __load_pretrained_weights(self):

        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "module.features.0.weight": "res2d.base_model.features.0.weight",
                        "module.features.0.bias": "res2d.base_model.features.0.bias",
                        # Conv2
                        "module.features.2.weight": "res2d.base_model.features.2.weight",
                        "module.features.2.bias": "res2d.base_model.features.2.bias",
                        # Conv3a
                        "module.features.5.weight": "res2d.base_model.features.5.weight",
                        "module.features.5.bias": "res2d.base_model.features.5.bias",
                        # Conv3b
                        "module.features.7.weight": "res2d.base_model.features.7.weight",
                        "module.features.7.bias": "res2d.base_model.features.7.bias",
                        # Conv4a
                        "module.features.10.weight": "res2d.base_model.features.10.weight",
                        "module.features.10.bias": "res2d.base_model.features.10.bias",
                        # Conv4b
                        "module.features.12.weight": "res2d.base_model.features.12.weight",
                        "module.features.12.bias": "res2d.base_model.features.12.bias",
                        # Conv5a
                        "module.features.14.weight": "res2d.base_model.features.14.weight",
                        "module.features.14.bias": "res2d.base_model.features.14.bias",
                         # Conv5b
                        "module.features.16.weight": "res2d.base_model.features.16.weight",
                        "module.features.16.bias": "res2d.base_model.features.16.bias",

                        "module.features.19.weight": "res2d.base_model.features.19.weight",
                        "module.features.19.bias": "res2d.base_model.features.19.bias",

                        "module.features.21.weight": "res2d.base_model.features.21.weight",
                        "module.features.21.bias": "res2d.base_model.features.21.bias",

                        "module.features.23.weight": "res2d.base_model.features.23.weight",
                        "module.features.23.bias": "res2d.base_model.features.23.bias",

                        "module.features.25.weight": "res2d.base_model.features.25.weight",
                        "module.features.25.bias": "res2d.base_model.features.25.bias",

                        "module.features.28.weight": "res2d.base_model.features.28.weight",
                        "module.features.28.bias": "res2d.base_model.features.28.bias",

                        "module.features.30.weight": "res2d.base_model.features.30.weight",
                        "module.features.30.bias": "res2d.base_model.features.30.bias",

                        "module.features.32.weight": "res2d.base_model.features.32.weight",
                        "module.features.32.bias": "res2d.base_model.features.32.bias",

                        "module.features.34.weight": "res2d.base_model.features.34.weight",
                        "module.features.34.bias": "res2d.base_model.features.34.bias",
                        # fc6
                        "module.fc6.weight": "res2d.base_model.fc6.weight",
                        "module.fc6.bias": "res2d.base_model.fc6.bias",
                        # fc7
                        "module.fc7.weight": "res2d.base_model.fc7.weight",
                        "module.fc7.bias": "res2d.base_model.fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        print("p_dict", [item for item in p_dict["state_dict"]])
        # print("p_dict", p_dict["state_dict"])
        s_dict = self.state_dict()
        # for item in s_dict:
        #     print("sdict", item)
        for name in p_dict['state_dict']:
            if name not in corresp_name:
                print("not", name)
                continue
            s_dict[corresp_name[name]] = p_dict["state_dict"][name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        # print("self.modules", self.modules)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # print("m",m.weight)
                # nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
                # nn.init.kaiming_normal_(m.weight)
                # print("m",m.weight)
            elif isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
                # nn.init.kaiming_normal_(m.weight)
                # print("m",m.weight)
                # print("m",m)
            # elif isinstance(m, nn.BatchNorm3d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight)
        #         # print("m",m.weight)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
            # print("m",m.weight)

if __name__ == "__main__":
    import torch
    inputs = torch.rand(1, 32, 12, 3, 224, 224)
    net = R2DNet(8, 512, 512)
    # bbox = torch.rand(1, 16, 100, 4)
    outputs = net.forward(inputs)
    print(outputs.size())

    #
    # def _prepare_base_model(self, base_model):
    #
    #     if 'resnet' in base_model or 'vgg' in base_model:
    #         self.base_model = getattr(torchvision.models, base_model)(True)
    #         self.base_model.last_layer_name = 'fc'
    #         self.input_size = 224
    #         self.input_mean = [0.485, 0.456, 0.406]
    #         self.input_std = [0.229, 0.224, 0.225]
    #     else:
    #         raise ValueError('Unknown base model: {}'.format(base_model))
