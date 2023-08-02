''' predict the initial 2D projections of given keypoints
We adopt the binary code representation for 2D pixel location
Input: low resolution image feature map with shape (C0, H0, W0), N 3D keypoint coordinates
Output: for each keypoint, predict
(a) whether in the RoI or not, can be represented as a binary bit
(b) spatial location (H0 x W0 possible results). Can be represented as a log2(H0) + log2(W0) binary code
Convention: we use (batch, feat_dim, *spatial_size) order for all input and output tensors except special notification
    The output (a) (b) will be represent by a binary code w/ 1+log2(H0)+log2(W0) bits
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import get_timm_backbone

CONV1X1_IN_CHANS = {
    "resnet34": 512,
    "convnext_tiny": 768,
    "convnext_small": 768,
    "convnext_base": 1024,
    "darknet53": 1024,
    "hrnet_w18": 1024,
    "hrnet_w18_small": 1024,
    "hrnet_w30": 1024,
}

# x.shape (B, C, N), idx.shape (B, N, K), device is the same as x
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

# x.shape: (B, C, N), knn_idx.shape (B, N, K), output feature shape (B, 2C, N, K)
# batch_indices: aux tensor with shape (B, #keypoint * K), where batch_indices[i, :] is filled with i
def get_graph_feature(x, knn_idx, batch_indices):
    batch_size = x.size(0)
    num_points = x.size(2)
    k = knn_idx.size(2)
    if knn_idx.shape[0] == 1:
        batch_knn_idx = knn_idx.expand(batch_size, -1, -1)
    else:
        batch_knn_idx = knn_idx
    batch_knn_idx = batch_knn_idx.view(batch_size, -1)  # shape: (B, NK)
    knn_feature = x[batch_indices, :, batch_knn_idx]  # shape: (B, NK, C)
    knn_feature = knn_feature.view(batch_size, num_points, k, -1).permute(0, 3, 1, 2)  # shape: (B, C, N, K)
    feature = x.unsqueeze(dim=3).repeat(1, 1, 1, k)  # shape: (B, C, N, K)
    feature = torch.cat([knn_feature - feature, feature], dim=1)  # shape: (B, 2C, N, K)
    return feature

# graph layers with fixed KNN index
# x.shape: (B, C, N), knn_idx.shape (B, N, K), output shape (B, C', N)
# batch_indices: aux tensor with shape (B, #keypoint * K), where batch_indices[i, :] is filled with i
class StaticGraph_module(nn.Module):
    def __init__(self, input_dim, output_dim, knn_idx, leaky_slope=0.2):
        super(StaticGraph_module, self).__init__()
        self.knn_idx = knn_idx
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim * 2, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=leaky_slope)
        )

    def forward(self, x, batch_indices):
        out = get_graph_feature(x, self.knn_idx, batch_indices)  # shape: (B, 2C, N, K)
        out = self.conv(out)  # shape: (B, C', N, K)
        out = out.max(dim=-1, keepdim=False)[0]  # shape: (B, C', N)
        return out


class InitNet_GNN(nn.Module):
    def __init__(self, npoint, p3d_normed, res_log2=3, backbone_name="resnet34", pretrain_backbone=True,
                 num_conv1x1=1, max_batch_size=64, num_graph_module=2, graph_k=20, graph_leaky_slope=0.2):
        ''' Args:
        npoint: number of 3D keypoint
        res_log2: log2 of the resolution of the 2D projection, default is 3 for 8x8 resolution
        '''
        super(InitNet_GNN, self).__init__()
        self.num_out_bits = 1 + 2 * res_log2
        self.npoint = npoint
        self.backbone_name = backbone_name
        self.img_backbone = get_timm_backbone(model_name=backbone_name, concat_decoder=True, pretrained=pretrain_backbone)
        # for initial graph generation
        if num_conv1x1 == 1:
            self.conv1x1 = nn.Conv2d(in_channels=CONV1X1_IN_CHANS[backbone_name], out_channels=npoint,
                                     kernel_size=1, stride=1, padding=0)
        else:
            conv1x1 = []
            conv1x1.append(nn.Conv2d(in_channels=CONV1X1_IN_CHANS[backbone_name], out_channels=npoint,
                                     kernel_size=1, stride=1, padding=0))
            for i in range(num_conv1x1 - 1):
                conv1x1.append(nn.LeakyReLU(negative_slope=0.01))
                conv1x1.append(nn.Conv2d(in_channels=npoint, out_channels=npoint, kernel_size=1, stride=1, padding=0))
            self.conv1x1 = nn.Sequential(*conv1x1)

        # graph modules to preprocess the local features before query
        self.pre_query_block = nn.ModuleList()
        knn_idx = knn(p3d_normed, graph_k)
        pre_batch_indices = torch.arange(max_batch_size, dtype=torch.long).view(max_batch_size, 1).repeat(1, npoint * graph_k)
        if torch.cuda.is_available():
            pre_batch_indices = pre_batch_indices.cuda()
        self.pre_batch_indices = pre_batch_indices
        for i in range(num_graph_module):
            graph_module = StaticGraph_module(input_dim=64, output_dim=64, knn_idx=knn_idx, leaky_slope=graph_leaky_slope)
            self.pre_query_block.append(graph_module)

        self.mlp = nn.Linear(in_features=64, out_features=self.num_out_bits)

    def forward(self, img, return_img_feats=False, return_graph_feats=False):
        # construct the initial graph
        img_feats = self.img_backbone(img)  # final one is shape: (batch, C, 8, 8)
        out = self.conv1x1(img_feats[-1])  # shape: (batch, #keypoint, 8, 8)
        batch_size = out.shape[0]
        graph_feats = out.view(-1, self.npoint, 64).permute(0, 2, 1)  # shape: (batch, 64, #keypoints)
        # use StaticGraph module to process the features before query
        pre_batch_indices = self.pre_batch_indices[:batch_size]
        for i, block in enumerate(self.pre_query_block):
            graph_feats = block(graph_feats, pre_batch_indices)  # shape: (batch, 64, #keypoints)
        # final query
        out = graph_feats.permute(0, 2, 1)  # shape: (batch, #keypoints, 64)
        out = self.mlp(out)  # shape: (batch, #keypoint, #bits)
        out = out.permute(0, 2, 1)  # shape: (batch, #bits, #keypoints)
        if return_img_feats:
            return out, img_feats
        elif return_graph_feats:
            return out, img_feats, graph_feats
        else:
            return out
