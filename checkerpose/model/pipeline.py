''' full pose estimation network '''
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_FEATS_DIMS = {
    "resnet34": [64, 128, 256, 512],
    "convnext_tiny": [192, 384, 768],
    "convnext_small": [192, 384, 768],
    "convnext_base": [256, 512, 1024],
    "darknet53": [64, 128, 256, 512, 1024],
    "hrnet_w18": [128, 256, 512, 1024],
    "hrnet_w18_small": [128, 256, 512, 1024],
    "hrnet_w30": [128, 256, 512, 1024],
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

def get_MLP_leakyReLU_layers(dims, doLastAct, negative_slope=0.1):
    layers = []
    for i in range(1, len(dims)):
        layers.append(nn.Linear(dims[i - 1], dims[i]))
        if i == len(dims) - 1 and not doLastAct:
            continue
        layers.append(nn.LeakyReLU(negative_slope=negative_slope))
    layers = nn.Sequential(*layers)
    return layers


def from_code_to_id(code, class_base=2):
    ''' convert (binary) code to index
    input::code: shape (batch, #bits, #keypoints), dtype torch.LongTensor
    output::ids shape (batch, #keypoints)
    '''
    # from code to index
    batch, codes_length, npoint = code.shape
    ids = code[:, 0, :] * (class_base**(codes_length - 1))
    for i in range(1, codes_length):
        ids = ids + code[:, i, :] * (class_base**(codes_length - 1 - i))
    return ids

def from_code_prob_to_id(code_prob, class_base=2):
    ''' convert (binary) code prob (e.g. network predicted codeds) to index
    input::code_prob: shape (batch, #bits, #keypoints)
    output::ids shape (batch, #keypoints)
    '''
    code = torch.sigmoid(code_prob)
    code = torch.where(code > 0.5, 1, 0)
    ids = from_code_to_id(code, class_base=class_base)
    return ids

def from_gt_code_to_id(gt_code, class_base=2):
    ''' convert GT code (Float tensor) to index
    input::gt_code: shape (batch, #bits, #keypoints)
    output::ids shape (batch, #keypoints)
    '''
    code = torch.where(gt_code > 0.5, 1, 0)
    ids = from_code_to_id(code, class_base=class_base)
    return ids

def from_bit_prob_to_id(bit_prob):
    ''' convert (binary) bit prob (e.g. network predicted codeds) to index
    input::bit_prob shape (batch, 1, #keypoints)
    output::ids shape (batch, #keypoints)
    '''
    bit = torch.sigmoid(bit_prob[:, 0, :])  # shape (batch, #keypoints)
    idx = torch.where(bit > 0.5, 1, 0)
    return idx

def from_gt_bit_to_id(gt_bit):
    ''' convert GT bit (Float value) to index
    input::gt_bit shape (batch, 1, #keypoints)
    output::ids shape (batch, #keypoints)
    '''
    idx = torch.where(gt_bit[:, 0, :] > 0.5, 1, 0)
    return idx

def from_mask_prob_to_mask(mask_prob):
    ''' convert the mask prob (e.g. network prediction) to mask with float values 0.0/1.0
    input: mask_prob, can be any shape, e.g. (batch, 1, #keypoints) for roi_mask_bit
    output: mask, w/ same shape of mask_prob
    '''
    mask = torch.sigmoid(mask_prob)
    mask = torch.where(mask > 0.5, 1.0, 0.0)
    return mask


class Index2Feat_module(nn.Module):
    def __init__(self, feat_dim, embed_dim=None, kernel_size=2):
        ''' from pixel x/y index in low resolution feature map to crop feature in high resolution (i.e. x2) map
        compared with from_id_to_local_feat, which only extracts the feature at the sub-pixel, we extract from a KxK region
        the KxK features of 4-subpixels do not overlap. padding zeros for out of RoI positions
        from_id_to_local_feat can be treated as the special case when K=1
        Args:
            feat_dim: feature dimension of the original feature maps
            embed_dim: embedding dim for local feature of each sub-pixel. If None, use feat_dim * K^2
            kernel_size: size of the local features
        '''
        super(Index2Feat_module, self).__init__()
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim if embed_dim is not None else (feat_dim * kernel_size * kernel_size)
        self.patch_generator = nn.Conv2d(in_channels=feat_dim, out_channels=self.embed_dim,
                                         kernel_size=kernel_size, stride=1, padding=kernel_size-1)

    def forward(self, img_feat_highres, batch_indices, pixel_x_id, pixel_y_id):
        ''' Args:
             img_feat_highres: high resolution (i.e. x2) image feature, shape (B, C, H, W)
             batch_indices: aux tensor with shape (B, #keypoint), where batch_indices[i, :] is filled with i
             pixel_x_id: pixel index on x direction, shape: (B, #keypoint)
             pixel_y_id: pixel index on y direction, shape: (B, #keypoint)
        Return local feature with shape (batch, 4CK^2, #keypoint)
        '''
        # first use Conv2D to obtain the local feature patches, similar to PatchEmbedding in ViT model
        patches = self.patch_generator(img_feat_highres)  # shape: (batch, feat_dim * k^2, H, W)
        # we need to get the feature from 4 locations: (2u, 2v), (2u+k, 2v), (2u, 2v+k), and (2u+k, 2v+k)
        sf1 = patches[batch_indices, :, 2 * pixel_y_id, 2 * pixel_x_id]  # shape: (batch, #keypoint, feat_dim * k^2)
        sf2 = patches[batch_indices, :, 2 * pixel_y_id + self.kernel_size, 2 * pixel_x_id]
        sf3 = patches[batch_indices, :, 2 * pixel_y_id, 2 * pixel_x_id + self.kernel_size]
        sf4 = patches[batch_indices, :, 2 * pixel_y_id + self.kernel_size, 2 * pixel_x_id + self.kernel_size]
        local_feat = torch.cat([sf1, sf2, sf3, sf4], dim=2)  # shape: (batch, #keypoint, feat_dim * 4k^2)
        local_feat = local_feat.permute(0, 2, 1)  # shape: (batch, feat_dim*4k^2, #keypoint)
        return local_feat


# note: treat this network as nn.Linear, using shape (B, N, C) for all tensors
class MLP_QueryNet(nn.Module):
    def __init__(self, feat_dims=(256, 256, 64), pt_dim=3, out_dim=4, leaky_slope=0.01):
        super(MLP_QueryNet, self).__init__()
        mlp_dims = feat_dims + (out_dim,)
        self.mlps = get_MLP_leakyReLU_layers(dims=mlp_dims, doLastAct=False, negative_slope=leaky_slope)

    def forward(self, img_feats, pts):
        ''' Args:
        img_feats: image feature corresponding to each 3D point, shape: (B, N, C)
        pts: coordinates of the points, shape: e.g. (B, N, 3)
        '''
        out = self.mlps(img_feats)
        return out


def get_gdrn_upsample_module(is_convtrans=False, in_channels=512, num_filters=256, kernel_size=3, padding=1, output_padding=1):
    layers = []
    if is_convtrans:
        layers.append(
            nn.ConvTranspose2d(
                in_channels,
                num_filters,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(num_features=num_filters))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
    else:
        layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False))

    layers.append(nn.BatchNorm2d(num_features=num_filters))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
    layers.append(nn.BatchNorm2d(num_features=num_filters))
    layers.append(nn.ReLU(inplace=True))
    module = nn.Sequential(*layers)
    return module


class Refine_moduleGNN(nn.Module):
    def __init__(self, npoint, p3d_normed, num_filters=256, max_batch_size=64, query_dims=None,
                 local_k=4, leaky_slope=0.01, num_graph_module=2, graph_k=20, graph_leaky_slope=0.2,
                 query_type="mlp", graph_feat_dim=64):
        ''' Args:
        npoint: number of 3D keypoint
        num_filters: feature dimension when upsampling the image features
        max_batch_size: max value of input batch size (to initialize the batch_indices on GPU)
        local_k: spatial size of the local feature
        leaky_slope: negative slope of the leaky ReLU activation
        graph_feat_dim: dim of graph feature from last step (either refine or init)
        '''
        super(Refine_moduleGNN, self).__init__()
        self.npoint = npoint
        # set the query dims here which will impact the local feature output dim
        if query_type == "mlp":
            self.query_dims = (num_filters, 256, 64) if query_dims is None else query_dims
        else:
            raise ValueError("query type {} not supported in Refine_module".format(query_type))

        # for obtaining local image features
        batch_indices = torch.arange(max_batch_size, dtype=torch.long).view(max_batch_size, 1).repeat(1, npoint)
        if torch.cuda.is_available():
            batch_indices = batch_indices.cuda()
        self.batch_indices = batch_indices  # shape: (max_batch_size, #keypoint)
        self.local_feat_ext_block = Index2Feat_module(feat_dim=num_filters, embed_dim=self.query_dims[0]//4, kernel_size=local_k)

        # concat with prev graph features and process
        self.pre_graph_module = get_MLP_leakyReLU_layers(
            dims=(self.query_dims[0]+graph_feat_dim, self.query_dims[0], self.query_dims[0]),
            doLastAct=True, negative_slope=leaky_slope)

        # graph modules to preprocess the local features before query
        self.pre_query_block = nn.ModuleList()
        knn_idx = knn(p3d_normed, graph_k)
        pre_batch_indices = torch.arange(max_batch_size, dtype=torch.long).view(max_batch_size, 1).repeat(1, npoint * graph_k)
        if torch.cuda.is_available():
            pre_batch_indices = pre_batch_indices.cuda()
        self.pre_batch_indices = pre_batch_indices
        for i in range(num_graph_module):
            graph_module = StaticGraph_module(input_dim=self.query_dims[0], output_dim=self.query_dims[0],
                                              knn_idx=knn_idx, leaky_slope=graph_leaky_slope)
            self.pre_query_block.append(graph_module)

        # query location in high resolution image feature map by predicting binary bit for x/y
        if query_type == "mlp":
            self.query_block = MLP_QueryNet(feat_dims=self.query_dims, pt_dim=3, out_dim=2, leaky_slope=leaky_slope)

    def forward(self, img_feat, graph_feat, p3d_normed, roi_mask_bit, prev_x_id, prev_y_id):
        ''' Args:
        img_feat: shape: (batch, C, 2^res_log2, 2^res_log2)
        graph_feat: shape: (batch, C, #keypoint)
        p3d_normed: shape: (batch, 3, npoint), normalized in range [-1, 1]
        roi_mask_bit: shape: (batch, 1, npoint)
        prev_x_id: shape (batch, npoint), pixel index on x direction obtained in previous stages, dtype torch.LongTensor
        prev_y_id: shape (batch, npoint), pixel index on y direction obtained in previous stages, dtype torch.LongTensor
            note: make sure prev_x_code and prev_y_code are torch.LongTensor
        Return:
            output_bits -- tensor (B, 2, #keypoint) representing new bit of x/y
            output_feat -- tensor (B, C, #keypoint) output graph feature
        '''
        batch_size = img_feat.shape[0]
        # obtain the local image feature based on x/y code
        batch_indices = self.batch_indices[:batch_size]
        local_feat = self.local_feat_ext_block(img_feat, batch_indices, prev_x_id, prev_y_id)  # shape: (batch, C, #keypoint)
        # reset local features of keypoint out of RoI to zeros
        local_feat = local_feat * roi_mask_bit.detach()  # shape: (batch, C, #keypoint)

        # concat with graph feature from last step and process before graph modules
        local_feat = torch.cat([local_feat, graph_feat], dim=1)  # shape: (batch, C, #keypoint)
        local_feat = local_feat.permute(0, 2, 1)  # shape: (batch, #keypoint, C)
        local_feat = self.pre_graph_module(local_feat)  # shape: (batch, #keypoint, C)
        local_feat = local_feat.permute(0, 2, 1)  # shape: (batch, C, #keypoint)

        # use StaticGraph module to process the features before query
        pre_batch_indices = self.pre_batch_indices[:batch_size]
        for i, block in enumerate(self.pre_query_block):
            local_feat = block(local_feat, pre_batch_indices)  # shape: (batch, C, #keypoint)
        local_feat = local_feat.permute(0, 2, 1)  # shape: (batch, #keypoint, C)

        # 4-class classification to decide the refined location
        output_bits = self.query_block(local_feat, p3d_normed.permute(0, 2, 1))  # shape: (batch, #keypoint, 2)
        output_bits = output_bits.permute(0, 2, 1)  # shape: (batch, 2, #keypoint)
        output_feat = local_feat.permute(0, 2, 1)  # shape: (batch, C, #keypoint)
        return output_bits, output_feat


class PoseNet_GNNskip(nn.Module):
    def __init__(self, init_net, npoint, p3d_normed, res_log2=6, num_filters=256, max_batch_size=64, query_dims=None,
                 seg_output_dim=2, local_k=4, leaky_slope=0.01, num_graph_module=2, graph_k=20, graph_leaky_slope=0.2,
                 query_type="mlp"):
        ''' Args:
        init_net: subnet for initial keypoint localization
        npoint: number of 3D keypoint
        res_log2: log2 of the resolution of the 2D projection, default is 6 for 64x64 resolution
        num_filters: feature dimension when upsampling the image features
        max_batch_size: max value of input batch size (to initialize the batch_indices on GPU)
        seg_output_dim: output dimensions for segmentation masks (e.g. 2 for visible+full)
        local_k: spatial size of the local feature
        leaky_slope: negative slope of the leaky ReLU activation
        '''
        super(PoseNet_GNNskip, self).__init__()
        self.npoint = npoint
        self.init_net = init_net

        self.num_refine_steps = res_log2 - 3
        # during refinement, also upsampling backbone feature until desired resolution
        self.up_net = nn.ModuleList()
        for i in range(self.num_refine_steps):
            if i == 0:
                block = get_gdrn_upsample_module(is_convtrans=True,
                                                 in_channels=IMG_FEATS_DIMS[self.init_net.backbone_name][-1],
                                                 num_filters=num_filters)
            else:
                block = get_gdrn_upsample_module(is_convtrans=False,
                                                 in_channels=num_filters+IMG_FEATS_DIMS[self.init_net.backbone_name][-i-1],
                                                 num_filters=num_filters)
            self.up_net.append(block)
        # core refinement component
        self.refine_net = nn.ModuleList()
        for i in range(self.num_refine_steps):
            num_graph_module_i = num_graph_module if isinstance(num_graph_module, int) else num_graph_module[i]
            if i == 0:
                graph_feat_dim_i = 64
            elif query_dims is None:
                graph_feat_dim_i = num_filters
            else:
                graph_feat_dim_i = query_dims[0]
            block = Refine_moduleGNN(npoint=npoint, p3d_normed=p3d_normed, num_filters=num_filters,
                                     max_batch_size=max_batch_size, query_dims=query_dims,
                                     local_k=local_k, leaky_slope=leaky_slope, num_graph_module=num_graph_module_i,
                                     graph_k=graph_k, graph_leaky_slope=graph_leaky_slope, query_type=query_type,
                                     graph_feat_dim=graph_feat_dim_i)
            self.refine_net.append(block)
        # final image segmentation
        self.seg_block = nn.Conv2d(num_filters, seg_output_dim, kernel_size=1, padding=0, bias=True)

    def forward(self, img, p3d_normed, stage=None):
        ''' Args:
        img: shape (batch, 3, H, W)
        p3d_normed: shape: (batch, 3, npoint), normalized in range [-1, 1]
        stage: current stage of progressive training (None means use all refinement blocks)
        gt_x_bits, gt_y_bits: teacher forcing, shape (batch, #bits, #keypoints)
        '''
        num_active_ref = stage if stage is not None else self.num_refine_steps
        # initial localization
        output_bits, img_feats, graph_feat = self.init_net(img, return_img_feats=False, return_graph_feats=True)  # shape: (batch, 7, #keypoints), 1 for in/out RoI, 3 for x/y
        img_feat = img_feats[-1]  # only use the last one
        # split output bits to roi_mask, x, y for easier concat in the refinement
        output_roi_bit = output_bits[:, 0:1, :]
        output_x_bits = output_bits[:, 1:4, :]
        output_y_bits = output_bits[:, 4:, :]
        # refinement step
        roi_mask_bit = from_mask_prob_to_mask(output_roi_bit.detach())
        pred_x_id = from_code_prob_to_id(output_x_bits.detach())
        pred_y_id = from_code_prob_to_id(output_y_bits.detach())
        for i in range(num_active_ref):
            if i > 0:
                img_feat = torch.cat([img_feat, img_feats[-i - 1]], dim=1)
            img_feat = self.up_net[i](img_feat)  # preprocess before refinement: upsampling the feature
            new_bits, graph_feat = self.refine_net[i](img_feat, graph_feat, p3d_normed, roi_mask_bit, pred_x_id, pred_y_id)
            new_x_bit = new_bits[:, 0:1, :]
            new_y_bit = new_bits[:, 1:2, :]
            output_x_bits = torch.cat([output_x_bits, new_x_bit], dim=1)  # shape: (batch, #bits, #keypoints)
            output_y_bits = torch.cat([output_y_bits, new_y_bit], dim=1)
            # update x/y pixel index
            pred_x_id = pred_x_id * 2 + from_bit_prob_to_id(new_x_bit.detach())
            pred_y_id = pred_y_id * 2 + from_bit_prob_to_id(new_y_bit.detach())
        # final image segmentation step
        output_seg = self.seg_block(img_feat)
        return output_roi_bit, output_x_bits, output_y_bits, output_seg, pred_x_id, pred_y_id
