''' For LM dataset only '''
import datetime
import os
import sys
import time
import mmcv
sys.path.insert(0, os.getcwd())
from config_parser import parse_cfg
import argparse

from tools_for_BOP import bop_io
from tools_for_BOP.common_dataset_info import get_obj_info
from lm_dataset_pytorch import load_lm_obj_diameters, load_lm_obj_sym_info, lm_dataset_single_obj_pytorch_code2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import glob
import cv2
from tqdm import tqdm
from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout, pose_error
from model.init_lm import InitNet_GNN
from model.pipeline_lm import PoseNet_GNNskip, PoseNet_GNNskip_ABwoProg
from torch.utils.tensorboard import SummaryWriter
from checkerpose.aux_utils.pointnet2_utils import pc_normalize
from common_ops import from_dim_str_to_tuple
from test_network_with_test_data import compute_mask_pixelwise_error, compute_mask_iou, from_id_to_pose

def get_closest_rot(rot_est, rot_gt, sym_info):
    """get the closest rot_gt given rot_est and sym_info.
    rot_est: ndarray
    rot_gt: ndarray
    sym_info: None or Kx3x3 ndarray, m2m
    """
    if sym_info is None:
        return rot_gt
    if isinstance(sym_info, torch.Tensor):
        sym_info = sym_info.cpu().numpy()
    if len(sym_info.shape) == 2:
        sym_info = sym_info.reshape((1, 3, 3))
    # find the closest rot_gt with smallest re
    r_err = pose_error.re(rot_est, rot_gt)
    closest_rot_gt = rot_gt
    for i in range(sym_info.shape[0]):
        # R_gt_m2c x R_sym_m2m ==> R_gt_sym_m2c
        rot_gt_sym = rot_gt.dot(sym_info[i])
        cur_re = pose_error.re(rot_est, rot_gt_sym)
        if cur_re < r_err:
            r_err = cur_re
            closest_rot_gt = rot_gt_sym
    return closest_rot_gt

def main(configs):
    datasets_root = configs['datasets_root']
    dataset_name = 'lm'
    #### training dataset
    training_data_folder = configs['training_data_folder']
    training_data_folder_2 = configs['training_data_folder_2']
    test_folder = configs['test_folder']  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']  # the percentage of second dataset in the batch
    num_workers = configs['num_workers']  # for data loader
    train_obj_visible_theshold = configs['train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values
    #### network settings
    init_network_type = configs.get("init_network_type", "naive")
    init_network_backbone_name = configs.get("init_network_backbone_name", "resnet34")
    init_pretrained_root = configs.get("init_pretrained_root", None)  # load pretrained network
    init_network_num_conv1x1 = configs.get("init_network_num_conv1x1", 1)
    init_network_num_graph_module = configs.get("init_network_num_graph_module", 2)  # for graph modules before query
    init_network_graph_k = configs.get("init_network_graph_k", 20)
    init_network_graph_leaky_slope = configs.get("init_network_graph_leaky_slope", 0.2)
    network_type = configs.get("network_type", "vanilla")
    network_res_log2 = configs.get("network_res_log2", 4)
    network_query_dims_str = configs.get("network_query_dims_str", None)
    network_query_type = configs.get("network_query_type", "mlp")
    network_num_filters = configs.get("network_num_filters", 256)
    network_local_k = configs.get("network_local_k", 4)  # spatial size for local feature extraction
    network_leaky_slope = configs.get("network_leaky_slope", 0.1)  # leakyReLU negative slope
    network_num_graph_module = configs.get("network_num_graph_module", 2)  # for graph modules before query
    network_graph_k = configs.get("network_graph_k", 20)
    network_graph_leaky_slope = configs.get("network_graph_leaky_slope", 0.2)
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = int(2 ** network_res_log2)  # network output size
    #### check points
    ckpt_file = configs['ckpt_file']
    #### optimizer
    batch_size = configs['batch_size']  # 32 is the best so far, set to 16 for debug in local machine
    #### augmentations
    Detection_reaults = configs['Detection_reaults']  # for the test, the detected bounding box provided by GDR Net
    padding_ratio = configs['padding_ratio']  # pad the bounding box for training and test
    resize_method = configs['resize_method']  # how to resize the roi images to 256*256
    use_peper_salt = configs['use_peper_salt']  # if add additional peper_salt in the augmentation
    use_motion_blur = configs['use_motion_blur']  # if add additional motion_blur in the augmentation
    #### 3D keypoints
    num_p3d = int(2 ** configs['num_p3d_log2'])
    fps_version = configs.get("fps_version", "fps_202212")
    #### pose estimation
    use_progressivex = configs['use_progressivex']
    prog_max_iters = configs['prog_max_iters']
    nbr_ball_radius = configs['nbr_ball_radius']
    spatial_coherence_weight = configs['spatial_coherence_weight']
    adx_type = configs['adx_type']

    # model info
    symmetry_ids = [10, 11, 7, 3]
    lm13_obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]  # no bowl, cup
    # get object diameter (for compute ADD metric)
    model_info_path = os.path.join(datasets_root, "BOP_DATASETS/lm/models/models_info.json")
    obj_diameter_dict = load_lm_obj_diameters(model_info_path)
    for key, val in obj_diameter_dict.items():
        print("object {:06d} diameter {}".format(key, val), flush=True)
    # get object vertices (for compute ADD metric)
    vertices_dict = {}
    for idx in range(15):
        mesh_path = os.path.join(datasets_root, "BOP_DATASETS/lm/models/obj_{:06d}.ply".format(idx + 1))
        vertices = inout.load_ply(mesh_path)["pts"]
        vertices_dict[idx + 1] = vertices
    # get sym info for rotation error computation
    obj_sym_info_dict = load_lm_obj_sym_info(model_info_path)

    # load 3D points and obtain the normalized 3D keypoints in range [-1, 1]
    lm_p3d_xyz = np.zeros((15, num_p3d, 3))
    lm_p3d_normed = np.zeros((15, num_p3d, 3))
    for idx in range(15):
        fps_path = "datasets/BOP_DATASETS/lm/{}/obj_{:06d}.pkl".format(fps_version, idx + 1)
        print("load FPS points from {}".format(fps_path))
        fps_data = mmcv.load(fps_path)
        p3d_xyz = fps_data['xyz'][:num_p3d, :]
        lm_p3d_xyz[idx] = p3d_xyz
        p3d_normed, p3d_centroid, p3d_range = pc_normalize(p3d_xyz.copy(), return_stat=True)
        print("FPS points, [before normalization] min {} max {} [after normalization] min {} max {}".format(
            p3d_xyz.min(), p3d_xyz.max(), p3d_normed.min(), p3d_normed.max()
        ))
        lm_p3d_normed[idx] = p3d_normed
    lm_p3d_normed = torch.as_tensor(lm_p3d_normed, dtype=torch.float32).transpose(2, 1)  # shape: (15, 3, n)
    if torch.cuda.is_available():
        lm_p3d_normed = lm_p3d_normed.cuda()

    # define test data loader
    test_dataset = lm_dataset_single_obj_pytorch_code2d(
        datasets_root, test_folder, False,
        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, lm_p3d_xyz,
        padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox_file=Detection_reaults,
        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
    )
    print("number of test images: ", len(test_dataset), flush=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    #############build the network
    # first create the initial network (no need to load weights here)
    if init_network_type == "GNN":
        init_net = InitNet_GNN(npoint=num_p3d, p3d_normed=lm_p3d_normed, res_log2=3,
                               backbone_name=init_network_backbone_name, num_conv1x1=init_network_num_conv1x1,
                               max_batch_size=batch_size, num_graph_module=init_network_num_graph_module,
                               graph_k=init_network_graph_k, graph_leaky_slope=init_network_graph_leaky_slope,
                               pretrain_backbone=False)
    else:
        raise ValueError("init network type {} not supported in test_lm".format(init_network_type))

    # create the full pipeline
    network_query_dims = from_dim_str_to_tuple(network_query_dims_str)
    if isinstance(network_num_graph_module, str):
        network_num_graph_module = from_dim_str_to_tuple(network_num_graph_module)
    if network_type == "vanilla_GNNskip":
        net = PoseNet_GNNskip(init_net=init_net, npoint=num_p3d, p3d_normed=lm_p3d_normed, res_log2=network_res_log2,
                              num_filters=network_num_filters, max_batch_size=batch_size, query_dims=network_query_dims,
                              local_k=network_local_k, leaky_slope=network_leaky_slope,
                              num_graph_module=network_num_graph_module, graph_k=network_graph_k,
                              graph_leaky_slope=network_graph_leaky_slope, query_type=network_query_type)
    elif network_type == "vanilla_GNNskip_ABwoProg":
        net = PoseNet_GNNskip_ABwoProg(init_net=init_net, npoint=num_p3d, p3d_normed=lm_p3d_normed, res_log2=network_res_log2,
                                       num_filters=network_num_filters, max_batch_size=batch_size, query_dims=network_query_dims,
                                       local_k=network_local_k, leaky_slope=network_leaky_slope,
                                       num_graph_module=network_num_graph_module, graph_k=network_graph_k,
                                       graph_leaky_slope=network_graph_leaky_slope, query_type=network_query_type)
    else:
        raise ValueError("network type {} not supported in test_lm".format(network_type))
    if torch.cuda.is_available():
        net = net.cuda()
    print("PoseNet: ", net)

    net_ckpt = torch.load(ckpt_file)
    net.load_state_dict(net_ckpt['model_state_dict'])
    net.eval()

    # test the network
    activation_function = nn.Sigmoid()  # for inference: convert prediction to probability
    seg_size = (2 ** network_res_log2, 2 ** network_res_log2)  # size of segmentation masks
    adx2_passed_dict = {idx: [] for idx in lm13_obj_ids}
    adx5_passed_dict = {idx: [] for idx in lm13_obj_ids}
    adx10_passed_dict = {idx: [] for idx in lm13_obj_ids}
    rete2_passed_dict = {idx: [] for idx in lm13_obj_ids}
    rete5_passed_dict = {idx: [] for idx in lm13_obj_ids}
    re2_passed_dict = {idx: [] for idx in lm13_obj_ids}
    re5_passed_dict = {idx: [] for idx in lm13_obj_ids}
    te2_passed_dict = {idx: [] for idx in lm13_obj_ids}
    te5_passed_dict = {idx: [] for idx in lm13_obj_ids}
    # some aux eval metrics not for pose estimation, for simplicity just average over all images
    roi_bit_acc_arr = np.zeros(len(test_loader.dataset))
    reproj_x_acc_arr = np.zeros(len(test_loader.dataset))
    reproj_y_acc_arr = np.zeros(len(test_loader.dataset))
    bit_err_arr = np.zeros((len(test_loader.dataset), 2 * network_res_log2 + 1))
    visib_pixel_acc_arr = np.zeros(len(test_loader.dataset))
    visib_iou_arr = np.zeros(len(test_loader.dataset))
    full_pixel_acc_arr = np.zeros(len(test_loader.dataset))
    full_iou_arr = np.zeros(len(test_loader.dataset))

    print("test dataset", flush=True)
    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, cam_Ks, obj_ids,
                    roi_mask_bits, pixel_x_codes, pixel_y_codes, roi_xy_oris) in enumerate(tqdm(test_loader)):
        # do the prediction and get the predicted binary code
        if torch.cuda.is_available():
            data = data.cuda()
            roi_mask_bits = roi_mask_bits.cuda()
            pixel_x_codes = pixel_x_codes.cuda()
            pixel_y_codes = pixel_y_codes.cuda()

        cur_batch_size = data.shape[0]
        batch_p3d_normed = lm_p3d_normed[obj_ids-1]  # shape: (batch, 3, #keypoint)
        pred_roi_bit, pred_x_bits, pred_y_bits, pred_seg, pred_x_id, pred_y_id = net(data, batch_p3d_normed, obj_ids)

        # split to roi bit, x_bit and y_bit
        pred_roi_bit = activation_function(pred_roi_bit)
        pred_roi_bit = torch.where(pred_roi_bit > 0.5, 1.0, 0.0)
        pred_x_bits = activation_function(pred_x_bits)
        pred_x_bits = torch.where(pred_x_bits > 0.5, 1.0, 0.0)
        pred_y_bits = activation_function(pred_y_bits)
        pred_y_bits = torch.where(pred_y_bits > 0.5, 1.0, 0.0)
        gt_x_bits = pixel_x_codes[:, :network_res_log2]
        gt_y_bits = pixel_y_codes[:, :network_res_log2]
        # convert to numpy array, shape: (batch, #keypoint, #bits)
        pred_roi_bit = pred_roi_bit.detach().cpu().numpy().transpose(0, 2, 1)
        pred_x_bits = pred_x_bits.detach().cpu().numpy().transpose(0, 2, 1)
        pred_y_bits = pred_y_bits.detach().cpu().numpy().transpose(0, 2, 1)
        gt_roi_bit = roi_mask_bits.detach().cpu().numpy().transpose(0, 2, 1)
        gt_x_bits = gt_x_bits.detach().cpu().numpy().transpose(0, 2, 1)
        gt_y_bits = gt_y_bits.detach().cpu().numpy().transpose(0, 2, 1)
        # convert to numpy array, shape: (batch, #keypoint)
        pred_x_id = pred_x_id.detach().cpu().numpy()
        pred_y_id = pred_y_id.detach().cpu().numpy()
        # split predicted segmentation masks to visible and full
        pred_seg = activation_function(pred_seg)
        pred_seg = torch.where(pred_seg > 0.5, 1.0, 0.0)
        pred_seg_visib = pred_seg[:, 0, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        pred_seg_full = pred_seg[:, 1, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        gt_seg_visib = F.interpolate(masks[:, None], size=seg_size, mode="nearest")
        gt_seg_visib = gt_seg_visib.squeeze(dim=1).detach().cpu().numpy()  # shape: (batch, h, w)
        gt_seg_full = F.interpolate(entire_masks[:, None], size=seg_size, mode="nearest")
        gt_seg_full = gt_seg_full.squeeze(dim=1).detach().cpu().numpy()  # shape: (batch, h, w)

        # for pose estimation
        roi_xy_oris = roi_xy_oris.detach().cpu().numpy().transpose(0, 2, 3, 1)  # shape: (B, H, W, 2)
        cam_Ks = cam_Ks.detach().cpu().numpy()
        Rs = Rs.detach().cpu().numpy()
        ts = ts.detach().cpu().numpy()
        obj_ids = obj_ids.detach().cpu().numpy()

        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            # compute pose
            roi_xy_ori = roi_xy_oris[counter]
            obj_id = obj_ids[counter]
            p3d_xyz = lm_p3d_xyz[obj_id-1]
            # pose using all 3D-2D correspondences that are in the RoI
            if adx_type == "default":
                pred_rot, pred_trans = from_id_to_pose(p3d_xyz=p3d_xyz, roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                       roi_mask_bit=pred_roi_bit[counter],
                                                       pixel_x_id=pred_x_id[counter],
                                                       pixel_y_id=pred_y_id[counter], check_seg=False,
                                                       use_progressivex=use_progressivex,
                                                       neighborhood_ball_radius=nbr_ball_radius,
                                                       spatial_coherence_weight=spatial_coherence_weight,
                                                       prog_max_iters=prog_max_iters)
            elif adx_type == "full":
            # discard correspondences that are out of predicted full segmentation masks
                pred_rot, pred_trans = from_id_to_pose(p3d_xyz=p3d_xyz, roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                       roi_mask_bit=pred_roi_bit[counter],
                                                       pixel_x_id=pred_x_id[counter],
                                                       pixel_y_id=pred_y_id[counter], check_seg=True,
                                                       seg_mask=pred_seg_full[counter],
                                                       use_progressivex=use_progressivex,
                                                       neighborhood_ball_radius=nbr_ball_radius,
                                                       spatial_coherence_weight=spatial_coherence_weight,
                                                       prog_max_iters=prog_max_iters)
            elif adx_type == "visib":
                # discard correspondences that are out of predicted visible segmentation masks
                pred_rot, pred_trans = from_id_to_pose(p3d_xyz=p3d_xyz, roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                       roi_mask_bit=pred_roi_bit[counter],
                                                       pixel_x_id=pred_x_id[counter],
                                                       pixel_y_id=pred_y_id[counter], check_seg=True,
                                                       seg_mask=pred_seg_visib[counter],
                                                       use_progressivex=use_progressivex,
                                                       neighborhood_ball_radius=nbr_ball_radius,
                                                       spatial_coherence_weight=spatial_coherence_weight,
                                                       prog_max_iters=prog_max_iters)
            # compute pose error
            if obj_id in symmetry_ids:
                Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
            else:
                Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
            obj_diameter = obj_diameter_dict[obj_id]
            vertices = vertices_dict[obj_id]
            adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, pred_rot, pred_trans, vertices)
            if np.isnan(adx_error):
                adx_error = 10000
            if obj_id in symmetry_ids:
                r_GT_sym = get_closest_rot(pred_rot, r_GT, obj_sym_info_dict[obj_id])
                err_rot = pose_error.re(pred_rot, r_GT_sym)
            else:
                err_rot = pose_error.re(pred_rot, r_GT)
            if np.isnan(err_rot):
                err_rot = 10000
            err_trans = pose_error.te(t_GT, pred_trans)
            if np.isnan(err_trans):
                err_trans = 10000

            adx2_passed_dict[obj_id].append(float(adx_error < obj_diameter * 0.02))
            adx5_passed_dict[obj_id].append(float(adx_error < obj_diameter * 0.05))
            adx10_passed_dict[obj_id].append(float(adx_error < obj_diameter * 0.1))
            rete2_passed_dict[obj_id].append(float(err_rot < 2 and err_trans < 20))
            rete5_passed_dict[obj_id].append(float(err_rot < 5 and err_trans < 50))
            re2_passed_dict[obj_id].append(float(err_rot < 2))
            re5_passed_dict[obj_id].append(float(err_rot < 5))
            te2_passed_dict[obj_id].append(float(err_trans < 20))
            te5_passed_dict[obj_id].append(float(err_trans < 50))

            # compute the reprojection accuracy
            npoint_in_roi = np.clip(gt_roi_bit[counter].sum(), a_min=1.0, a_max=None)
            err_roi_bit = np.mean(np.abs(gt_roi_bit[counter] - pred_roi_bit[counter]))
            roi_bit_acc_arr[batch_idx] = 1.0 - err_roi_bit
            diff_x_bits = (gt_x_bits[counter] - pred_x_bits[counter]) * gt_roi_bit[counter]
            diff_y_bits = (gt_y_bits[counter] - pred_y_bits[counter]) * gt_roi_bit[counter]
            reproj_err_x, reproj_err_y = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(network_res_log2):
                reproj_err_x += diff_x_bits[:, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
                reproj_err_y += diff_y_bits[:, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
            reproj_err_x = np.sum(np.abs(reproj_err_x)) / npoint_in_roi
            reproj_err_y = np.sum(np.abs(reproj_err_y)) / npoint_in_roi
            reproj_x_acc_arr[batch_idx] = 1.0 - reproj_err_x / (2 ** network_res_log2)
            reproj_y_acc_arr[batch_idx] = 1.0 - reproj_err_y / (2 ** network_res_log2)
            # compute the bit-wise error
            err_x_bits = np.sum(np.abs(diff_x_bits), axis=0) / npoint_in_roi
            err_y_bits = np.sum(np.abs(diff_y_bits), axis=0) / npoint_in_roi
            bit_err_arr[batch_idx, 0] = err_roi_bit
            bit_err_arr[batch_idx, 1:(network_res_log2 + 1)] = err_x_bits
            bit_err_arr[batch_idx, (network_res_log2 + 1):] = err_y_bits

            # compute the segmentation accuracy
            visib_pixel_acc_arr[batch_idx] = 1.0 - compute_mask_pixelwise_error(pred_seg_visib[counter], gt_seg_visib[counter])
            visib_iou_arr[batch_idx] = compute_mask_iou(pred_seg_visib[counter], gt_seg_visib[counter])
            full_pixel_acc_arr[batch_idx] = 1.0 - compute_mask_pixelwise_error(pred_seg_full[counter], gt_seg_full[counter])
            full_iou_arr[batch_idx] = compute_mask_iou(pred_seg_full[counter], gt_seg_full[counter])

            # obtain the GT xy id
            gt_x_id, gt_y_id = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(network_res_log2):
                gt_x_id += gt_x_bits[counter, :, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
                gt_y_id += gt_y_bits[counter, :, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
            gt_x_id = gt_x_id.astype(int)
            gt_y_id = gt_y_id.astype(int)
            # obtain the pred xy id
            pred_x_id, pred_y_id = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(network_res_log2):
                pred_x_id += pred_x_bits[counter, :, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
                pred_y_id += pred_y_bits[counter, :, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
            pred_x_id = pred_x_id.astype(int)
            pred_y_id = pred_y_id.astype(int)

    # summarize the results
    adx2_passed_arr = np.zeros(13)
    adx5_passed_arr = np.zeros(13)
    adx10_passed_arr = np.zeros(13)
    rete2_passed_arr = np.zeros(13)
    rete5_passed_arr = np.zeros(13)
    re2_passed_arr = np.zeros(13)
    re5_passed_arr = np.zeros(13)
    te2_passed_arr = np.zeros(13)
    te5_passed_arr = np.zeros(13)
    for i, obj_id in enumerate(lm13_obj_ids):
        adx2_passed_arr[i] = np.mean(adx2_passed_dict[obj_id])
        adx5_passed_arr[i] = np.mean(adx5_passed_dict[obj_id])
        adx10_passed_arr[i] = np.mean(adx10_passed_dict[obj_id])
        rete2_passed_arr[i] = np.mean(rete2_passed_dict[obj_id])
        rete5_passed_arr[i] = np.mean(rete5_passed_dict[obj_id])
        re2_passed_arr[i] = np.mean(re2_passed_dict[obj_id])
        re5_passed_arr[i] = np.mean(re5_passed_dict[obj_id])
        te2_passed_arr[i] = np.mean(te2_passed_dict[obj_id])
        te5_passed_arr[i] = np.mean(te5_passed_dict[obj_id])

    adx2_passed = np.mean(adx2_passed_arr)
    adx5_passed = np.mean(adx5_passed_arr)
    adx10_passed = np.mean(adx10_passed_arr)
    rete2_passed = np.mean(rete2_passed_arr)
    rete5_passed = np.mean(rete5_passed_arr)
    re2_passed = np.mean(re2_passed_arr)
    re5_passed = np.mean(re5_passed_arr)
    te2_passed = np.mean(te2_passed_arr)
    te5_passed = np.mean(te5_passed_arr)
    test_acc = adx10_passed
    roi_bit_acc = np.mean(roi_bit_acc_arr)
    reproj_x_acc = np.mean(reproj_x_acc_arr)
    reproj_y_acc = np.mean(reproj_y_acc_arr)
    bit_err_arr = np.mean(bit_err_arr, axis=0)
    visib_pixel_acc = np.mean(visib_pixel_acc_arr)
    visib_iou = np.mean(visib_iou_arr)
    full_pixel_acc = np.mean(full_pixel_acc_arr)
    full_iou = np.mean(full_iou_arr)

    test_res_str = "adx_type {}\nacc {:.4f}\nadx2 {:.4f}\nadx5 {:.4f}\nadx10 {:.4f}\n" \
                   "rete2 {:.4f}\nrete5 {:.4f}\nre2 {:.4f}\nre5 {:.4f}\nte2 {:.4f}\nte5 {:.4f}\n" \
                   "roi_bit_acc {:.4f}\nreproj_x_acc {:.4f}\nreproj_y_acc {:.4f}\nbit_err_arr {}\n" \
                   "visib_pixel_acc {:.4f}\nvisib_iou {:.4f}\nfull_pixel_acc {:.4f}\nfull_iou {:.4f}\n".format(
        adx_type, test_acc, adx2_passed, adx5_passed, adx10_passed,
        rete2_passed, rete5_passed, re2_passed, re5_passed, te2_passed, te5_passed,
        roi_bit_acc, reproj_x_acc, reproj_y_acc, bit_err_arr,
        visib_pixel_acc, visib_iou, full_pixel_acc, full_iou,
    )
    print("test score: ")
    print(test_res_str)

    # save test results to file
    path = os.path.join(eval_output_path, "score/")
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + "{}".format(dataset_name) + ".txt"
    with open(path, 'w') as f:
        f.write(test_res_str)
    print("test scores saved to {}".format(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str)  # config file
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--eval_output_path', type=str)
    parser.add_argument('--use_progressivex', action='store_true', help="use progressivex solver in inference")
    parser.add_argument('--prog_max_iters', type=int, default=400)
    parser.add_argument('--nbr_ball_radius', type=float, default=20.0)
    parser.add_argument('--spatial_coherence_weight', type=float, default=0.1)
    parser.add_argument('--adx_type', choices=["default", "full", "visib"], default="default")
    args = parser.parse_args()
    config_file = args.cfg
    configs = parse_cfg(config_file)
    configs['ckpt_file'] = args.ckpt_file
    configs['use_progressivex'] = args.use_progressivex
    configs['prog_max_iters'] = args.prog_max_iters
    configs['nbr_ball_radius'] = args.nbr_ball_radius
    configs['spatial_coherence_weight'] = args.spatial_coherence_weight
    configs['adx_type'] = args.adx_type

    config_file_name = os.path.basename(config_file)
    config_file_name = os.path.splitext(config_file_name)[0]
    if args.eval_output_path is None:
        eval_output_path = 'eval/' + config_file_name
    else:
        eval_output_path = args.eval_output_path

    configs['config_file_name'] = config_file_name
    configs['eval_output_path'] = eval_output_path

    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults

    # print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)
    main(configs)
