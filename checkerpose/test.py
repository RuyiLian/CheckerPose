''' test the pose network '''
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
from bop_dataset_pytorch import bop_dataset_single_obj_pytorch_code2d

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
from model.init import InitNet_GNN
from model.pipeline import PoseNet_GNNskip
from torch.utils.tensorboard import SummaryWriter
from checkerpose.aux_utils.pointnet2_utils import pc_normalize
from get_detection_results import get_detection_results, ycbv_select_keyframe, get_detection_scores
from common_ops import from_dim_str_to_tuple
from test_network_with_test_data import compute_mask_pixelwise_error, compute_mask_iou, from_id_to_pose
from tools_for_BOP import write_to_cvs

def compute_auc_posecnn(errors):
    # NOTE: Adapted from https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m
    errors = errors.copy()
    d = np.sort(errors)
    d[d > 0.1] = np.inf
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids - 1]) * mpre[ids]).sum() * 10
    return ap

def main(configs):
    config_file_name = configs['config_file_name']
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder = configs['training_data_folder']
    training_data_folder_2 = configs['training_data_folder_2']
    test_folder = configs['test_folder']  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']  # the percentage of second dataset in the batch
    num_workers = configs['num_workers']  # for data loader
    train_obj_visible_theshold = configs['train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values
    #### network settings
    init_network_type = configs.get("init_network_type", "naive")
    init_network_backbone_name = configs.get("init_network_backbone_name", "resnet34")
    init_pretrained_root = configs["init_pretrained_root"]  # load pretrained network
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
    reprojErr_thresh = configs['reprojErr_thresh']
    cv_max_iters = configs['cv_max_iters']

    # whether use visible masks to filter the correspondences when recording the estimated poses
    use_filter_visib = False
    if dataset_name == "lmo" and obj_name in ['can', 'cat', 'driller', 'eggbox']:
        use_filter_visib = True
    if dataset_name == "ycbv" and obj_name in ['pudding_box', 'foam_brick']:
        use_filter_visib = True
    print("discard correspondences outside visible masks: ", use_filter_visib)

    # get dataset informations
    dataset_dir, source_dir, model_plys, model_info, model_ids, rgb_files, depth_files, mask_files, mask_visib_files, gts, gt_infos, cam_param_global, cam_params = bop_io.get_dataset(
        bop_path, dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True,
        train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1)  # now the obj_id started from 0
    if obj_name in symmetry_obj:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
        main_metric_name = 'ADI'
        supp_metric_name = 'ADD'
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
        main_metric_name = 'ADD'
        supp_metric_name = 'ADI'

    mesh_path = model_plys[obj_id + 1]  # mesh_path is a dict, the obj_id should start from 1
    print(mesh_path, flush=True)
    obj_diameter = model_info[str(obj_id + 1)]['diameter']
    print("obj_diameter", obj_diameter, flush=True)
    vertices = inout.load_ply(mesh_path)["pts"]

    # load 3D keypoints
    fps_path = "datasets/BOP_DATASETS/{}/{}/obj_{:06d}.pkl".format(dataset_name, fps_version, obj_id + 1)
    print("load FPS points from {}".format(fps_path))
    fps_data = mmcv.load(fps_path)
    p3d_xyz = fps_data['xyz'][:num_p3d, :]

    # obtain the normalized 3D keypoints, in range [-1, 1]
    p3d_normed, p3d_centroid, p3d_range = pc_normalize(p3d_xyz.copy(), return_stat=True)
    print("FPS points, [before normalization] min {} max {} [after normalization] min {} max {}".format(
        p3d_xyz.min(), p3d_xyz.max(), p3d_normed.min(), p3d_normed.max()
    ))
    p3d_normed = torch.as_tensor(p3d_normed, dtype=torch.float32).transpose(1, 0).unsqueeze(dim=0)  # shape: (1, 3, n)
    if torch.cuda.is_available():
        p3d_normed = p3d_normed.cuda()

    # define test data loader
    if not bop_challange:
        dataset_dir_test, _, _, _, _, test_rgb_files, _, test_mask_files, test_mask_visib_files, test_gts, test_gt_infos, _, camera_params_test = bop_io.get_dataset(
            bop_path, dataset_name, train=False, data_folder=test_folder, data_per_obj=True, incl_param=True,
            train_obj_visible_theshold=train_obj_visible_theshold)
        if dataset_name == 'ycbv':
            print("select key frames from ycbv test images")
            key_frame_index = ycbv_select_keyframe(Detection_reaults, test_rgb_files[obj_id])
            test_rgb_files_keyframe = [test_rgb_files[obj_id][i] for i in key_frame_index]
            test_mask_files_keyframe = [test_mask_files[obj_id][i] for i in key_frame_index]
            test_mask_visib_files_keyframe = [test_mask_visib_files[obj_id][i] for i in key_frame_index]
            test_gts_keyframe = [test_gts[obj_id][i] for i in key_frame_index]
            test_gt_infos_keyframe = [test_gt_infos[obj_id][i] for i in key_frame_index]
            camera_params_test_keyframe = [camera_params_test[obj_id][i] for i in key_frame_index]
            test_rgb_files[obj_id] = test_rgb_files_keyframe
            test_mask_files[obj_id] = test_mask_files_keyframe
            test_mask_visib_files[obj_id] = test_mask_visib_files_keyframe
            test_gts[obj_id] = test_gts_keyframe
            test_gt_infos[obj_id] = test_gt_infos_keyframe
            camera_params_test[obj_id] = camera_params_test_keyframe
    else:
        dataset_dir_test, _, _, _, _, test_rgb_files, _, test_mask_files, test_mask_visib_files, test_gts, test_gt_infos, _, camera_params_test = bop_io.get_bop_challange_test_data(
            bop_path, dataset_name, target_obj_id=obj_id + 1, data_folder=test_folder)
    print('test_rgb_file exsample', test_rgb_files[obj_id][0])

    if Detection_reaults != 'none':
        Det_Bbox = get_detection_results(Detection_reaults, test_rgb_files[obj_id], obj_id + 1, 0)
        scores = get_detection_scores(Detection_reaults, test_rgb_files[obj_id], obj_id + 1, 0)
    else:
        Det_Bbox = None

    test_dataset = bop_dataset_single_obj_pytorch_code2d(
        dataset_dir_test, test_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id],
        test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False,
        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, p3d_xyz,
        padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox=Det_Bbox,
        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
    )
    print("number of test images: ", len(test_dataset), flush=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    #############build the network
    # first create the initial network (no need to load weights here)
    if init_network_type == "GNN":
        init_net = InitNet_GNN(npoint=num_p3d, p3d_normed=p3d_normed, res_log2=3,
                               backbone_name=init_network_backbone_name, num_conv1x1=init_network_num_conv1x1,
                               max_batch_size=batch_size, num_graph_module=init_network_num_graph_module,
                               graph_k=init_network_graph_k, graph_leaky_slope=init_network_graph_leaky_slope)
    else:
        raise ValueError("init network type {} not supported in test".format(init_network_type))

    # create the full pipeline
    network_query_dims = from_dim_str_to_tuple(network_query_dims_str)
    if isinstance(network_num_graph_module, str):
        network_num_graph_module = from_dim_str_to_tuple(network_num_graph_module)
    if network_type == "vanilla_GNNskip":
        net = PoseNet_GNNskip(init_net=init_net, npoint=num_p3d, p3d_normed=p3d_normed, res_log2=network_res_log2,
                              num_filters=network_num_filters, max_batch_size=batch_size, query_dims=network_query_dims,
                              local_k=network_local_k, leaky_slope=network_leaky_slope,
                              num_graph_module=network_num_graph_module, graph_k=network_graph_k,
                              graph_leaky_slope=network_graph_leaky_slope, query_type=network_query_type)
    else:
        raise ValueError("network type {} not supported in test".format(network_type))
    if torch.cuda.is_available():
        net = net.cuda()
    print("PoseNet: ", net)

    net_ckpt = torch.load(ckpt_file)
    net.load_state_dict(net_ckpt['model_state_dict'])
    net.eval()

    # test the network
    calc_add_and_adi = True if dataset_name == "ycbv" else False
    activation_function = nn.Sigmoid()  # for inference: convert prediction to probability
    seg_size = (2 ** network_res_log2, 2 ** network_res_log2)  # size of segmentation masks
    adx2_passed_arr = np.zeros(len(test_loader.dataset))
    adx5_passed_arr = np.zeros(len(test_loader.dataset))
    adx10_passed_arr = np.zeros(len(test_loader.dataset))
    adx_err_arr = np.zeros(len(test_loader.dataset))
    rot_err_arr = np.zeros(len(test_loader.dataset))
    trans_err_arr = np.zeros(len(test_loader.dataset))
    if calc_add_and_adi:
        ady_error_arr = np.zeros(len(test_loader.dataset))
    full_adx2_passed_arr = np.zeros(len(test_loader.dataset))
    full_adx5_passed_arr = np.zeros(len(test_loader.dataset))
    full_adx10_passed_arr = np.zeros(len(test_loader.dataset))
    full_adx_err_arr = np.zeros(len(test_loader.dataset))
    full_rot_err_arr = np.zeros(len(test_loader.dataset))
    full_trans_err_arr = np.zeros(len(test_loader.dataset))
    if calc_add_and_adi:
        full_ady_error_arr = np.zeros(len(test_loader.dataset))
    visib_adx2_passed_arr = np.zeros(len(test_loader.dataset))
    visib_adx5_passed_arr = np.zeros(len(test_loader.dataset))
    visib_adx10_passed_arr = np.zeros(len(test_loader.dataset))
    visib_adx_err_arr = np.zeros(len(test_loader.dataset))
    visib_rot_err_arr = np.zeros(len(test_loader.dataset))
    visib_trans_err_arr = np.zeros(len(test_loader.dataset))
    if calc_add_and_adi:
        visib_ady_error_arr = np.zeros(len(test_loader.dataset))
    roi_bit_acc_arr = np.zeros(len(test_loader.dataset))
    reproj_x_acc_arr = np.zeros(len(test_loader.dataset))
    reproj_y_acc_arr = np.zeros(len(test_loader.dataset))
    bit_err_arr = np.zeros((len(test_loader.dataset), 2 * network_res_log2 + 1))
    visib_pixel_acc_arr = np.zeros(len(test_loader.dataset))
    visib_iou_arr = np.zeros(len(test_loader.dataset))
    full_pixel_acc_arr = np.zeros(len(test_loader.dataset))
    full_iou_arr = np.zeros(len(test_loader.dataset))

    img_ids = []
    scene_ids = []
    estimated_Rs = []
    estimated_Ts = []
    for rgb_fn in test_rgb_files[obj_id]:
        rgb_fn = rgb_fn.split("/")
        scene_id = rgb_fn[-3]
        img_id = rgb_fn[-1].split(".")[0]
        img_ids.append(img_id)
        scene_ids.append(scene_id)

    print("test dataset", flush=True)
    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, cam_Ks,
                    roi_mask_bits, pixel_x_codes, pixel_y_codes, roi_xy_oris) in enumerate(tqdm(test_loader)):
        # do the prediction and get the predicted binary code
        if torch.cuda.is_available():
            data = data.cuda()
            roi_mask_bits = roi_mask_bits.cuda()
            pixel_x_codes = pixel_x_codes.cuda()
            pixel_y_codes = pixel_y_codes.cuda()

        cur_batch_size = data.shape[0]
        batch_p3d_normed = p3d_normed.expand(cur_batch_size, -1, -1)  # shape: (batch, 3, #keypoint)
        pred_roi_bit, pred_x_bits, pred_y_bits, pred_seg, pred_x_id, pred_y_id = net(data, batch_p3d_normed)

        # split to roi bit, x_bit and y_bit
        num_proj_bits = pred_x_bits.shape[1]
        pred_roi_bit = activation_function(pred_roi_bit)
        pred_roi_bit = torch.where(pred_roi_bit > 0.5, 1.0, 0.0)
        pred_x_bits = activation_function(pred_x_bits)
        pred_x_bits = torch.where(pred_x_bits > 0.5, 1.0, 0.0)
        pred_y_bits = activation_function(pred_y_bits)
        pred_y_bits = torch.where(pred_y_bits > 0.5, 1.0, 0.0)
        gt_x_bits = pixel_x_codes[:, :num_proj_bits]
        gt_y_bits = pixel_y_codes[:, :num_proj_bits]
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
        # for pose estimation
        pred_pose_seg_visib = pred_seg[:, 0, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        pred_pose_seg_full = pred_seg[:, 1, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
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

        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            # compute pose
            roi_xy_ori = roi_xy_oris[counter]
            # pose using all 3D-2D correspondences that are in the RoI
            pred_rot, pred_trans = from_id_to_pose(p3d_xyz=p3d_xyz, roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                   roi_mask_bit=pred_roi_bit[counter],
                                                   pixel_x_id=pred_x_id[counter],
                                                   pixel_y_id=pred_y_id[counter], check_seg=False,
                                                   use_progressivex=use_progressivex,
                                                   neighborhood_ball_radius=nbr_ball_radius,
                                                   spatial_coherence_weight=spatial_coherence_weight,
                                                   reprojErr_thresh=reprojErr_thresh,
                                                   cv_max_iters=cv_max_iters,
                                                   prog_max_iters=prog_max_iters)
            # discard correspondences that are out of predicted full segmentation masks
            pred_full_rot, pred_full_trans = from_id_to_pose(p3d_xyz=p3d_xyz, roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                             roi_mask_bit=pred_roi_bit[counter],
                                                             pixel_x_id=pred_x_id[counter],
                                                             pixel_y_id=pred_y_id[counter], check_seg=True,
                                                             seg_mask=pred_pose_seg_full[counter],
                                                             use_progressivex=use_progressivex,
                                                             neighborhood_ball_radius=nbr_ball_radius,
                                                             spatial_coherence_weight=spatial_coherence_weight,
                                                             reprojErr_thresh=reprojErr_thresh,
                                                             cv_max_iters=cv_max_iters,
                                                             prog_max_iters=prog_max_iters)
            # discard correspondences that are out of predicted visible segmentation masks
            pred_visib_rot, pred_visib_trans = from_id_to_pose(p3d_xyz=p3d_xyz, roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                               roi_mask_bit=pred_roi_bit[counter],
                                                               pixel_x_id=pred_x_id[counter],
                                                               pixel_y_id=pred_y_id[counter], check_seg=True,
                                                               seg_mask=pred_pose_seg_visib[counter],
                                                               use_progressivex=use_progressivex,
                                                               neighborhood_ball_radius=nbr_ball_radius,
                                                               spatial_coherence_weight=spatial_coherence_weight,
                                                               reprojErr_thresh=reprojErr_thresh,
                                                               cv_max_iters=cv_max_iters,
                                                               prog_max_iters=prog_max_iters)

            if use_filter_visib:
                estimated_Rs.append(pred_visib_rot)
                estimated_Ts.append(pred_visib_trans)
            else:
                estimated_Rs.append(pred_rot)
                estimated_Ts.append(pred_trans)

            # compute pose error
            adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, pred_rot, pred_trans, vertices)
            if np.isnan(adx_error):
                adx_error = 10000
            adx_err_arr[batch_idx] = adx_error
            if adx_error < obj_diameter * 0.02:
                adx2_passed_arr[batch_idx] = 1
            if adx_error < obj_diameter * 0.05:
                adx5_passed_arr[batch_idx] = 1
            if adx_error < obj_diameter * 0.1:
                adx10_passed_arr[batch_idx] = 1
            rot_err_arr[batch_idx] = pose_error.re(r_GT, pred_rot)
            trans_err_arr[batch_idx] = pose_error.te(t_GT, pred_trans)
            if calc_add_and_adi:
                ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, pred_rot, pred_trans, vertices)
                if np.isnan(ady_error):
                    ady_error = 10000
                ady_error_arr[batch_idx] = ady_error

            full_adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, pred_full_rot, pred_full_trans, vertices)
            if np.isnan(full_adx_error):
                full_adx_error = 10000
            full_adx_err_arr[batch_idx] = full_adx_error
            if full_adx_error < obj_diameter * 0.02:
                full_adx2_passed_arr[batch_idx] = 1
            if full_adx_error < obj_diameter * 0.05:
                full_adx5_passed_arr[batch_idx] = 1
            if full_adx_error < obj_diameter * 0.1:
                full_adx10_passed_arr[batch_idx] = 1
            full_rot_err_arr[batch_idx] = pose_error.re(r_GT, pred_full_rot)
            full_trans_err_arr[batch_idx] = pose_error.te(t_GT, pred_full_trans)
            if calc_add_and_adi:
                full_ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, pred_full_rot, pred_full_trans, vertices)
                if np.isnan(full_ady_error):
                    full_ady_error = 10000
                full_ady_error_arr[batch_idx] = full_ady_error

            visib_adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, pred_visib_rot, pred_visib_trans, vertices)
            if np.isnan(visib_adx_error):
                visib_adx_error = 10000
            visib_adx_err_arr[batch_idx] = visib_adx_error
            if visib_adx_error < obj_diameter * 0.02:
                visib_adx2_passed_arr[batch_idx] = 1
            if visib_adx_error < obj_diameter * 0.05:
                visib_adx5_passed_arr[batch_idx] = 1
            if visib_adx_error < obj_diameter * 0.1:
                visib_adx10_passed_arr[batch_idx] = 1
            visib_rot_err_arr[batch_idx] = pose_error.re(r_GT, pred_visib_rot)
            visib_trans_err_arr[batch_idx] = pose_error.te(t_GT, pred_visib_trans)
            if calc_add_and_adi:
                visib_ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, pred_visib_rot, pred_visib_trans, vertices)
                if np.isnan(visib_ady_error):
                    visib_ady_error = 10000
                visib_ady_error_arr[batch_idx] = visib_ady_error

            # compute the reprojection accuracy
            npoint_in_roi = np.clip(gt_roi_bit[counter].sum(), a_min=1.0, a_max=None)
            err_roi_bit = np.mean(np.abs(gt_roi_bit[counter] - pred_roi_bit[counter]))
            roi_bit_acc_arr[batch_idx] = 1.0 - err_roi_bit
            diff_x_bits = (gt_x_bits[counter] - pred_x_bits[counter]) * gt_roi_bit[counter]
            diff_y_bits = (gt_y_bits[counter] - pred_y_bits[counter]) * gt_roi_bit[counter]
            reproj_err_x, reproj_err_y = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(num_proj_bits):
                reproj_err_x += diff_x_bits[:, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
                reproj_err_y += diff_y_bits[:, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
            reproj_err_x = np.sum(np.abs(reproj_err_x)) / npoint_in_roi
            reproj_err_y = np.sum(np.abs(reproj_err_y)) / npoint_in_roi
            reproj_x_acc_arr[batch_idx] = 1.0 - reproj_err_x / (2 ** num_proj_bits)
            reproj_y_acc_arr[batch_idx] = 1.0 - reproj_err_y / (2 ** num_proj_bits)
            # compute the bit-wise error
            err_x_bits = np.sum(np.abs(diff_x_bits), axis=0) / npoint_in_roi
            err_y_bits = np.sum(np.abs(diff_y_bits), axis=0) / npoint_in_roi
            bit_err_arr[batch_idx, 0] = err_roi_bit
            bit_err_arr[batch_idx, 1:(num_proj_bits + 1)] = err_x_bits
            bit_err_arr[batch_idx, (num_proj_bits + 1):(2 * num_proj_bits + 1)] = err_y_bits

            # compute the segmentation accuracy
            visib_pixel_acc_arr[batch_idx] = 1.0 - compute_mask_pixelwise_error(pred_seg_visib[counter], gt_seg_visib[counter])
            visib_iou_arr[batch_idx] = compute_mask_iou(pred_seg_visib[counter], gt_seg_visib[counter])
            full_pixel_acc_arr[batch_idx] = 1.0 - compute_mask_pixelwise_error(pred_seg_full[counter], gt_seg_full[counter])
            full_iou_arr[batch_idx] = compute_mask_iou(pred_seg_full[counter], gt_seg_full[counter])

            # obtain the GT xy id
            gt_x_id, gt_y_id = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(num_proj_bits):
                gt_x_id += gt_x_bits[counter, :, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
                gt_y_id += gt_y_bits[counter, :, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
            gt_x_id = gt_x_id.astype(int)
            gt_y_id = gt_y_id.astype(int)
            # obtain the pred xy id
            pred_x_id, pred_y_id = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(num_proj_bits):
                pred_x_id += pred_x_bits[counter, :, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
                pred_y_id += pred_y_bits[counter, :, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
            pred_x_id = pred_x_id.astype(int)
            pred_y_id = pred_y_id.astype(int)

    adx2_passed = np.mean(adx2_passed_arr)
    adx5_passed = np.mean(adx5_passed_arr)
    adx10_passed = np.mean(adx10_passed_arr)
    adx_err = np.mean(adx_err_arr)
    rot_err = np.mean(rot_err_arr)
    trans_err = np.mean(trans_err_arr)
    AUC_ADX_error_posecnn = compute_auc_posecnn(adx_err_arr / 1000.)
    full_adx2_passed = np.mean(full_adx2_passed_arr)
    full_adx5_passed = np.mean(full_adx5_passed_arr)
    full_adx10_passed = np.mean(full_adx10_passed_arr)
    full_adx_err = np.mean(full_adx_err_arr)
    full_rot_err = np.mean(full_rot_err_arr)
    full_trans_err = np.mean(full_trans_err_arr)
    full_AUC_ADX_error_posecnn = compute_auc_posecnn(full_adx_err_arr / 1000.)
    visib_adx2_passed = np.mean(visib_adx2_passed_arr)
    visib_adx5_passed = np.mean(visib_adx5_passed_arr)
    visib_adx10_passed = np.mean(visib_adx10_passed_arr)
    visib_adx_err = np.mean(visib_adx_err_arr)
    visib_rot_err = np.mean(visib_rot_err_arr)
    visib_trans_err = np.mean(visib_trans_err_arr)
    visib_AUC_ADX_error_posecnn = compute_auc_posecnn(visib_adx_err_arr / 1000.)
    test_acc = adx10_passed
    roi_bit_acc = np.mean(roi_bit_acc_arr)
    reproj_x_acc = np.mean(reproj_x_acc_arr)
    reproj_y_acc = np.mean(reproj_y_acc_arr)
    bit_err_arr = np.mean(bit_err_arr, axis=0)
    visib_pixel_acc = np.mean(visib_pixel_acc_arr)
    visib_iou = np.mean(visib_iou_arr)
    full_pixel_acc = np.mean(full_pixel_acc_arr)
    full_iou = np.mean(full_iou_arr)

    test_res_str = "acc {:.4f}\nadx2 {:.4f}\nadx5 {:.4f}\nadx10 {:.4f}\nadx_err {:.4f}\nre {:.4f}\nte {:.4f}\n" \
                   "full_adx2 {:.4f}\nfull_adx5 {:.4f}\nfull_adx10 {:.4f}\nfull_adx_err {:.4f}\nfull_re {:.4f}\nfull_te {:.4f}\n" \
                   "visib_adx2 {:.4f}\nvisib_adx5 {:.4f}\nvisib_adx10 {:.4f}\nvisib_adx_err {:.4f}\nvisib_re {:.4f}\nvisib_te {:.4f}\n" \
                   "roi_bit_acc {:.4f}\nreproj_x_acc {:.4f}\nreproj_y_acc {:.4f}\nbit_err_arr {}\n" \
                   "visib_pixel_acc {:.4f}\nvisib_iou {:.4f}\nfull_pixel_acc {:.4f}\nfull_iou {:.4f}\n".format(
        test_acc, adx2_passed, adx5_passed, adx10_passed, adx_err, rot_err, trans_err,
        full_adx2_passed, full_adx5_passed, full_adx10_passed, full_adx_err, full_rot_err, full_trans_err,
        visib_adx2_passed, visib_adx5_passed, visib_adx10_passed, visib_adx_err, visib_rot_err, visib_trans_err,
        roi_bit_acc, reproj_x_acc, reproj_y_acc, bit_err_arr,
        visib_pixel_acc, visib_iou, full_pixel_acc, full_iou,
    )
    test_res_str += "AUC_posecnn_{} {:.4f}\n".format(main_metric_name, AUC_ADX_error_posecnn)
    test_res_str += "full_AUC_posecnn_{} {:.4f}\n".format(main_metric_name, full_AUC_ADX_error_posecnn)
    test_res_str += "visib_AUC_posecnn_{} {:.4f}\n".format(main_metric_name, visib_AUC_ADX_error_posecnn)
    if calc_add_and_adi:
        AUC_ADY_error_posecnn = compute_auc_posecnn(ady_error_arr / 1000.)
        test_res_str += "AUC_posecnn_{} {:.4f}\n".format(supp_metric_name, AUC_ADY_error_posecnn)
        full_AUC_ADY_error_posecnn = compute_auc_posecnn(full_ady_error_arr / 1000.)
        test_res_str += "full_AUC_posecnn_{} {:.4f}\n".format(supp_metric_name, full_AUC_ADY_error_posecnn)
        visib_AUC_ADY_error_posecnn = compute_auc_posecnn(visib_ady_error_arr / 1000.)
        test_res_str += "visib_AUC_posecnn_{} {:.4f}\n".format(supp_metric_name, visib_AUC_ADY_error_posecnn)

    print("test score: ")
    print(test_res_str)

    # save test results to file
    path = os.path.join(eval_output_path, "score/")
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + "{}_{}".format(dataset_name, obj_name) + ".txt"
    with open(path, 'w') as f:
        f.write(test_res_str)
    print("test scores saved to {}".format(path))

    # save the estimated poses as well
    cvs_path = os.path.join(eval_output_path, 'pose_result_bop/')
    if not os.path.exists(cvs_path):
        os.makedirs(cvs_path)
    write_to_cvs.write_cvs(cvs_path, "{}_{}".format(dataset_name, obj_name), obj_id + 1, scene_ids, img_ids,
                           estimated_Rs, estimated_Ts, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str)  # config file
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--eval_output_path', type=str)
    parser.add_argument('--use_progressivex', action='store_true', help="use progressivex solver in inference")
    parser.add_argument('--prog_max_iters', type=int, default=400)
    parser.add_argument('--nbr_ball_radius', type=float, default=20.0)
    parser.add_argument('--spatial_coherence_weight', type=float, default=0.1)
    parser.add_argument('--reprojErr_thresh', type=float, default=2)
    parser.add_argument('--cv_max_iters', type=int, default=150)
    args = parser.parse_args()
    config_file = args.cfg
    configs = parse_cfg(config_file)
    configs['obj_name'] = args.obj_name
    configs['ckpt_file'] = args.ckpt_file
    configs['use_progressivex'] = args.use_progressivex
    configs['prog_max_iters'] = args.prog_max_iters
    configs['nbr_ball_radius'] = args.nbr_ball_radius
    configs['spatial_coherence_weight'] = args.spatial_coherence_weight
    configs['reprojErr_thresh'] = args.reprojErr_thresh
    configs['cv_max_iters'] = args.cv_max_iters

    config_file_name = os.path.basename(config_file)
    config_file_name = os.path.splitext(config_file_name)[0]
    if args.eval_output_path is None:
        eval_output_path = 'eval/' + config_file_name + '/' + args.obj_name
    else:
        eval_output_path = args.eval_output_path + '/' + args.obj_name

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
