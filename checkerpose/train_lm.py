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
from lm_dataset_pytorch import load_lm_obj_diameters, lm_dataset_single_obj_pytorch_code2d
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import glob

sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout
from model.init_lm import InitNet_GNN
from model.pipeline_lm import PoseNet_GNNskip, PoseNet_GNNskip_ABwoProg
from losses.code_loss import MaskedCodeLoss, UnmaskedCodeLoss
from losses.mask_loss import MaskLoss_interpolate
from torch.utils.tensorboard import SummaryWriter
from checkerpose.aux_utils.pointnet2_utils import pc_normalize
from utils import save_checkpoint, get_checkpoint, save_best_checkpoint
from common_ops import get_batch_size, from_dim_str_to_tuple
from test_network_with_test_data import test_pipeline_lm

# stage_start_steps: tuple of the start steps of each training stage
# note: stage starts from 1
def get_train_stage(step, stage_start_steps_tup):
    stage = 0
    for start_step in stage_start_steps_tup:
        if step < start_step:
            break
        stage += 1
    return stage

def main(configs):
    datasets_root = configs['datasets_root']
    dataset_name = 'lm'
    config_file_name = configs['config_file_name']
    #### training dataset
    training_data_folder = configs['training_data_folder']
    training_data_folder_2 = configs['training_data_folder_2']
    val_folder = configs['val_folder']  # usually is 'test'
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
    network_res_log2 = configs.get("network_res_log2", 6)
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
    load_checkpoint = configs['load_checkpoint']
    tensorboard_path = configs['tensorboard_path']
    check_point_path = configs['check_point_path']
    #### optimizer
    optimizer_type = configs['optimizer_type']  # Adam is the best sofar
    total_iteration = configs['total_iteration']  # train how many steps
    stage_start_steps_str = configs.get("stage_start_steps_str", "0_1000_5000")  # from which training step a new stage starts
    stage_start_steps = from_dim_str_to_tuple(stage_start_steps_str)
    batch_size = configs['batch_size']  # 32 is the best so far, set to 16 for debug in local machine
    learning_rate = configs['learning_rate']  # 0.002 or 0.003 is the best so far
    learning_rate2 = configs['learning_rate2']  # reduced learning rate near the end of the training
    learning_rate2_start = configs['learning_rate2_start']  # iteration when using the reduced learning rate
    RoiBit_Loss_Type = configs['RoiBit_Loss_Type']  # "BCE" | "L1" (for the 0th bit, indicting in/out RoI)
    ProjBit_Loss_Type = configs['ProjBit_Loss_Type']  # "BCE" | "L1" (for the remaining bits, indicting the 2D
    seg_visib_loss_weight = configs['seg_visib_loss_weight']
    seg_full_loss_weight = configs['seg_full_loss_weight']
    #### augmentations
    Detection_reaults = configs['Detection_reaults']  # for the test, the detected bounding box provided by GDR Net
    padding_ratio = configs['padding_ratio']  # pad the bounding box for training and test
    resize_method = configs['resize_method']  # how to resize the roi images to 256*256
    use_peper_salt = configs['use_peper_salt']  # if add additional peper_salt in the augmentation
    use_motion_blur = configs['use_motion_blur']  # if add additional motion_blur in the augmentation
    #### 3D keypoints
    num_p3d = int(2 ** configs['num_p3d_log2'])
    fps_version = configs.get("fps_version", "fps_202212")

    # model info
    symmetry_ids = [10, 11, 7, 3]
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

    ########################## define data loader
    batch_size_1_dataset, batch_size_2_dataset = get_batch_size(second_dataset_ratio, batch_size)

    train_dataset = lm_dataset_single_obj_pytorch_code2d(
        datasets_root, training_data_folder, True,
        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, lm_p3d_xyz,
        padding_ratio=padding_ratio, resize_method=resize_method,
        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
    )

    if training_data_folder_2 != 'none':
        train_dataset_2 = lm_dataset_single_obj_pytorch_code2d(
            datasets_root, training_data_folder_2, True,
            BoundingBox_CropSize_image, BoundingBox_CropSize_GT, lm_p3d_xyz,
            padding_ratio=padding_ratio, resize_method=resize_method,
            use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
        )
        train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size_2_dataset, shuffle=True,
                                                     num_workers=num_workers, drop_last=True)
        train_loader_2_iter = iter(train_loader_2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_1_dataset, shuffle=True,
                                                   num_workers=num_workers, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, drop_last=True)

    # define test data loader
    test_dataset = lm_dataset_single_obj_pytorch_code2d(
        datasets_root, val_folder, False,
        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, lm_p3d_xyz,
        padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox_file=Detection_reaults,
        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
    )
    print("number of test images: ", len(test_dataset), flush=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    #############build the network
    # first load pretrained initial network
    if init_network_type == "GNN":
        init_net = InitNet_GNN(npoint=num_p3d, p3d_normed=lm_p3d_normed, res_log2=3,
                               backbone_name=init_network_backbone_name, num_conv1x1=init_network_num_conv1x1,
                               max_batch_size=batch_size, num_graph_module=init_network_num_graph_module,
                               graph_k=init_network_graph_k, graph_leaky_slope=init_network_graph_leaky_slope,
                               pretrain_backbone=False)
    else:
        raise ValueError("init network type {} not supported in train_lm".format(init_network_type))
    if init_pretrained_root is not None:  # load the last checkpoint
        pretrain_ckpt_path = os.path.join(init_pretrained_root)
        pretrain_ckpt = torch.load(get_checkpoint(pretrain_ckpt_path))
        missing_keys, unexpected_keys = init_net.load_state_dict(pretrain_ckpt['model_state_dict'], strict=False)
        print("use pretrained feature from {}\nmissing keys: {}\nunexpected_keys: {}".format(
            init_pretrained_root, missing_keys, unexpected_keys
        ))  # for debug
    else:
        print("not using pretrained init model!")

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
        raise ValueError("network type {} not supported in train_lm".format(network_type))
    if torch.cuda.is_available():
        net = net.cuda()
    print("PoseNet: ", net)

    # count parameters
    num_params = sum(p.numel() for p in net.parameters()) / 1e6
    print("#parameters: {}M".format(num_params))

    roi_bit_loss_func = UnmaskedCodeLoss(loss_type=RoiBit_Loss_Type)
    proj_bit_loss_func = MaskedCodeLoss(loss_type=ProjBit_Loss_Type)
    seg_loss_func = MaskLoss_interpolate()
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(f"unknown optimizer type: {optimizer_type}")

    # visulize input image, ground truth code, ground truth mask
    writer = SummaryWriter(tensorboard_path)

    best_score_path = os.path.join(check_point_path, 'best_score')
    if not os.path.isdir(best_score_path):
        os.makedirs(best_score_path)
    best_score = 0
    iteration_step = 0
    if load_checkpoint:
        print("load_checkpoint from check_point_path: ", check_point_path)
        checkpoint = torch.load(get_checkpoint(check_point_path))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_score = checkpoint['best_score']
        iteration_step = checkpoint['iteration_step']

    # train the network
    while True:
        end_training = False
        for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, cam_Ks, obj_ids,
                        roi_mask_bits, pixel_x_codes, pixel_y_codes, roi_xy_oris) in enumerate(train_loader):
            step_start_time = time.time()
            cur_train_stage = get_train_stage(iteration_step, stage_start_steps)  # current training stage (i.e. number of active refinement blocks)
            if iteration_step == learning_rate2_start:  # a naive way to reduce the learning rate near the end
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate2
            # if multiple training sets, get data from the second set
            if training_data_folder_2 != 'none':
                try:
                    data_2, entire_masks_2, masks_2, Rs_2, ts_2, Bboxes_2, cam_Ks_2, obj_ids_2, \
                    roi_mask_bits_2, pixel_x_codes_2, pixel_y_codes_2, roi_xy_oris_2 = next(train_loader_2_iter)
                except StopIteration:
                    train_loader_2_iter = iter(train_loader_2)
                    data_2, entire_masks_2, masks_2, Rs_2, ts_2, Bboxes_2, cam_Ks_2, obj_ids_2, \
                    roi_mask_bits_2, pixel_x_codes_2, pixel_y_codes_2, roi_xy_oris_2 = next(train_loader_2_iter)

                data = torch.cat((data, data_2), 0)
                entire_masks = torch.cat((entire_masks, entire_masks_2), 0)
                masks = torch.cat((masks, masks_2), 0)
                Rs = torch.cat((Rs, Rs_2), 0)
                ts = torch.cat((ts, ts_2), 0)
                cam_Ks = torch.cat((cam_Ks, cam_Ks_2), 0)
                Bboxes = torch.cat((Bboxes, Bboxes_2), 0)
                roi_mask_bits = torch.cat((roi_mask_bits, roi_mask_bits_2), 0)
                pixel_x_codes = torch.cat((pixel_x_codes, pixel_x_codes_2), 0)
                pixel_y_codes = torch.cat((pixel_y_codes, pixel_y_codes_2), 0)
                roi_xy_oris = torch.cat((roi_xy_oris, roi_xy_oris_2), 0)
                obj_ids = torch.cat((obj_ids, obj_ids_2), 0)
            # data to GPU
            if torch.cuda.is_available():
                data = data.cuda()
                entire_masks = entire_masks.cuda()
                masks = masks.cuda()
                roi_mask_bits = roi_mask_bits.cuda()
                pixel_x_codes = pixel_x_codes.cuda()
                pixel_y_codes = pixel_y_codes.cuda()
                obj_ids = obj_ids.cuda()

            optimizer.zero_grad()
            if data.shape[0] != batch_size:
                raise ValueError(f"batch size wrong")
            batch_p3d_normed = lm_p3d_normed[obj_ids-1]  # shape: (batch, 3, #keypoint) (note: obj_ids start from 1)
            pred_roi_bit, pred_x_bits, pred_y_bits, pred_seg, _, _ = net(data, batch_p3d_normed, obj_ids, cur_train_stage)

            # split the binary code to compute the loss
            loss_roi_bit = roi_bit_loss_func(pred_roi_bit, roi_mask_bits)
            num_proj_bits = pred_x_bits.shape[1]
            loss_proj_bit_x = proj_bit_loss_func(pred_x_bits, pixel_x_codes[:, :num_proj_bits], roi_mask_bits)
            loss_proj_bit_y = proj_bit_loss_func(pred_y_bits, pixel_y_codes[:, :num_proj_bits], roi_mask_bits)
            # image segmentation mask loss
            loss_seg_visib = seg_loss_func(pred_seg[:, 0:1, :, :], masks)
            loss_seg_full = seg_loss_func(pred_seg[:, 1:2, :, :], entire_masks)
            loss = loss_roi_bit + loss_proj_bit_x + loss_proj_bit_y + \
                loss_seg_visib * seg_visib_loss_weight + loss_seg_full * seg_full_loss_weight
            loss.backward()
            optimizer.step()

            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            est_time = step_time * (total_iteration - iteration_step - 1) / 3600.0  # in hours
            print(datetime.datetime.now(), config_file_name,
                  "iteration_step:", iteration_step, "train_stage:", cur_train_stage,
                  "loss_roi_bit:", loss_roi_bit.item(),
                  "loss_proj_bit_x:", loss_proj_bit_x.item(),
                  "loss_proj_bit_y:", loss_proj_bit_y.item(),
                  "loss_seg_visib:", loss_seg_visib.item(),
                  "loss_seg_full:", loss_seg_full.item(),
                  "loss:", loss.item(),
                  "time: {:.6f}s".format(step_time),
                  "est: {:.3f}h".format(est_time),
                  flush=True
                  )
            writer.add_scalar('Loss/training loss total', loss, iteration_step)
            writer.add_scalar('Loss/training loss roi_bit', loss_roi_bit, iteration_step)
            writer.add_scalar('Loss/training loss proj_bit_x', loss_proj_bit_x, iteration_step)
            writer.add_scalar('Loss/training loss proj_bit_y', loss_proj_bit_y, iteration_step)
            writer.add_scalar('Loss/training loss seg_visib', loss_seg_visib, iteration_step)
            writer.add_scalar('Loss/training loss seg_full', loss_seg_full, iteration_step)

            # test the trained CNN
            log_freq = 1000
            test_freq = 10000
            if (iteration_step) % log_freq == 0 or (iteration_step == total_iteration - 1):
                save_checkpoint(check_point_path, net, iteration_step, best_score, optimizer, 3)

            if (iteration_step) % test_freq == 0 or (iteration_step == total_iteration - 1):
                test_acc, adx2_passed, adx5_passed, adx10_passed, \
                    roi_bit_acc, reproj_x_acc, reproj_y_acc, bit_err_arr, \
                    test_visib_pixel_acc, test_visib_iou, test_full_pixel_acc, test_full_iou = \
                    test_pipeline_lm(net, test_loader, writer, iteration_step, configs, lm_p3d_normed, lm_p3d_xyz,
                                     vertices_dict, obj_diameter_dict, symmetry_ids, cur_train_stage)
                with np.printoptions(precision=4):
                    print("[test] acc {:.4f} adx2 {:.4f} adx5 {:.4f} adx10 {:.4f} "
                          "roi_bit_acc {:.4f} reproj_x_acc {:.4f} reproj_y_acc {:.4f} bit_err_arr {} "
                          "|| visib: pixel_acc {:.4f} IoU {:.4f} || full: pixel_acc {:.4f} IoU {:.4f}".format(
                        test_acc, adx2_passed, adx5_passed, adx10_passed,
                        roi_bit_acc, reproj_x_acc, reproj_y_acc, bit_err_arr,
                        test_visib_pixel_acc, test_visib_iou, test_full_pixel_acc, test_full_iou), flush=True)
                if test_acc >= best_score:
                    best_score = test_acc
                    print("best_score", best_score)
                    save_best_checkpoint(best_score_path, net, optimizer, best_score, iteration_step)

            iteration_step = iteration_step + 1
            if iteration_step >= total_iteration:
                end_training = True
                break

        if end_training == True:
            print('end the training in iteration_step:', iteration_step)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str)  # config file
    parser.add_argument('--load_checkpoint', action='store_true', help="load checkpoint of the current stage")
    args = parser.parse_args()
    config_file = args.cfg
    configs = parse_cfg(config_file)
    configs['load_checkpoint'] = args.load_checkpoint

    check_point_path = configs['check_point_path']
    tensorboard_path = configs['tensorboard_path']
    config_file_name = os.path.basename(config_file)
    config_file_name = os.path.splitext(config_file_name)[0]
    check_point_path = check_point_path + config_file_name
    tensorboard_path = tensorboard_path + config_file_name
    configs['check_point_path'] = check_point_path
    configs['tensorboard_path'] = tensorboard_path
    configs['config_file_name'] = config_file_name

    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults

    # print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)
    main(configs)
