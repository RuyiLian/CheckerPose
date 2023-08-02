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
from torch import optim
import numpy as np
import glob
sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout
from model.init_lm import InitNet_GNN
from losses.code_loss import MaskedCodeLoss, UnmaskedCodeLoss
from torch.utils.tensorboard import SummaryWriter
from checkerpose.aux_utils.pointnet2_utils import pc_normalize
from utils import save_checkpoint, get_checkpoint, save_best_checkpoint
from get_detection_results import get_detection_results, ycbv_select_keyframe
from common_ops import get_batch_size
from test_network_with_test_data import test_init_lm

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
    network_type = configs.get("network_type", "naive")
    network_res_log2 = configs.get("network_res_log2", 3)
    network_backbone_name = configs.get("network_backbone_name", "resnet34")
    network_num_conv1x1 = configs.get("network_num_conv1x1", 1)
    network_num_graph_module = configs.get("network_num_graph_module", 2)  # for graph modules before query
    network_graph_k = configs.get("network_graph_k", 20)
    network_graph_leaky_slope = configs.get("network_graph_leaky_slope", 0.2)
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']  # network output size
    #### check points
    load_checkpoint = configs['load_checkpoint']
    tensorboard_path = configs['tensorboard_path']
    check_point_path = configs['check_point_path']
    #### optimizer
    optimizer_type = configs['optimizer_type']  # Adam is the best sofar
    total_iteration = configs['total_iteration']  # train how many steps
    batch_size = configs['batch_size']  # 32 is the best so far, set to 16 for debug in local machine
    learning_rate = configs['learning_rate']  # 0.002 or 0.003 is the best so far
    MaskBit_Loss_Type = configs['MaskBit_Loss_Type']  # "BCE" | "L1" (for the 0th bit, indicting in/out RoI)
    ProjBit_Loss_Type = configs['ProjBit_Loss_Type']  # "BCE" | "L1" (for the remaining bits, indicting the 2D
    proj_bit_loss_weight = configs['proj_bit_loss_weight']
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

    # load 3D points
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
    if network_type == "GNN":
        net = InitNet_GNN(npoint=num_p3d, p3d_normed=lm_p3d_normed, res_log2=network_res_log2,
                          backbone_name=network_backbone_name, num_conv1x1=network_num_conv1x1,
                          max_batch_size=batch_size, num_graph_module=network_num_graph_module,
                          graph_k=network_graph_k, graph_leaky_slope=network_graph_leaky_slope)
    else:
        raise ValueError("network type {} not supported in pretrain_lm".format(network_type))
    if torch.cuda.is_available():
        net = net.cuda()
    print("Init Net: ", net)
    num_code_bit = net.num_out_bits

    # count parameters
    num_params = sum(p.numel() for p in net.parameters()) / 1e6
    print("#parameters: {}M".format(num_params))

    mask_bit_loss_func = UnmaskedCodeLoss(loss_type=MaskBit_Loss_Type)
    proj_bit_loss_func = MaskedCodeLoss(loss_type=ProjBit_Loss_Type)
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
            pred = net(data, obj_ids)
            # split the binary code to compute the loss
            pred_mask_bit = pred[:, 0:1, :]
            pred_proj_bit_x = pred[:, 1:(network_res_log2+1), :]
            pred_proj_bit_y = pred[:, (network_res_log2+1):, :]

            loss_mask_bit = mask_bit_loss_func(pred_mask_bit, roi_mask_bits)
            loss_proj_bit_x = proj_bit_loss_func(pred_proj_bit_x, pixel_x_codes[:, :network_res_log2], roi_mask_bits)
            loss_proj_bit_y = proj_bit_loss_func(pred_proj_bit_y, pixel_y_codes[:, :network_res_log2], roi_mask_bits)
            loss = loss_mask_bit + (loss_proj_bit_x + loss_proj_bit_y) * proj_bit_loss_weight
            loss.backward()
            optimizer.step()

            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            est_time = step_time * (total_iteration - iteration_step - 1) / 3600.0  # in hours
            print(datetime.datetime.now(), config_file_name, " iteration_step:", iteration_step,
                  "loss_mask_bit:", loss_mask_bit.item(),
                  "loss_proj_bit_x:", loss_proj_bit_x.item(),
                  "loss_proj_bit_y:", loss_proj_bit_y.item(),
                  "loss:", loss.item(),
                  "time: {:.6f}s".format(step_time),
                  "est: {:.3f}h".format(est_time),
                  flush=True
                  )
            writer.add_scalar('Loss/training loss total', loss, iteration_step)
            writer.add_scalar('Loss/training loss mask_bit', loss_mask_bit, iteration_step)
            writer.add_scalar('Loss/training loss proj_bit_x', loss_proj_bit_x, iteration_step)
            writer.add_scalar('Loss/training loss proj_bit_y', loss_proj_bit_y, iteration_step)

            # test the trained CNN
            log_freq = 1000
            if (iteration_step) % log_freq == 0 or (iteration_step == total_iteration - 1):
                save_checkpoint(check_point_path, net, iteration_step, best_score, optimizer, 3)

                test_acc, test_mask_bit_acc, test_reproj_acc_x, test_reproj_acc_y, test_bitwise_err = \
                    test_init_lm(net, test_loader, writer, iteration_step, configs, best_score_metric="mean_bit_acc")
                print("[test] acc {} mask_bit_acc {} reproj_acc_x {} reproj_acc_y {} bitwise err {}".format(
                    test_acc, test_mask_bit_acc, test_reproj_acc_x, test_reproj_acc_y, test_bitwise_err), flush=True)
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
