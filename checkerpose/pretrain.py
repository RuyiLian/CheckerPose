''' pretrain the layers that generates the first 3 bits '''
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
from torch import optim
import numpy as np
import glob

sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout
from model.init import InitNet_GNN
from losses.code_loss import MaskedCodeLoss, UnmaskedCodeLoss
from torch.utils.tensorboard import SummaryWriter
from checkerpose.aux_utils.pointnet2_utils import pc_normalize
from utils import save_checkpoint, get_checkpoint, save_best_checkpoint
from get_detection_results import get_detection_results, ycbv_select_keyframe
from common_ops import get_batch_size
from test_network_with_test_data import test_init

def main(configs):
    config_file_name = configs['config_file_name']
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
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

    # get dataset informations
    dataset_dir, source_dir, model_plys, model_info, model_ids, rgb_files, depth_files, mask_files, mask_visib_files, gts, gt_infos, cam_param_global, cam_params = bop_io.get_dataset(
        bop_path, dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True,
        train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1)  # now the obj_id started from 0

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

    ########################## define data loader
    batch_size_1_dataset, batch_size_2_dataset = get_batch_size(second_dataset_ratio, batch_size)

    train_dataset = bop_dataset_single_obj_pytorch_code2d(
        dataset_dir, training_data_folder, rgb_files[obj_id], mask_files[obj_id], mask_visib_files[obj_id],
        gts[obj_id], gt_infos[obj_id], cam_params[obj_id], True, BoundingBox_CropSize_image,
        BoundingBox_CropSize_GT, p3d_xyz, padding_ratio=padding_ratio, resize_method=resize_method,
        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
    )
    print("training_data_folder image example:", rgb_files[obj_id][0], flush=True)

    if training_data_folder_2 != 'none':
        dataset_dir_pbr, _, _, _, _, rgb_files_pbr, _, mask_files_pbr, mask_visib_files_pbr, gts_pbr, gt_infos_pbr, _, camera_params_pbr = bop_io.get_dataset(
            bop_path, dataset_name, train=True, data_folder=training_data_folder_2, data_per_obj=True, incl_param=True,
            train_obj_visible_theshold=train_obj_visible_theshold)
        train_dataset_2 = bop_dataset_single_obj_pytorch_code2d(
            dataset_dir_pbr, training_data_folder_2, rgb_files_pbr[obj_id], mask_files_pbr[obj_id],
            mask_visib_files_pbr[obj_id],
            gts_pbr[obj_id], gt_infos_pbr[obj_id], camera_params_pbr[obj_id], True,
            BoundingBox_CropSize_image, BoundingBox_CropSize_GT, p3d_xyz,
            padding_ratio=padding_ratio, resize_method=resize_method,
            use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
        )
        print("training_data_folder_2 image example:", rgb_files_pbr[obj_id][0], flush=True)
        train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size_2_dataset, shuffle=True,
                                                     num_workers=num_workers, drop_last=True)
        train_loader_2_iter = iter(train_loader_2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_1_dataset, shuffle=True,
                                                   num_workers=num_workers, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, drop_last=True)

    # define test data loader
    if not bop_challange:
        dataset_dir_test, _, _, _, _, test_rgb_files, _, test_mask_files, test_mask_visib_files, test_gts, test_gt_infos, _, camera_params_test = bop_io.get_dataset(
            bop_path, dataset_name, train=False, data_folder=val_folder, data_per_obj=True, incl_param=True,
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
            bop_path, dataset_name, target_obj_id=obj_id + 1, data_folder=val_folder)
    print('test_rgb_file exsample', test_rgb_files[obj_id][0])

    if Detection_reaults != 'none':
        Det_Bbox = get_detection_results(Detection_reaults, test_rgb_files[obj_id], obj_id + 1, 0)
    else:
        Det_Bbox = None

    test_dataset = bop_dataset_single_obj_pytorch_code2d(
        dataset_dir_test, val_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id],
        test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False,
        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, p3d_xyz,
        padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox=Det_Bbox,
        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
    )
    print("number of test images: ", len(test_dataset), flush=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    #############build the network
    if network_type == "GNN":
        net = InitNet_GNN(npoint=num_p3d, p3d_normed=p3d_normed, res_log2=network_res_log2,
                          backbone_name=network_backbone_name, num_conv1x1=network_num_conv1x1,
                          max_batch_size=batch_size, num_graph_module=network_num_graph_module,
                          graph_k=network_graph_k, graph_leaky_slope=network_graph_leaky_slope)
    else:
        raise ValueError("network type {} not supported in pretrain".format(network_type))
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
        for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, cam_Ks,
                        roi_mask_bits, pixel_x_codes, pixel_y_codes, roi_xy_oris) in enumerate(train_loader):
            step_start_time = time.time()
            # if multiple training sets, get data from the second set
            if training_data_folder_2 != 'none':
                try:
                    data_2, entire_masks_2, masks_2, Rs_2, ts_2, Bboxes_2, cam_Ks_2, \
                    roi_mask_bits_2, pixel_x_codes_2, pixel_y_codes_2, roi_xy_oris_2 = next(train_loader_2_iter)
                except StopIteration:
                    train_loader_2_iter = iter(train_loader_2)
                    data_2, entire_masks_2, masks_2, Rs_2, ts_2, Bboxes_2, cam_Ks_2, \
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
            # data to GPU
            if torch.cuda.is_available():
                data = data.cuda()
                entire_masks = entire_masks.cuda()
                masks = masks.cuda()
                roi_mask_bits = roi_mask_bits.cuda()
                pixel_x_codes = pixel_x_codes.cuda()
                pixel_y_codes = pixel_y_codes.cuda()

            optimizer.zero_grad()
            if data.shape[0] != batch_size:
                raise ValueError(f"batch size wrong")
            pred = net(data)
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
                    test_init(net, test_loader, writer, iteration_step, configs, best_score_metric="mean_bit_acc")
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
    parser.add_argument('--obj_name', type=str)  # config file
    parser.add_argument('--load_checkpoint', action='store_true', help="load checkpoint of the current stage")
    args = parser.parse_args()
    config_file = args.cfg
    configs = parse_cfg(config_file)
    configs['obj_name'] = args.obj_name
    configs['load_checkpoint'] = args.load_checkpoint

    check_point_path = configs['check_point_path']
    tensorboard_path = configs['tensorboard_path']
    config_file_name = os.path.basename(config_file)
    config_file_name = os.path.splitext(config_file_name)[0]
    check_point_path = check_point_path + config_file_name + '/' + args.obj_name
    tensorboard_path = tensorboard_path + config_file_name + '/' + args.obj_name
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
