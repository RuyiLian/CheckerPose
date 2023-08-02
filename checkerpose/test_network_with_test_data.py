import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from common_ops import from_output_to_class_mask, from_output_to_class_binary_code
from tools_for_BOP.common_dataset_info import get_obj_info
from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP
from tqdm import tqdm
import sys
import cv2
from binary_code_helper.class_id_encoder_decoder import class_code_vecs_to_class_id_vec
sys.path.append("../bop_toolkit")
from bop_toolkit_lib import pose_error

# compute the pixel-wise mean error of the predicted masks
# supposed that the mask shape is (h, w), value is binary
def compute_mask_pixelwise_error(pred, gt):
    err = np.mean(np.abs(pred - gt))
    return err

# compute the IoU of the predicted masks
# supposed that the mask shape is (h, w), value is binary
def compute_mask_iou(pred, gt):
    intersection = np.sum(np.logical_and(pred, gt))
    union = np.sum(np.logical_or(pred, gt))
    if union < 1:  # both pred and gt is empty, which is possible for wrong detection box results
        return 1.0
    iou = intersection / union
    return iou

def from_id_to_pose(p3d_xyz, roi_xy_ori, cam_K, roi_mask_bit, pixel_x_id, pixel_y_id, check_seg=False,
                    seg_mask=None, use_progressivex=False, neighborhood_ball_radius=20, spatial_coherence_weight=0.1,
                    prog_max_iters=400, discard_bd_pixel=0, return_inliers=False,
                    reprojErr_thresh=2, cv_max_iters=150):
    ''' from pixel id to pose (all inputs are numpy arrays)
    Args:
        p3d_xyz: original 3D coordinates (no normalization)
        roi_xy_ori: original RoI 2D coordinate grid
        cam_K: camera intrinsic matrix, shape: (3, 3)
        roi_mask_bit: binary bit for whether the keypoint is in RoI or not, shape: (#keypoint, 1)
        pixel_x_id: pixel index on x direction, shape: (#keypoint,)
        pixel_y_id: pixel index on y direction, shape: (#keypoint,)
        check_seg: only keep keypoints that are in the segmentation masks
        seg_mask: image segmentation mask, shape: (h, w)
        use_progressivex: whether use progressive-x solver or not
            params: neighborhood_ball_radius, spatial_coherence_weight, prog_max_iters
        discard_bd_pixel: discard the result if the distance (on x or y) to boundary is less than the given number of pixels
    '''
    # convert pixel index to 2D coordinates
    num_all_pt = p3d_xyz.shape[0]
    pt_idx = np.arange(num_all_pt)
    roi_h, roi_w, _ = roi_xy_ori.shape
    disc_p2d = roi_xy_ori[pixel_y_id, pixel_x_id]
    # filter out invalid correspondences
    valid_mask = (roi_mask_bit[:, 0] > 0.5)
    if check_seg:
        valid_mask = np.logical_and(valid_mask, seg_mask[pixel_y_id, pixel_x_id] > 0.5)
    if discard_bd_pixel > 0:
        bd_mask = np.zeros((roi_h, roi_w))
        bd_mask[discard_bd_pixel:(roi_h-discard_bd_pixel), discard_bd_pixel:(roi_w-discard_bd_pixel)] = 1.0
        valid_mask = np.logical_and(valid_mask, bd_mask[pixel_y_id, pixel_x_id] > 0.5)
    valid_p3d = p3d_xyz[valid_mask]
    valid_disc_p2d = disc_p2d[valid_mask]
    valid_pt_idx = pt_idx[valid_mask]
    num_valid = valid_p3d.shape[0]
    # compute the pose
    if use_progressivex:
        import pyprogressivex
        if num_valid >= 6:
            coord_3d = np.ascontiguousarray(valid_p3d)
            coord_2d = np.ascontiguousarray(valid_disc_p2d)
            intrinsic_matrix = np.ascontiguousarray(cam_K)
            try:
                pose_ests, label = pyprogressivex.find6DPoses(
                    x1y1=coord_2d.astype(np.float64),
                    x2y2z2=coord_3d.astype(np.float64),
                    K=intrinsic_matrix.astype(np.float64),
                    threshold=reprojErr_thresh,
                    neighborhood_ball_radius=neighborhood_ball_radius,
                    spatial_coherence_weight=spatial_coherence_weight,
                    maximum_tanimoto_similarity=0.9,
                    max_iters=prog_max_iters,
                    minimum_point_number=6,
                    maximum_model_number=1
                )
                if pose_ests.shape[0] != 0:
                    R_predict = pose_ests[0:3, :3]
                    t_predict = pose_ests[0:3, 3].reshape((3,1))
                else:
                    R_predict = np.eye(3)
                    t_predict = np.zeros((3, 1))
            except:
                R_predict = np.eye(3)
                t_predict = np.zeros((3, 1))
        else:
            R_predict = np.eye(3)
            t_predict = np.zeros((3, 1))
        inliers = None  # todo: compute the inliers for progressive-X solver
    else:  # use opencv
        if num_valid >= 4:
            # note: opencv return inliers shape (N, 1)
            _, rvecs_predict, t_predict, inliers = cv2.solvePnPRansac(valid_p3d, valid_disc_p2d, cam_K, distCoeffs=None,
                                                                      reprojectionError=reprojErr_thresh,
                                                                      iterationsCount=cv_max_iters,
                                                                      flags=cv2.SOLVEPNP_EPNP)
            R_predict, _ = cv2.Rodrigues(rvecs_predict, jacobian=None)
            # need to convert the inlier ids to the original keypoints (including invalid ones)
            if inliers is not None:
                inliers = inliers.reshape(-1)
                inliers = valid_pt_idx[inliers]
        else:
            R_predict = np.eye(3)
            t_predict = np.zeros((3, 1))
            inliers = None
    if return_inliers:
        return R_predict, t_predict, inliers
    else:
        return R_predict, t_predict


def test_init(net, dataloader, writer, step, configs, best_score_metric=None):
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    activation_function = nn.Sigmoid()  # for inference: convert prediction to probability
    num_code_bit = net.num_out_bits
    network_res_log2 = (num_code_bit - 1) // 2
    num_p3d = int(2 ** configs['num_p3d_log2'])
    net.eval()
    mask_bit_acc_arr = np.zeros(len(dataloader.dataset))
    reproj_x_acc_arr = np.zeros(len(dataloader.dataset))
    reproj_y_acc_arr = np.zeros(len(dataloader.dataset))
    bit_err_arr = np.zeros((len(dataloader.dataset), num_code_bit))  # to see the error of each bit

    print("test dataset", flush=True)
    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, cam_Ks,
                    roi_mask_bits, pixel_x_codes, pixel_y_codes, roi_xy_oris) in enumerate(tqdm(dataloader)):
        # do the prediction and get the predicted binary code
        if torch.cuda.is_available():
            data = data.cuda()
            roi_mask_bits = roi_mask_bits.cuda()
            pixel_x_codes = pixel_x_codes.cuda()
            pixel_y_codes = pixel_y_codes.cuda()

        pred = net(data)
        pred = activation_function(pred)
        pred = torch.where(pred > 0.5, 1.0, 0.0)
        # split to roi_mask_bit, pixel_x_code, pixel_y_code, and apply GT masks
        pred_mask_bits = pred[:, 0:1, :]
        pred_x_codes = pred[:, 1:(network_res_log2+1), :] * roi_mask_bits
        pred_y_codes = pred[:, (network_res_log2+1):, :] * roi_mask_bits
        pixel_x_codes = pixel_x_codes[:, :network_res_log2] * roi_mask_bits
        pixel_y_codes = pixel_y_codes[:, :network_res_log2] * roi_mask_bits
        # convert to numpy array, shape: (batch, #keypoint, #bits)
        pred_mask_bits = pred_mask_bits.detach().cpu().numpy().transpose(0, 2, 1)
        pred_x_codes = pred_x_codes.detach().cpu().numpy().transpose(0, 2, 1)
        pred_y_codes = pred_y_codes.detach().cpu().numpy().transpose(0, 2, 1)
        roi_mask_bits = roi_mask_bits.detach().cpu().numpy().transpose(0, 2, 1)
        pixel_x_codes = pixel_x_codes.detach().cpu().numpy().transpose(0, 2, 1)
        pixel_y_codes = pixel_y_codes.detach().cpu().numpy().transpose(0, 2, 1)

        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            batchsize = dataloader.batch_size
            sample_idx = batch_idx * batchsize + counter
            npoint_in_roi = np.clip(roi_mask_bits[counter].sum(), a_min=1.0, a_max=None)

            # compute the reprojection error
            err_mask_bit = np.mean(np.abs(roi_mask_bits[counter] - pred_mask_bits[counter]))
            diff_x_codes = pixel_x_codes[counter] - pred_x_codes[counter]  # shape: (#keypoint, res_log2)
            diff_y_codes = pixel_y_codes[counter] - pred_y_codes[counter]
            reproj_err_x, reproj_err_y = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(network_res_log2):
                reproj_err_x += diff_x_codes[:, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
                reproj_err_y += diff_y_codes[:, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
            reproj_err_x = np.sum(np.abs(reproj_err_x)) / npoint_in_roi
            reproj_err_y = np.sum(np.abs(reproj_err_y)) / npoint_in_roi
            # convert to accuracy
            reproj_acc_x = 1.0 - reproj_err_x / (2 ** network_res_log2)
            reproj_acc_y = 1.0 - reproj_err_y / (2 ** network_res_log2)
            mask_bit_acc = 1.0 - err_mask_bit
            mask_bit_acc_arr[sample_idx] = mask_bit_acc
            reproj_x_acc_arr[sample_idx] = reproj_acc_x
            reproj_y_acc_arr[sample_idx] = reproj_acc_y

            # compute the bit-wise error
            err_x_codes = np.sum(np.abs(diff_x_codes), axis=0) / npoint_in_roi
            err_y_codes = np.sum(np.abs(diff_y_codes), axis=0) / npoint_in_roi
            bit_err_arr[sample_idx, 0] = err_mask_bit
            bit_err_arr[sample_idx, 1:(network_res_log2 + 1)] = err_x_codes
            bit_err_arr[sample_idx, (network_res_log2 + 1):] = err_y_codes

    mask_bit_acc_final = np.mean(mask_bit_acc_arr)
    reproj_acc_x_final = np.mean(reproj_x_acc_arr)
    reproj_acc_y_final = np.mean(reproj_y_acc_arr)
    bit_err_arr = np.mean(bit_err_arr, axis=0)
    if best_score_metric is None:
        test_acc = mask_bit_acc_final * 0.5 + reproj_acc_x_final * 0.25 + reproj_acc_y_final * 0.25
    elif best_score_metric == "mean_bit_acc":
        test_acc = 1.0 - np.mean(bit_err_arr)
    else:
        raise ValueError("best_score_metric {} not supported".format(best_score_metric))
    writer.add_scalar('TESTDATA_ACC/ACC_test', test_acc, step)

    # net back to train mode
    net.train()
    return test_acc, mask_bit_acc_final, reproj_acc_x_final, reproj_acc_y_final, bit_err_arr


def test_init_lm(net, dataloader, writer, step, configs, best_score_metric=None):
    activation_function = nn.Sigmoid()  # for inference: convert prediction to probability
    num_code_bit = net.num_out_bits
    network_res_log2 = (num_code_bit - 1) // 2
    num_p3d = int(2 ** configs['num_p3d_log2'])
    net.eval()
    mask_bit_acc_arr = np.zeros(len(dataloader.dataset))
    reproj_x_acc_arr = np.zeros(len(dataloader.dataset))
    reproj_y_acc_arr = np.zeros(len(dataloader.dataset))
    bit_err_arr = np.zeros((len(dataloader.dataset), num_code_bit))  # to see the error of each bit

    print("test dataset", flush=True)
    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, cam_Ks, obj_ids,
                    roi_mask_bits, pixel_x_codes, pixel_y_codes, roi_xy_oris) in enumerate(tqdm(dataloader)):
        # do the prediction and get the predicted binary code
        if torch.cuda.is_available():
            data = data.cuda()
            roi_mask_bits = roi_mask_bits.cuda()
            pixel_x_codes = pixel_x_codes.cuda()
            pixel_y_codes = pixel_y_codes.cuda()

        pred = net(data, obj_ids)
        pred = activation_function(pred)
        pred = torch.where(pred > 0.5, 1.0, 0.0)
        # split to roi_mask_bit, pixel_x_code, pixel_y_code, and apply GT masks
        pred_mask_bits = pred[:, 0:1, :]
        pred_x_codes = pred[:, 1:(network_res_log2+1), :] * roi_mask_bits
        pred_y_codes = pred[:, (network_res_log2+1):, :] * roi_mask_bits
        pixel_x_codes = pixel_x_codes[:, :network_res_log2] * roi_mask_bits
        pixel_y_codes = pixel_y_codes[:, :network_res_log2] * roi_mask_bits
        # convert to numpy array, shape: (batch, #keypoint, #bits)
        pred_mask_bits = pred_mask_bits.detach().cpu().numpy().transpose(0, 2, 1)
        pred_x_codes = pred_x_codes.detach().cpu().numpy().transpose(0, 2, 1)
        pred_y_codes = pred_y_codes.detach().cpu().numpy().transpose(0, 2, 1)
        roi_mask_bits = roi_mask_bits.detach().cpu().numpy().transpose(0, 2, 1)
        pixel_x_codes = pixel_x_codes.detach().cpu().numpy().transpose(0, 2, 1)
        pixel_y_codes = pixel_y_codes.detach().cpu().numpy().transpose(0, 2, 1)

        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            batchsize = dataloader.batch_size
            sample_idx = batch_idx * batchsize + counter
            npoint_in_roi = np.clip(roi_mask_bits[counter].sum(), a_min=1.0, a_max=None)

            # compute the reprojection error
            err_mask_bit = np.mean(np.abs(roi_mask_bits[counter] - pred_mask_bits[counter]))
            diff_x_codes = pixel_x_codes[counter] - pred_x_codes[counter]  # shape: (#keypoint, res_log2)
            diff_y_codes = pixel_y_codes[counter] - pred_y_codes[counter]
            reproj_err_x, reproj_err_y = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(network_res_log2):
                reproj_err_x += diff_x_codes[:, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
                reproj_err_y += diff_y_codes[:, bit_i] * (2 ** (network_res_log2 - 1 - bit_i))
            reproj_err_x = np.sum(np.abs(reproj_err_x)) / npoint_in_roi
            reproj_err_y = np.sum(np.abs(reproj_err_y)) / npoint_in_roi
            # convert to accuracy
            reproj_acc_x = 1.0 - reproj_err_x / (2 ** network_res_log2)
            reproj_acc_y = 1.0 - reproj_err_y / (2 ** network_res_log2)
            mask_bit_acc = 1.0 - err_mask_bit
            mask_bit_acc_arr[sample_idx] = mask_bit_acc
            reproj_x_acc_arr[sample_idx] = reproj_acc_x
            reproj_y_acc_arr[sample_idx] = reproj_acc_y

            # compute the bit-wise error
            err_x_codes = np.sum(np.abs(diff_x_codes), axis=0) / npoint_in_roi
            err_y_codes = np.sum(np.abs(diff_y_codes), axis=0) / npoint_in_roi
            bit_err_arr[sample_idx, 0] = err_mask_bit
            bit_err_arr[sample_idx, 1:(network_res_log2 + 1)] = err_x_codes
            bit_err_arr[sample_idx, (network_res_log2 + 1):] = err_y_codes

    mask_bit_acc_final = np.mean(mask_bit_acc_arr)
    reproj_acc_x_final = np.mean(reproj_x_acc_arr)
    reproj_acc_y_final = np.mean(reproj_y_acc_arr)
    if best_score_metric is None:
        test_acc = mask_bit_acc_final * 0.5 + reproj_acc_x_final * 0.25 + reproj_acc_y_final * 0.25
    elif best_score_metric == "mean_bit_acc":
        test_acc = 1.0 - np.mean(bit_err_arr)
    else:
        raise ValueError("best_score_metric {} not supported".format(best_score_metric))
    bit_err_arr = np.mean(bit_err_arr, axis=0)
    writer.add_scalar('TESTDATA_ACC/ACC_test', test_acc, step)

    # net back to train mode
    net.train()
    return test_acc, mask_bit_acc_final, reproj_acc_x_final, reproj_acc_y_final, bit_err_arr


# p3d_normed: normalized 3D keypoints, torch tensor, shape: (1, 3, #keypoints)
# p3d_xyz: unnormalized 3D keypoints, numpy array, shape: (#keypoints, 3)
# vertices: object mesh vertices, used for computing ADX metric (unit is mm)
# obj_diameter: object diameter for computing ADX metric
# train_stage: current training stage for the progressive pipeline (i.e. number of active refinement blocks)
#  if None, use all refinement blocks
# seg_cur_stage: segment at the resolution of the current stage
def test_pipeline(net, dataloader, writer, step, configs, p3d_normed, p3d_xyz, vertices, obj_diameter,
                  train_stage=None, seg_cur_stage=False):
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    network_res_log2 = configs.get("network_res_log2", 4)
    activation_function = nn.Sigmoid()  # for inference: convert prediction to probability
    num_p3d = int(2 ** configs['num_p3d_log2'])
    if seg_cur_stage:
        seg_size = (2 ** (train_stage + 3), 2 ** (train_stage + 3))
    else:
        seg_size = (2 ** network_res_log2, 2 ** network_res_log2)  # size of segmentation masks
    _, symmetry_obj = get_obj_info(dataset_name)
    if obj_name in symmetry_obj:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
    net.eval()
    adx2_passed_arr = np.zeros(len(dataloader.dataset))
    adx5_passed_arr = np.zeros(len(dataloader.dataset))
    adx10_passed_arr = np.zeros(len(dataloader.dataset))
    adx_err_arr = np.zeros(len(dataloader.dataset))
    rot_err_arr = np.zeros(len(dataloader.dataset))
    trans_err_arr = np.zeros(len(dataloader.dataset))
    full_adx2_passed_arr = np.zeros(len(dataloader.dataset))
    full_adx5_passed_arr = np.zeros(len(dataloader.dataset))
    full_adx10_passed_arr = np.zeros(len(dataloader.dataset))
    full_adx_err_arr = np.zeros(len(dataloader.dataset))
    full_rot_err_arr = np.zeros(len(dataloader.dataset))
    full_trans_err_arr = np.zeros(len(dataloader.dataset))
    visib_adx2_passed_arr = np.zeros(len(dataloader.dataset))
    visib_adx5_passed_arr = np.zeros(len(dataloader.dataset))
    visib_adx10_passed_arr = np.zeros(len(dataloader.dataset))
    visib_adx_err_arr = np.zeros(len(dataloader.dataset))
    visib_rot_err_arr = np.zeros(len(dataloader.dataset))
    visib_trans_err_arr = np.zeros(len(dataloader.dataset))
    roi_bit_acc_arr = np.zeros(len(dataloader.dataset))
    reproj_x_acc_arr = np.zeros(len(dataloader.dataset))
    reproj_y_acc_arr = np.zeros(len(dataloader.dataset))
    bit_err_arr = np.zeros((len(dataloader.dataset), 2 * network_res_log2 + 1))
    visib_pixel_acc_arr = np.zeros(len(dataloader.dataset))
    visib_iou_arr = np.zeros(len(dataloader.dataset))
    full_pixel_acc_arr = np.zeros(len(dataloader.dataset))
    full_iou_arr = np.zeros(len(dataloader.dataset))

    print("test dataset", flush=True)
    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, cam_Ks,
                    roi_mask_bits, pixel_x_codes, pixel_y_codes, roi_xy_oris) in enumerate(tqdm(dataloader)):
        # do the prediction and get the predicted binary code
        if torch.cuda.is_available():
            data = data.cuda()
            roi_mask_bits = roi_mask_bits.cuda()
            pixel_x_codes = pixel_x_codes.cuda()
            pixel_y_codes = pixel_y_codes.cuda()

        cur_batch_size = data.shape[0]
        batch_p3d_normed = p3d_normed.expand(cur_batch_size, -1, -1)  # shape: (batch, 3, #keypoint)
        pred_roi_bit, pred_x_bits, pred_y_bits, pred_seg, pred_x_id, pred_y_id = net(data, batch_p3d_normed, train_stage)

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
        if train_stage is not None and num_proj_bits < network_res_log2:
            roi_xy_size = (2 ** num_proj_bits, 2 ** num_proj_bits)
            pred_pose_seg = F.interpolate(pred_seg, size=roi_xy_size, mode="nearest")
            pred_pose_seg_visib = pred_pose_seg[:, 0, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
            pred_pose_seg_full = pred_pose_seg[:, 1, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        else:
            pred_pose_seg_visib = pred_seg[:, 0, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
            pred_pose_seg_full = pred_seg[:, 1, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        pred_seg_visib = pred_seg[:, 0, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        pred_seg_full = pred_seg[:, 1, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        gt_seg_visib = F.interpolate(masks[:, None], size=seg_size, mode="nearest")
        gt_seg_visib = gt_seg_visib.detach().cpu().numpy()  # shape: (batch, h, w)
        gt_seg_full = F.interpolate(entire_masks[:, None], size=seg_size, mode="nearest")
        gt_seg_full = gt_seg_full.detach().cpu().numpy()  # shape: (batch, h, w)

        # for pose estimation
        # if the refinement does not reach the final resolution, we need to downsample the roi 2D coordinates
        if train_stage is not None and num_proj_bits < network_res_log2:
            roi_xy_size = (2 ** num_proj_bits, 2 ** num_proj_bits)
            roi_xy_oris = F.interpolate(roi_xy_oris, size=roi_xy_size, mode='bilinear', align_corners=False)
        roi_xy_oris = roi_xy_oris.detach().cpu().numpy().transpose(0, 2, 3, 1)  # shape: (B, H, W, 2)
        cam_Ks = cam_Ks.detach().cpu().numpy()
        Rs = Rs.detach().cpu().numpy()
        ts = ts.detach().cpu().numpy()

        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            batchsize = dataloader.batch_size
            sample_idx = batch_idx * batchsize + counter
            # compute pose
            roi_xy_ori = roi_xy_oris[counter]
            # pose using all 3D-2D correspondences that are in the RoI
            pred_rot, pred_trans = from_id_to_pose(p3d_xyz=p3d_xyz, roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                   roi_mask_bit=pred_roi_bit[counter], pixel_x_id=pred_x_id[counter],
                                                   pixel_y_id=pred_y_id[counter], check_seg=False)
            # discard correspondences that are out of predicted full segmentation masks
            pred_full_rot, pred_full_trans = from_id_to_pose(p3d_xyz=p3d_xyz, roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                             roi_mask_bit=pred_roi_bit[counter], pixel_x_id=pred_x_id[counter],
                                                             pixel_y_id=pred_y_id[counter], check_seg=True,
                                                             seg_mask=pred_pose_seg_full[counter])
            # discard correspondences that are out of predicted visible segmentation masks
            pred_visib_rot, pred_visib_trans = from_id_to_pose(p3d_xyz=p3d_xyz, roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                               roi_mask_bit=pred_roi_bit[counter], pixel_x_id=pred_x_id[counter],
                                                               pixel_y_id=pred_y_id[counter], check_seg=True,
                                                               seg_mask=pred_pose_seg_visib[counter])
            # compute pose error
            adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, pred_rot, pred_trans, vertices)
            if np.isnan(adx_error):
                adx_error = 10000
            adx_err_arr[sample_idx] = adx_error
            if adx_error < obj_diameter * 0.02:
                adx2_passed_arr[sample_idx] = 1
            if adx_error < obj_diameter * 0.05:
                adx5_passed_arr[sample_idx] = 1
            if adx_error < obj_diameter * 0.1:
                adx10_passed_arr[sample_idx] = 1
            rot_err_arr[sample_idx] = pose_error.re(r_GT, pred_rot)
            trans_err_arr[sample_idx] = pose_error.te(t_GT, pred_trans)
            full_adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, pred_full_rot, pred_full_trans, vertices)
            if np.isnan(full_adx_error):
                full_adx_error = 10000
            full_adx_err_arr[sample_idx] = full_adx_error
            if full_adx_error < obj_diameter * 0.02:
                full_adx2_passed_arr[sample_idx] = 1
            if full_adx_error < obj_diameter * 0.05:
                full_adx5_passed_arr[sample_idx] = 1
            if full_adx_error < obj_diameter * 0.1:
                full_adx10_passed_arr[sample_idx] = 1
            full_rot_err_arr[sample_idx] = pose_error.re(r_GT, pred_full_rot)
            full_trans_err_arr[sample_idx] = pose_error.te(t_GT, pred_full_trans)
            visib_adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, pred_visib_rot, pred_visib_trans, vertices)
            if np.isnan(visib_adx_error):
                visib_adx_error = 10000
            visib_adx_err_arr[sample_idx] = visib_adx_error
            if visib_adx_error < obj_diameter * 0.02:
                visib_adx2_passed_arr[sample_idx] = 1
            if visib_adx_error < obj_diameter * 0.05:
                visib_adx5_passed_arr[sample_idx] = 1
            if visib_adx_error < obj_diameter * 0.1:
                visib_adx10_passed_arr[sample_idx] = 1
            visib_rot_err_arr[sample_idx] = pose_error.re(r_GT, pred_visib_rot)
            visib_trans_err_arr[sample_idx] = pose_error.te(t_GT, pred_visib_trans)

            # compute the reprojection accuracy
            npoint_in_roi = np.clip(gt_roi_bit[counter].sum(), a_min=1.0, a_max=None)
            err_roi_bit = np.mean(np.abs(gt_roi_bit[counter] - pred_roi_bit[counter]))
            roi_bit_acc_arr[sample_idx] = 1.0 - err_roi_bit
            diff_x_bits = (gt_x_bits[counter] - pred_x_bits[counter]) * gt_roi_bit[counter]
            diff_y_bits = (gt_y_bits[counter] - pred_y_bits[counter]) * gt_roi_bit[counter]
            reproj_err_x, reproj_err_y = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(num_proj_bits):
                reproj_err_x += diff_x_bits[:, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
                reproj_err_y += diff_y_bits[:, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
            reproj_err_x = np.sum(np.abs(reproj_err_x)) / npoint_in_roi
            reproj_err_y = np.sum(np.abs(reproj_err_y)) / npoint_in_roi
            reproj_x_acc_arr[sample_idx] = 1.0 - reproj_err_x / (2 ** num_proj_bits)
            reproj_y_acc_arr[sample_idx] = 1.0 - reproj_err_y / (2 ** num_proj_bits)
            # compute the bit-wise error
            err_x_bits = np.sum(np.abs(diff_x_bits), axis=0) / npoint_in_roi
            err_y_bits = np.sum(np.abs(diff_y_bits), axis=0) / npoint_in_roi
            bit_err_arr[sample_idx, 0] = err_roi_bit
            bit_err_arr[sample_idx, 1:(num_proj_bits + 1)] = err_x_bits
            bit_err_arr[sample_idx, (num_proj_bits + 1):(2 * num_proj_bits + 1)] = err_y_bits

            # compute the segmentation accuracy
            visib_pixel_acc_arr[sample_idx] = 1.0 - compute_mask_pixelwise_error(pred_seg_visib[counter], gt_seg_visib[counter])
            visib_iou_arr[sample_idx] = compute_mask_iou(pred_seg_visib[counter], gt_seg_visib[counter])
            full_pixel_acc_arr[sample_idx] = 1.0 - compute_mask_pixelwise_error(pred_seg_full[counter], gt_seg_full[counter])
            full_iou_arr[sample_idx] = compute_mask_iou(pred_seg_full[counter], gt_seg_full[counter])

    # todo: compute average pose error and set it as the test_acc
    adx2_passed = np.mean(adx2_passed_arr)
    adx5_passed = np.mean(adx5_passed_arr)
    adx10_passed = np.mean(adx10_passed_arr)
    adx_err = np.mean(adx_err_arr)
    rot_err = np.mean(rot_err_arr)
    trans_err = np.mean(trans_err_arr)
    full_adx2_passed = np.mean(full_adx2_passed_arr)
    full_adx5_passed = np.mean(full_adx5_passed_arr)
    full_adx10_passed = np.mean(full_adx10_passed_arr)
    full_adx_err = np.mean(full_adx_err_arr)
    full_rot_err = np.mean(full_rot_err_arr)
    full_trans_err = np.mean(full_trans_err_arr)
    visib_adx2_passed = np.mean(visib_adx2_passed_arr)
    visib_adx5_passed = np.mean(visib_adx5_passed_arr)
    visib_adx10_passed = np.mean(visib_adx10_passed_arr)
    visib_adx_err = np.mean(visib_adx_err_arr)
    visib_rot_err = np.mean(visib_rot_err_arr)
    visib_trans_err = np.mean(visib_trans_err_arr)
    test_acc = adx10_passed
    roi_bit_acc = np.mean(roi_bit_acc_arr)
    reproj_x_acc = np.mean(reproj_x_acc_arr)
    reproj_y_acc = np.mean(reproj_y_acc_arr)
    bit_err_arr = np.mean(bit_err_arr, axis=0)
    visib_pixel_acc = np.mean(visib_pixel_acc_arr)
    visib_iou = np.mean(visib_iou_arr)
    full_pixel_acc = np.mean(full_pixel_acc_arr)
    full_iou = np.mean(full_iou_arr)
    writer.add_scalar('TESTDATA_ACC/ACC_test', test_acc, step)
    # net back to train mode
    net.train()
    return test_acc, adx2_passed, adx5_passed, adx10_passed, adx_err, rot_err, trans_err, \
           full_adx2_passed, full_adx5_passed, full_adx10_passed, full_adx_err, full_rot_err, full_trans_err, \
           visib_adx2_passed, visib_adx5_passed, visib_adx10_passed, visib_adx_err, visib_rot_err, visib_trans_err, \
           roi_bit_acc, reproj_x_acc, reproj_y_acc, bit_err_arr, \
           visib_pixel_acc, visib_iou, full_pixel_acc, full_iou


def test_pipeline_lm(net, dataloader, writer, step, configs, p3d_normed, p3d_xyz, vertices_dict,
                     obj_diameter_dict, symmetry_ids, train_stage=None):
    network_res_log2 = configs.get("network_res_log2", 4)
    lm13_obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]  # no bowl, cup
    activation_function = nn.Sigmoid()  # for inference: convert prediction to probability
    num_p3d = int(2 ** configs['num_p3d_log2'])
    seg_size = (2 ** network_res_log2, 2 ** network_res_log2)  # size of segmentation masks
    net.eval()
    # main eval metrics should be averaged based on objects
    adx2_passed_dict = {idx: [] for idx in lm13_obj_ids}
    adx5_passed_dict = {idx: [] for idx in lm13_obj_ids}
    adx10_passed_dict = {idx: [] for idx in lm13_obj_ids}
    roi_bit_acc_arr = np.zeros(len(dataloader.dataset))
    reproj_x_acc_arr = np.zeros(len(dataloader.dataset))
    reproj_y_acc_arr = np.zeros(len(dataloader.dataset))
    bit_err_arr = np.zeros((len(dataloader.dataset), 2 * network_res_log2 + 1))
    visib_pixel_acc_arr = np.zeros(len(dataloader.dataset))
    visib_iou_arr = np.zeros(len(dataloader.dataset))
    full_pixel_acc_arr = np.zeros(len(dataloader.dataset))
    full_iou_arr = np.zeros(len(dataloader.dataset))

    print("test dataset", flush=True)
    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, cam_Ks, obj_ids,
                    roi_mask_bits, pixel_x_codes, pixel_y_codes, roi_xy_oris) in enumerate(tqdm(dataloader)):
        # do the prediction and get the predicted binary code
        if torch.cuda.is_available():
            data = data.cuda()
            roi_mask_bits = roi_mask_bits.cuda()
            pixel_x_codes = pixel_x_codes.cuda()
            pixel_y_codes = pixel_y_codes.cuda()

        cur_batch_size = data.shape[0]
        batch_p3d_normed = p3d_normed[obj_ids-1]  # shape: (batch, 3, #keypoint), note obj_ids start from 1
        pred_roi_bit, pred_x_bits, pred_y_bits, pred_seg, pred_x_id, pred_y_id = \
            net(data, batch_p3d_normed, obj_ids, train_stage)

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
        if train_stage is not None and num_proj_bits < network_res_log2:
            roi_xy_size = (2 ** num_proj_bits, 2 ** num_proj_bits)
            pred_pose_seg = F.interpolate(pred_seg, size=roi_xy_size, mode="nearest")
            pred_pose_seg_visib = pred_pose_seg[:, 0, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
            pred_pose_seg_full = pred_pose_seg[:, 1, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        else:
            pred_pose_seg_visib = pred_seg[:, 0, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
            pred_pose_seg_full = pred_seg[:, 1, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        pred_seg_visib = pred_seg[:, 0, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        pred_seg_full = pred_seg[:, 1, :, :].detach().cpu().numpy()  # shape: (batch, h, w)
        gt_seg_visib = F.interpolate(masks[:, None], size=seg_size, mode="nearest")
        gt_seg_visib = gt_seg_visib.detach().cpu().numpy()  # shape: (batch, h, w)
        gt_seg_full = F.interpolate(entire_masks[:, None], size=seg_size, mode="nearest")
        gt_seg_full = gt_seg_full.detach().cpu().numpy()  # shape: (batch, h, w)

        # for pose estimation
        # if the refinement does not reach the final resolution, we need to downsample the roi 2D coordinates
        if train_stage is not None and num_proj_bits < network_res_log2:
            roi_xy_size = (2 ** num_proj_bits, 2 ** num_proj_bits)
            roi_xy_oris = F.interpolate(roi_xy_oris, size=roi_xy_size, mode='bilinear', align_corners=False)
        roi_xy_oris = roi_xy_oris.detach().cpu().numpy().transpose(0, 2, 3, 1)  # shape: (B, H, W, 2)
        cam_Ks = cam_Ks.detach().cpu().numpy()
        Rs = Rs.detach().cpu().numpy()
        ts = ts.detach().cpu().numpy()
        obj_ids = obj_ids.detach().cpu().numpy()

        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            batchsize = dataloader.batch_size
            sample_idx = batch_idx * batchsize + counter
            # compute pose
            obj_id = obj_ids[counter]
            roi_xy_ori = roi_xy_oris[counter]
            # pose using all 3D-2D correspondences that are in the RoI
            pred_rot, pred_trans = from_id_to_pose(p3d_xyz=p3d_xyz[obj_id-1], roi_xy_ori=roi_xy_ori, cam_K=cam_K,
                                                   roi_mask_bit=pred_roi_bit[counter], pixel_x_id=pred_x_id[counter],
                                                   pixel_y_id=pred_y_id[counter], check_seg=False)
            # compute pose error
            if obj_id in symmetry_ids:
                Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
            else:
                Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
            adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, pred_rot, pred_trans, vertices_dict[obj_id])
            if np.isnan(adx_error):
                adx_error = 10000

            adx2_passed_dict[obj_id].append(float(adx_error < obj_diameter_dict[obj_id] * 0.02))
            adx5_passed_dict[obj_id].append(float(adx_error < obj_diameter_dict[obj_id] * 0.05))
            adx10_passed_dict[obj_id].append(float(adx_error < obj_diameter_dict[obj_id] * 0.1))

            # compute the reprojection accuracy
            npoint_in_roi = np.clip(gt_roi_bit[counter].sum(), a_min=1.0, a_max=None)
            err_roi_bit = np.mean(np.abs(gt_roi_bit[counter] - pred_roi_bit[counter]))
            roi_bit_acc_arr[sample_idx] = 1.0 - err_roi_bit
            diff_x_bits = (gt_x_bits[counter] - pred_x_bits[counter]) * gt_roi_bit[counter]
            diff_y_bits = (gt_y_bits[counter] - pred_y_bits[counter]) * gt_roi_bit[counter]
            reproj_err_x, reproj_err_y = np.zeros(num_p3d), np.zeros(num_p3d)
            for bit_i in range(num_proj_bits):
                reproj_err_x += diff_x_bits[:, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
                reproj_err_y += diff_y_bits[:, bit_i] * (2 ** (num_proj_bits - 1 - bit_i))
            reproj_err_x = np.sum(np.abs(reproj_err_x)) / npoint_in_roi
            reproj_err_y = np.sum(np.abs(reproj_err_y)) / npoint_in_roi
            reproj_x_acc_arr[sample_idx] = 1.0 - reproj_err_x / (2 ** num_proj_bits)
            reproj_y_acc_arr[sample_idx] = 1.0 - reproj_err_y / (2 ** num_proj_bits)
            # compute the bit-wise error
            err_x_bits = np.sum(np.abs(diff_x_bits), axis=0) / npoint_in_roi
            err_y_bits = np.sum(np.abs(diff_y_bits), axis=0) / npoint_in_roi
            bit_err_arr[sample_idx, 0] = err_roi_bit
            bit_err_arr[sample_idx, 1:(num_proj_bits + 1)] = err_x_bits
            bit_err_arr[sample_idx, (num_proj_bits + 1):(2 * num_proj_bits + 1)] = err_y_bits

            # compute the segmentation accuracy
            visib_pixel_acc_arr[sample_idx] = 1.0 - compute_mask_pixelwise_error(pred_seg_visib[counter], gt_seg_visib[counter])
            visib_iou_arr[sample_idx] = compute_mask_iou(pred_seg_visib[counter], gt_seg_visib[counter])
            full_pixel_acc_arr[sample_idx] = 1.0 - compute_mask_pixelwise_error(pred_seg_full[counter], gt_seg_full[counter])
            full_iou_arr[sample_idx] = compute_mask_iou(pred_seg_full[counter], gt_seg_full[counter])

    # summarize the results
    adx2_passed_arr = np.zeros(13)
    adx5_passed_arr = np.zeros(13)
    adx10_passed_arr = np.zeros(13)
    for i, obj_id in enumerate(lm13_obj_ids):
        adx2_passed_arr[i] = np.mean(adx2_passed_dict[obj_id])
        adx5_passed_arr[i] = np.mean(adx5_passed_dict[obj_id])
        adx10_passed_arr[i] = np.mean(adx10_passed_dict[obj_id])
    adx2_passed = np.mean(adx2_passed_arr)
    adx5_passed = np.mean(adx5_passed_arr)
    adx10_passed = np.mean(adx10_passed_arr)
    test_acc = adx10_passed
    # for aux metrics simply average all data samples
    roi_bit_acc = np.mean(roi_bit_acc_arr)
    reproj_x_acc = np.mean(reproj_x_acc_arr)
    reproj_y_acc = np.mean(reproj_y_acc_arr)
    bit_err_arr = np.mean(bit_err_arr, axis=0)
    visib_pixel_acc = np.mean(visib_pixel_acc_arr)
    visib_iou = np.mean(visib_iou_arr)
    full_pixel_acc = np.mean(full_pixel_acc_arr)
    full_iou = np.mean(full_iou_arr)
    writer.add_scalar('TESTDATA_ACC/ACC_test', test_acc, step)
    # net back to train mode
    net.train()
    return test_acc, adx2_passed, adx5_passed, adx10_passed, \
           roi_bit_acc, reproj_x_acc, reproj_y_acc, bit_err_arr, \
           visib_pixel_acc, visib_iou, full_pixel_acc, full_iou
