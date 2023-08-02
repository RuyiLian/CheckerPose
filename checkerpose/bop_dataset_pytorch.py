import os

import torch
import numpy as np
from PIL import Image
import cv2
import mmcv
import imageio
import math
from torch.utils.data import Dataset

import sys
from binary_code_helper.class_id_encoder_decoder import class_id_vec_to_class_code_vecs
import torchvision.transforms as transforms
import scipy.ndimage as scin

import GDR_Net_Augmentation
from GDR_Net_Augmentation import get_affine_transform


def project_pts(pts, K, R, t):
    """Projects 3D points.

    :param pts: nx3 ndarray with the 3D points.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx2 ndarray with 2D image coordinates of the projections.
    """
    assert (pts.shape[1] == 3)
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    depth = pts_im[2, :].copy()
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T, depth


def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img


def crop_square_resize(img, Bbox, crop_size=None, interpolation=None):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh

    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    if bh > bw:
        x1 = bbox_center[0] - bh / 2
        x2 = bbox_center[0] + bh / 2
    else:
        y1 = bbox_center[1] - bw / 2
        y2 = bbox_center[1] + bw / 2

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if img.ndim > 2:
        roi_img = np.zeros((max(bh, bw), max(bh, bw), img.shape[2]), dtype=img.dtype)
    else:
        roi_img = np.zeros((max(bh, bw), max(bh, bw)), dtype=img.dtype)
    roi_x1 = max((0 - x1), 0)
    x1 = max(x1, 0)
    roi_x2 = roi_x1 + min((img.shape[1] - x1), (x2 - x1))
    roi_y1 = max((0 - y1), 0)
    y1 = max(y1, 0)
    roi_y2 = roi_y1 + min((img.shape[0] - y1), (y2 - y1))
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    roi_img[roi_y1:roi_y2, roi_x1:roi_x2] = img[y1:y2, x1:x2].copy()
    roi_img = cv2.resize(roi_img, (crop_size, crop_size), interpolation=interpolation)
    return roi_img


def crop_resize(img, Bbox, crop_size=None, interpolation=None):
    x1 = max(0, Bbox[0])
    x2 = min(img.shape[1], Bbox[0] + Bbox[2])
    y1 = max(0, Bbox[1])
    y2 = min(img.shape[0], Bbox[1] + Bbox[3])
    ####
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])
    ####

    img = img[y1:y2, x1:x2]
    roi_img = cv2.resize(img, (crop_size, crop_size), interpolation=interpolation)
    return roi_img


def get_scale_and_Bbox_center(Bbox, image):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh

    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    if bh > bw:
        x1 = bbox_center[0] - bh / 2
        x2 = bbox_center[0] + bh / 2
    else:
        y1 = bbox_center[1] - bw / 2
        y2 = bbox_center[1] + bw / 2

    scale = max(bh, bw)
    scale = min(scale, max(image.shape[0], image.shape[1])) * 1.0
    return scale, bbox_center


def get_roi(input, Bbox, crop_size, interpolation, resize_method):
    if resize_method == "crop_resize":
        roi = crop_resize(input, Bbox, crop_size, interpolation=interpolation)
        return roi
    elif resize_method == "crop_resize_by_warp_affine":
        scale, bbox_center = get_scale_and_Bbox_center(Bbox, input)
        roi = crop_resize_by_warp_affine(input, bbox_center, scale, crop_size, interpolation=interpolation)
        return roi
    elif resize_method == "crop_square_resize":
        roi = crop_square_resize(input, Bbox, crop_size, interpolation=interpolation)
        return roi
    else:
        raise NotImplementedError(f"unknown decoder type: {resize_method}")


def padding_Bbox(Bbox, padding_ratio):
    x1 = Bbox[0]
    x2 = Bbox[0] + Bbox[2]
    y1 = Bbox[1]
    y2 = Bbox[1] + Bbox[3]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1

    padded_bw = int(bw * padding_ratio)
    padded_bh = int(bh * padding_ratio)

    padded_Box = np.array([int(cx - padded_bw / 2), int(cy - padded_bh / 2), int(padded_bw), int(padded_bh)])
    return padded_Box


def aug_Bbox(GT_Bbox, padding_ratio):
    x1 = GT_Bbox[0].copy()
    x2 = GT_Bbox[0] + GT_Bbox[2]
    y1 = GT_Bbox[1].copy()
    y2 = GT_Bbox[1] + GT_Bbox[3]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1

    scale_ratio = 1 + 0.25 * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
    shift_ratio = 0.25 * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
    bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
    # 1.5 is the additional pad scale
    augmented_bw = int(bw * scale_ratio * padding_ratio)
    augmented_bh = int(bh * scale_ratio * padding_ratio)

    augmented_Box = np.array(
        [int(bbox_center[0] - augmented_bw / 2), int(bbox_center[1] - augmented_bh / 2), augmented_bw, augmented_bh])
    return augmented_Box


def get_final_Bbox(Bbox, resize_method, max_x, max_y):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh
    if resize_method == "crop_square_resize" or resize_method == "crop_resize_by_warp_affine":
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        if bh > bw:
            x1 = bbox_center[0] - bh / 2
            x2 = bbox_center[0] + bh / 2
        else:
            y1 = bbox_center[1] - bw / 2
            y2 = bbox_center[1] + bw / 2
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        Bbox = np.array([x1, y1, x2 - x1, y2 - y1])

    elif resize_method == "crop_resize":
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, max_x)
        y2 = min(y2, max_y)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        Bbox = np.array([x1, y1, x2 - x1, y2 - y1])

    return Bbox


def mapping_pixel_position_to_original_position_2d(pixels, Bbox, Bbox_Size):
    """
    The image was cropped and resized. This function returns the original pixel position
    input:
        pixels: pixel position after cropping and resize, which is a numpy array, (H, W, 2)
        Bbox: Bounding box for the cropping, minx miny width height
    """
    ratio_x = Bbox[2] / Bbox_Size
    ratio_y = Bbox[3] / Bbox_Size
    original_xy = np.zeros_like(pixels)
    original_xy[:, :, 0] = ratio_x * pixels[:, :, 0] + Bbox[0]
    original_xy[:, :, 1] = ratio_y * pixels[:, :, 1] + Bbox[1]
    return original_xy


# compute 2D projections of given 3D points and represent it as 2D codes
# 2D codes: 1 bit for within RoI or not, L bit for x direction, L bit for y direction
class bop_dataset_single_obj_pytorch_code2d(Dataset):
    def __init__(self, dataset_dir, data_folder, rgb_files, mask_files, mask_visib_files, gts, gt_infos, cam_params,
                 is_train, crop_size_img, crop_size_gt, unnorm_xyz, padding_ratio=1.5, resize_method="crop_resize",
                 use_peper_salt=False, use_motion_blur=False, Detect_Bbox=None):
        # unnorm_xyz: unnormalized xyz coordinates of the 3D points, unit is mm, shape (num_pt, 3)
        self.rgb_files = rgb_files
        self.mask_visib_files = mask_visib_files
        self.mask_files = mask_files
        self.gts = gts
        self.gt_infos = gt_infos
        self.cam_params = cam_params
        self.dataset_dir = dataset_dir
        self.data_folder = data_folder
        self.is_train = is_train
        self.crop_size_img = crop_size_img
        self.crop_size_gt = crop_size_gt
        self.unnorm_xyz = unnorm_xyz
        self.num_p3d = unnorm_xyz.shape[0]
        self.resize_method = resize_method
        self.Detect_Bbox = Detect_Bbox
        self.padding_ratio = padding_ratio
        self.use_peper_salt = use_peper_salt
        self.use_motion_blur = use_motion_blur
        self.nSamples = len(self.rgb_files)

        # precompute the relative coordinates of roi xy
        roi_x = np.linspace(0, crop_size_gt - 1, crop_size_gt)
        roi_y = np.linspace(0, crop_size_gt - 1, crop_size_gt)
        roi_xy = np.asarray(np.meshgrid(roi_x, roi_y))  # shape: (2, h, w)
        self.roi_xy = roi_xy.transpose((1, 2, 0))  # shape: (h, w, 2)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        # return training image, mask, bounding box, R, T, GT_image
        rgb_fn = self.rgb_files[index]
        mask_visib_fns = self.mask_visib_files[index]
        mask_fns = self.mask_files[index]

        x = cv2.imread(rgb_fn)
        mask = cv2.imread(mask_visib_fns[0], 0)
        entire_mask = cv2.imread(mask_fns[0], 0)

        gt = self.gts[index]
        gt_info = self.gt_infos[index]

        R = np.array(gt['cam_R_m2c']).reshape(3, 3)
        t = np.array(gt['cam_t_m2c'])
        Bbox = np.array(gt_info['bbox_visib'])
        cam_param = self.cam_params[index]['cam_K'].reshape((3, 3))

        # compute the 2D projections of the 3D points
        proj_xy, proj_depth = project_pts(self.unnorm_xyz, cam_param, R, t)
        num_code_dir_bit = int(math.log2(self.crop_size_gt))  # number of bits of the x/y code

        if self.is_train:
            try:
                x = self.apply_augmentation(x)
            except:
                print("fail to apply_augmentation, ", rgb_fn, flush=True)

            Bbox = aug_Bbox(Bbox, padding_ratio=self.padding_ratio)

            try:
                roi_x = get_roi(x, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR,
                                resize_method=self.resize_method)
            except:
                print("fail to get_roi of rgb image, ", rgb_fn, flush=True)
            try:
                roi_mask = get_roi(mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST,
                                   resize_method=self.resize_method)
            except:
                print("fail to get_roi of mask, ", mask_visib_fns[0], flush=True)
            try:
                roi_entire_mask = get_roi(entire_mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST,
                                          resize_method=self.resize_method)
            except:
                print("fail to get_roi of entire mask, ", mask_fns[0], flush=True)
            try:
                Bbox = get_final_Bbox(Bbox, self.resize_method, x.shape[1], x.shape[0])
            except:
                print("fail to get_final_box, ", x is None, rgb_fn, Bbox, flush=True)

        else:
            if self.Detect_Bbox != None:
                # replace the Bbox with detected Bbox
                Bbox = self.Detect_Bbox[index]
                if Bbox == None:  # no valid detection, give a dummy input
                    roi_x = torch.zeros((3, self.crop_size_img, self.crop_size_img))
                    roi_mask = torch.zeros((int(self.crop_size_gt), int(self.crop_size_gt)))
                    roi_entire_mask = torch.zeros((int(self.crop_size_gt), int(self.crop_size_gt)))
                    Bbox = np.array([0, 0, 0, 0], dtype='int')
                    roi_mask_bit = torch.zeros((1, self.num_p3d))
                    pixel_x_code = torch.zeros((num_code_dir_bit, self.num_p3d))
                    pixel_y_code = torch.zeros((num_code_dir_bit, self.num_p3d))
                    roi_xy_ori = torch.zeros((2, self.crop_size_gt, self.crop_size_gt))
                    return roi_x, roi_entire_mask, roi_mask, R, t, Bbox, cam_param, \
                           roi_mask_bit, pixel_x_code, pixel_y_code, roi_xy_ori

            # todo: some test fold doesn't provide GT, fill GT with dummy value
            # mask = np.zeros((x.shape[0], x.shape[1]))
            # entire_mask = np.zeros((x.shape[0], x.shape[1]))

            Bbox = padding_Bbox(Bbox, padding_ratio=self.padding_ratio)
            try:
                roi_x = get_roi(x, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR,
                                resize_method=self.resize_method)
            except:
                print("fail to get_roi of rgb image, ", rgb_fn, flush=True)
            roi_mask = get_roi(mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST,
                               resize_method=self.resize_method)
            roi_entire_mask = get_roi(entire_mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST,
                                      resize_method=self.resize_method)
            Bbox = get_final_Bbox(Bbox, self.resize_method, x.shape[1], x.shape[0])

        # discretize the 2D projections as 2D codes
        # todo: in default setting we use the crop_size_gt as code length on x/y direction
        roi_mask_bit = np.zeros((self.num_p3d, 1))
        roi_xy_ori = mapping_pixel_position_to_original_position_2d(self.roi_xy, Bbox, self.crop_size_gt)
        pixel_x_size = Bbox[2] / self.crop_size_gt
        pixel_y_size = Bbox[3] / self.crop_size_gt
        # first figure out the points outside the roi
        out_roi_mask1 = np.logical_or(proj_xy[:, 0] < Bbox[0], proj_xy[:, 1] < Bbox[1])
        pixel_x_id = ((proj_xy[:, 0] - Bbox[0]) / pixel_x_size).astype(int)
        pixel_y_id = ((proj_xy[:, 1] - Bbox[1]) / pixel_y_size).astype(int)
        out_roi_mask2 = np.logical_or(pixel_x_id >= self.crop_size_gt, pixel_y_id >= self.crop_size_gt)
        out_roi_mask = np.logical_or(out_roi_mask1, out_roi_mask2)
        roi_mask_bit[~out_roi_mask, 0] = 1.0
        # convert pixel x/y id to binary codes
        pixel_x_id = np.clip(pixel_x_id, 0, self.crop_size_gt - 1)
        pixel_y_id = np.clip(pixel_y_id, 0, self.crop_size_gt - 1)
        pixel_x_code = class_id_vec_to_class_code_vecs(pixel_x_id, class_base=2, iteration=num_code_dir_bit)  # shape: (#keypoint, #bits)
        pixel_y_code = class_id_vec_to_class_code_vecs(pixel_y_id, class_base=2, iteration=num_code_dir_bit)  # shape: (#keypoint, #bits)

        # add the augmentations and transfrom in torch tensor
        roi_x, roi_entire_mask, roi_mask = self.transform_pre(roi_x, roi_entire_mask, roi_mask)
        roi_mask_bit = torch.from_numpy(roi_mask_bit).type(torch.float).permute(1, 0)  # shape: (C, N)
        pixel_x_code = torch.from_numpy(pixel_x_code).type(torch.float).permute(1, 0)  # shape: (C, N)
        pixel_y_code = torch.from_numpy(pixel_y_code).type(torch.float).permute(1, 0)  # shape: (C, N)
        roi_xy_ori = torch.from_numpy(roi_xy_ori).type(torch.float).permute(2, 0, 1)  # shape: (2, h, w)
        # for single obj, only one gt
        return roi_x, roi_entire_mask, roi_mask, R, t, Bbox, cam_param, \
               roi_mask_bit, pixel_x_code, pixel_y_code, roi_xy_ori

    def transform_pre(self, sample_x, sample_entire_mask, sample_mask):
        composed_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        x_pil = Image.fromarray(np.uint8(sample_x)).convert('RGB')

        sample_entire_mask = sample_entire_mask / 255.
        sample_entire_mask = torch.from_numpy(sample_entire_mask).type(torch.float)
        sample_mask = sample_mask / 255.
        sample_mask = torch.from_numpy(sample_mask).type(torch.float)

        return composed_transforms_img(x_pil), sample_entire_mask, sample_mask

    def apply_augmentation(self, x):
        augmentations = GDR_Net_Augmentation.build_augmentations(self.use_peper_salt, self.use_motion_blur)
        color_aug_prob = 0.8
        if np.random.rand() < color_aug_prob:
            x = augmentations.augment_image(x)
        return x
