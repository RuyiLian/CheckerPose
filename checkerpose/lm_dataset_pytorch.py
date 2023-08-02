import os
import torch
import numpy as np
from PIL import Image
import cv2
import mmcv
import imageio
import math
from torch.utils.data import Dataset
import hashlib
import random
import sys
from binary_code_helper.class_id_encoder_decoder import class_id_vec_to_class_code_vecs
import torchvision.transforms as transforms
import scipy.ndimage as scin
import GDR_Net_Augmentation
from GDR_Net_Augmentation import get_affine_transform
from tools_for_LM.get_lm_datasets import get_lm_data_dicts
from get_detection_results import get_detection_results_LM
sys.path.append("../bop_toolkit")
from bop_toolkit_lib import misc


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

def resize_short_edge(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR, return_scale=False):
    """Scale the shorter edge to the given size, with a limit of `max_size` on
    the longer edge. If `max_size` is reached, then downscale so that the
    longer edge does not exceed max_size. only resize input image to target
    size and return scale.
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        if return_scale:
            return im, im_scale
        else:
            return im
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        if return_scale:
            return padded_im, im_scale
        else:
            return padded_im

def get_bg_image(filename, imH, imW, channel=3):
    """keep aspect ratio of bg during resize target image size:
    imHximWxchannel.
    """
    target_size = min(imH, imW)
    max_size = max(imH, imW)
    real_hw_ratio = float(imH) / float(imW)
    bg_image = cv2.imread(filename)
    bg_h, bg_w, bg_c = bg_image.shape
    bg_image_resize = np.zeros((imH, imW, channel), dtype="uint8")
    if (float(imH) / float(imW) < 1 and float(bg_h) / float(bg_w) < 1) or (
            float(imH) / float(imW) >= 1 and float(bg_h) / float(bg_w) >= 1
    ):
        if bg_h >= bg_w:
            bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
            if bg_h_new < bg_h:
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:
                bg_image_crop = bg_image
        else:
            bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
            if bg_w_new < bg_w:
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
            else:
                bg_image_crop = bg_image
    else:
        if bg_h >= bg_w:
            bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
            bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
        else:  # bg_h < bg_w
            bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
            # logger.info(bg_w_new)
            bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
    bg_image_resize_0 = resize_short_edge(bg_image_crop, target_size, max_size)
    h, w, c = bg_image_resize_0.shape
    bg_image_resize[0:h, 0:w, :] = bg_image_resize_0
    return bg_image_resize

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
class lm_dataset_single_obj_pytorch_code2d(Dataset):
    def __init__(self, dataset_root, dataset_name, is_train, crop_size_img, crop_size_gt,
                 unnorm_xyz, padding_ratio=1.5, resize_method="crop_resize", use_peper_salt=False, use_motion_blur=False,
                 Detect_Bbox_file=None, num_bg_imgs=10000, change_bg_prob=0.5):
        # unnorm_xyz: unnormalized xyz coordinates of the 3D points, unit is mm, shape (num_obj, num_pt, 3)
        self.data_dicts = get_lm_data_dicts(dataset_name, dataset_root)
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.is_train = is_train
        self.crop_size_img = crop_size_img
        self.crop_size_gt = crop_size_gt
        self.unnorm_xyz = unnorm_xyz
        self.num_p3d = unnorm_xyz.shape[1]
        self.resize_method = resize_method
        self.padding_ratio = padding_ratio
        self.use_peper_salt = use_peper_salt
        self.use_motion_blur = use_motion_blur
        self.num_bg_imgs = num_bg_imgs
        self.change_bg_prob = change_bg_prob
        self.bg_img_paths = self.get_bg_img_paths()
        self.nSamples = len(self.data_dicts)
        # precompute the relative coordinates of roi xy
        roi_x = np.linspace(0, crop_size_gt - 1, crop_size_gt)
        roi_y = np.linspace(0, crop_size_gt - 1, crop_size_gt)
        roi_xy = np.asarray(np.meshgrid(roi_x, roi_y))  # shape: (2, h, w)
        self.roi_xy = roi_xy.transpose((1, 2, 0))  # shape: (h, w, 2)
        # load detection results only for test dataset
        if self.is_train:
            self.Detect_Bbox = None
        else:
            if Detect_Bbox_file != 'none':
                self.Detect_Bbox = get_detection_results_LM(Detect_Bbox_file, self.data_dicts)
            else:
                self.Detect_Bbox = None

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        data = self.data_dicts[index]
        rgb_fn = data["file_name"]
        mask_visib_fn = data["annotations"][0]["mask_visib_file"]
        mask_fn = data["annotations"][0]["mask_file"]
        im_H = data["height"]
        im_W = data["width"]

        x = cv2.imread(rgb_fn)
        depth_fn = data["depth_file"]
        if mask_visib_fn is not None:  # for LM real images, the masks are stored in the disk
            mask = cv2.imread(mask_visib_fn, 0)
            entire_mask = cv2.imread(mask_fn, 0)
        else:  # for imgn images, the masks need to be computed from the depth images
            depth = mmcv.imread(depth_fn, "unchanged")
            mask = (depth > 0).astype(np.uint8) * 255
            entire_mask = mask.copy()

        if x is None:
            raise ValueError("x is None, ", rgb_fn)

        # replace the background image during training
        if self.is_train:
            # some synthetic data already has bg, img_type should be real or something else but not syn
            img_type = data.get("img_type", "real")
            if img_type == "syn":
                x = self.replace_bg(x.copy(), mask, return_mask=False)
            else:  # real image
                if np.random.rand() < self.change_bg_prob:
                    x = self.replace_bg(x.copy(), mask, return_mask=False)

        R = data["annotations"][0]["rotation"].reshape(3, 3)
        t = data["annotations"][0]["trans"].reshape(3, 1)
        Bbox = np.array(data["annotations"][0]["bbox"])
        cam_param = data["cam"].reshape(3, 3)
        obj_id = data["annotations"][0]["obj_id"]  # note: obj_id starts from 1

        # compute the 2D projections of the 3D points
        proj_xy, proj_depth = project_pts(self.unnorm_xyz[obj_id-1], cam_param, R, t)
        num_code_dir_bit = int(math.log2(self.crop_size_gt))  # number of bits of the x/y code

        if self.is_train:
            try:
                x = self.apply_augmentation(x)
            except:
                print("fail to apply_augmentation, ", rgb_fn)
            Bbox = aug_Bbox(Bbox, padding_ratio=self.padding_ratio)
            try:
                roi_x = get_roi(x, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR, resize_method=self.resize_method)
            except:
                print("fail to get_roi of rgb image, ", rgb_fn)
            try:
                roi_mask = get_roi(mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.resize_method)
            except:
                print("fail to get_roi of mask, ", mask_visib_fn, depth_fn)
            try:
                roi_entire_mask = get_roi(entire_mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.resize_method)
            except:
                print("fail to get_roi of entire mask, ", mask_fn, depth_fn)
            Bbox = get_final_Bbox(Bbox, self.resize_method, x.shape[1], x.shape[0])

        else:
            if self.Detect_Bbox != None:
                # replace the Bbox with detected Bbox
                Bbox = self.Detect_Bbox[index]
                if Bbox == None:  # no valid detection, give a dummy input
                    roi_x = torch.zeros((3, self.crop_size_img, self.crop_size_img))
                    roi_mask = torch.zeros((int(self.crop_size_gt), int(self.crop_size_gt)))
                    roi_entire_mask = torch.zeros((int(self.crop_size_gt), int(self.crop_size_gt)))
                    Bbox = np.array([0, 0, 0, 0], dtype='int')
                    roi_mask_bit = np.zeros((1, self.num_p3d))
                    pixel_x_code = np.zeros((num_code_dir_bit, self.num_p3d))
                    pixel_y_code = np.zeros((num_code_dir_bit, self.num_p3d))
                    roi_xy_ori = np.zeros((2, self.crop_size_gt, self.crop_size_gt))
                    return roi_x, roi_entire_mask, roi_mask, R, t, Bbox, cam_param, obj_id, \
                           roi_mask_bit, pixel_x_code, pixel_y_code, roi_xy_ori

            Bbox = padding_Bbox(Bbox, padding_ratio=self.padding_ratio)
            roi_x = get_roi(x, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR, resize_method=self.resize_method)
            roi_mask = get_roi(mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.resize_method)
            roi_entire_mask = get_roi(entire_mask, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.resize_method)
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
        return roi_x, roi_entire_mask, roi_mask, R, t, Bbox, cam_param, obj_id, \
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

    def get_bg_img_paths(self):
        bg_type = "VOC_table"
        bg_root = os.path.join(self.dataset_root, "VOCdevkit/VOC2012")
        hashed_file_name = hashlib.md5(
            ("{}_{}_get_bg_imgs".format(bg_type, self.num_bg_imgs)).encode("utf-8")
        ).hexdigest()
        cache_path = os.path.join(".cache/bg_paths_{}_{}.pkl".format(bg_type, hashed_file_name))
        mmcv.mkdir_or_exist(os.path.dirname(cache_path))
        if os.path.exists(cache_path):
            print("get bg_paths from cache file: {}".format(cache_path))
            bg_img_paths = mmcv.load(cache_path)
            print("num bg imgs: {}".format(len(bg_img_paths)))
            assert len(bg_img_paths) > 0
            return bg_img_paths

        print("building bg imgs cache {}...".format(bg_type))
        assert os.path.exists(bg_root), f"BG ROOT: {bg_root} does not exist"
        VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
        VOC_image_set_dir = os.path.join(VOC_root, "ImageSets/Main")
        VOC_bg_list_path = os.path.join(VOC_image_set_dir, "diningtable_trainval.txt")
        with open(VOC_bg_list_path, "r") as f:
            VOC_bg_list = [
                line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
            ]
        img_paths = [os.path.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx)) for bg_idx in VOC_bg_list]
        assert len(img_paths) > 0, len(img_paths)

        num_bg_imgs = min(len(img_paths), self.num_bg_imgs)
        bg_img_paths = np.random.choice(img_paths, num_bg_imgs)
        mmcv.dump(bg_img_paths, cache_path)
        print("num bg imgs: {}".format(len(bg_img_paths)))
        assert len(bg_img_paths) > 0
        return bg_img_paths

    def replace_bg(self, im, im_mask, return_mask=False):
        # add background to the image
        H, W = im.shape[:2]
        ind = random.randint(0, len(self.bg_img_paths) - 1)
        filename = self.bg_img_paths[ind]
        bg_img = get_bg_image(filename, H, W)

        if len(bg_img.shape) != 3:
            bg_img = np.zeros((H, W, 3), dtype=np.uint8)
            print("bad background image: {}".format(filename))

        mask = im_mask.copy().astype(np.bool)
        mask_bg = ~mask
        im[mask_bg] = bg_img[mask_bg]
        im = im.astype(np.uint8)
        if return_mask:
            return im, mask  # bool fg mask
        else:
            return im


def load_lm_obj_diameters(model_info_path):
    model_info = mmcv.load(model_info_path)
    diameter_dict = {}
    for idx in range(15):
        obj_info = model_info[str(idx+1)]
        diameter_dict[idx+1] = obj_info["diameter"]
    return diameter_dict

def load_lm_obj_sym_info(model_info_path):
    model_info = mmcv.load(model_info_path)
    sym_info_dict = {}
    for idx in range(15):
        obj_info = model_info[str(idx + 1)]
        if "symmetries_discrete" in obj_info or "symmetries_continuous" in obj_info:
            sym_transforms = misc.get_symmetry_transformations(obj_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        else:
            sym_info = None
        sym_info_dict[idx+1] = sym_info
    return sym_info_dict
