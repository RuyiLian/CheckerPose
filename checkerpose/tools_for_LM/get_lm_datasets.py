# reference: https://github.com/THU-DA-6D-Pose-Group/GDR-Net/blob/main/core/gdrn_modeling/datasets/lm_dataset_d2.py
import hashlib
import os
import os.path as osp
import sys
import time
import mmcv
import numpy as np
from tqdm import tqdm


lm_full_id2obj = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}
lm_full_obj2id = {_name: _id for _id, _name in lm_full_id2obj.items()}


def mask2bbox_xywh(mask):
    ys, xs = np.nonzero(mask)[:2]
    bb_tl = [xs.min(), ys.min()]
    bb_br = [xs.max(), ys.max()]
    return [bb_tl[0], bb_tl[1], bb_br[0] - bb_tl[0] + 1, bb_br[1] - bb_tl[1] + 1]


def get_lm_13_dicts(data_cfg):
    name = data_cfg["name"]
    objs = data_cfg["objs"]  # selected objects
    ann_files = data_cfg["ann_files"]  # idx files with image ids
    image_prefixes = data_cfg["image_prefixes"]
    code_prefixes = data_cfg["code_prefixes"]
    dataset_root = data_cfg["dataset_root"]  # BOP_DATASETS/lm/
    assert osp.exists(dataset_root), dataset_root
    height = data_cfg["height"]  # 480
    width = data_cfg["width"]  # 640
    num_to_load = data_cfg["num_to_load"]  # -1
    filter_invalid = data_cfg["filter_invalid"]
    filter_scene = data_cfg.get("filter_scene", False)
    cache_dir = data_cfg["cache_dir"]
    # record the CAD model ids for selected objects
    cat_ids = [cat_id for cat_id, obj_name in lm_full_id2obj.items() if obj_name in objs]

    hashed_file_name = hashlib.md5(
        (
                "".join([str(fn) for fn in objs])
                + "dataset_dicts_{}_{}_{}".format(
            name, dataset_root, __name__
        )
        ).encode("utf-8")
    ).hexdigest()
    cache_path = osp.join(cache_dir, "dataset_dicts_{}_{}.pkl".format(name, hashed_file_name))

    if osp.exists(cache_path):
        print("load cached dataset dicts from {}".format(cache_path))
        return mmcv.load(cache_path)

    t_start = time.perf_counter()
    print("loading dataset dicts: {}".format(name))
    num_instances_without_valid_segmentation = 0
    num_instances_without_valid_box = 0
    dataset_dicts = []  # ######################################################
    assert len(ann_files) == len(image_prefixes), f"{len(ann_files)} != {len(image_prefixes)}"
    assert len(ann_files) == len(code_prefixes), f"{len(ann_files)} != {len(code_prefixes)}"
    for ann_file, scene_root, code_root in zip(tqdm(ann_files), image_prefixes, code_prefixes):
        with open(ann_file, "r") as f_ann:
            indices = [line.strip("\r\n") for line in f_ann.readlines()]  # string ids
        gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
        gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))  # bbox_obj, bbox_visib
        cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))
        for im_id in tqdm(indices):
            int_im_id = int(im_id)
            str_im_id = str(int_im_id)
            rgb_path = osp.join(scene_root, "rgb/{:06d}.png").format(int_im_id)
            assert osp.exists(rgb_path), rgb_path
            depth_path = osp.join(scene_root, "depth/{:06d}.png".format(int_im_id))

            scene_id = int(rgb_path.split("/")[-3])
            scene_im_id = f"{scene_id}/{int_im_id}"

            K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
            if filter_scene:
                if scene_id not in cat_ids:
                    continue
            record = {
                "dataset_name": name,
                "file_name": rgb_path,
                "depth_file": depth_path,
                "height": height,
                "width": width,
                "image_id": int_im_id,
                "scene_im_id": scene_im_id,  # for evaluation
                "cam": K,
                "img_type": "real",
            }

            insts = []
            for anno_i, anno in enumerate(gt_dict[str_im_id]):
                obj_id = anno["obj_id"]
                if obj_id not in cat_ids:
                    continue
                R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                t = np.array(anno["cam_t_m2c"], dtype="float32")   # note: remain the unit to mm

                bbox_visib = gt_info_dict[str_im_id][anno_i]["bbox_visib"]
                x1, y1, w, h = bbox_visib
                if filter_invalid:
                    if h <= 1 or w <= 1:
                        num_instances_without_valid_box += 1
                        continue

                mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i))
                assert osp.exists(mask_file), mask_file
                assert osp.exists(mask_visib_file), mask_visib_file
                mask_single = mmcv.imread(mask_visib_file, "unchanged")
                area = mask_single.sum()
                if area < 3:  # filter out too small or nearly invisible instances
                    num_instances_without_valid_segmentation += 1
                    continue

                code_file = osp.join(code_root, f"{int_im_id:06d}_{anno_i:06d}.pkl")
                inst = {
                    "obj_id": obj_id,
                    "bbox": bbox_visib, # xyhw
                    "rotation": R,
                    "trans": t,
                    "mask_visib_file": mask_visib_file,
                    "mask_file": mask_file,
                    "code_path": code_file,
                }
                insts.append(inst)
            if len(insts) == 0:  # filter im without anno
                continue
            record["annotations"] = insts
            dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        print(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    if num_instances_without_valid_box > 0:
        print(
            "Filtered out {} instances without valid box. "
            "There might be issues in your dataset generation process.".format(num_instances_without_valid_box)
        )

    if num_to_load > 0:
        num_to_load = min(int(num_to_load), len(dataset_dicts))
        dataset_dicts = dataset_dicts[: num_to_load]
    print("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

    mmcv.mkdir_or_exist(osp.dirname(cache_path))
    mmcv.dump(dataset_dicts, cache_path, protocol=4)
    print("Dumped dataset_dicts to {}".format(cache_path))
    return dataset_dicts


# lm synthetic data, imgn(imagine) from DeepIM
def get_imgn_dicts(data_cfg):
    name = data_cfg["name"]
    objs = data_cfg["objs"]  # selected objects
    ann_files = data_cfg["ann_files"]  # idx files with image ids
    image_prefixes = data_cfg["image_prefixes"]
    code_prefixes = data_cfg["code_prefixes"]
    dataset_root = data_cfg["dataset_root"]  # lm_imgn
    models_root = data_cfg["models_root"]  # BOP_DATASETS/lm/models
    cam = data_cfg["cam"]  #
    height = data_cfg["height"]  # 480
    width = data_cfg["width"]  # 640

    # sample uniformly to get n items
    n_per_obj = data_cfg.get("n_per_obj", 1000)
    filter_invalid = data_cfg["filter_invalid"]
    filter_scene = data_cfg.get("filter_scene", False)
    if cam is None:
        cam = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    # only the selected objects
    cat_ids = [cat_id for cat_id, obj_name in lm_full_id2obj.items() if obj_name in objs]

    # cache the dataset_dicts
    hashed_file_name = hashlib.md5(
        (
                "".join([str(fn) for fn in objs])
                + "dataset_dicts_{}_{}_{}_{}".format(
            name, dataset_root, n_per_obj, __name__
        )
        ).encode("utf-8")
    ).hexdigest()
    cache_path = osp.join(dataset_root, "dataset_dicts_{}_{}.pkl".format(name, hashed_file_name))

    if osp.exists(cache_path):
        print("load cached dataset dicts from {}".format(cache_path))
        return mmcv.load(cache_path)

    t_start = time.perf_counter()
    print("loading dataset dicts: {}".format(name))
    num_instances_without_valid_segmentation = 0
    num_instances_without_valid_box = 0
    dataset_dicts = []
    assert len(ann_files) == len(image_prefixes), f"{len(ann_files)} != {len(image_prefixes)}"
    assert len(ann_files) == len(code_prefixes), f"{len(ann_files)} != {len(code_prefixes)}"
    for ann_file, scene_root, code_root in zip(ann_files, image_prefixes, code_prefixes):
        # linemod each scene is an object
        with open(ann_file, "r") as f_ann:
            indices = [line.strip("\r\n").split()[-1] for line in f_ann.readlines()]  # string ids
        # sample uniformly (equal space)
        if n_per_obj > 0:
            sample_num = min(n_per_obj, len(indices))
            sel_indices_idx = np.linspace(0, len(indices) - 1, sample_num, dtype=np.int32)
            sel_indices = [indices[int(_i)] for _i in sel_indices_idx]
        else:
            sel_indices = indices

        for im_id in tqdm(sel_indices):
            rgb_path = osp.join(scene_root, "{}-color.png").format(im_id)
            assert osp.exists(rgb_path), rgb_path

            depth_path = osp.join(scene_root, "{}-depth.png".format(im_id))

            obj_name = im_id.split("/")[0]
            if obj_name == "benchviseblue":
                obj_name = "benchvise"
            obj_id = lm_full_obj2id[obj_name]
            if filter_scene:
                if obj_name not in objs:
                    continue
            record = {
                "dataset_name": name,
                "file_name": rgb_path,
                "depth_file": depth_path,
                "height": height,
                "width": width,
                "image_id": im_id.split("/")[-1],
                "scene_im_id": im_id,
                "cam": cam,
                "img_type": "syn",
            }

            pose_path = osp.join(scene_root, "{}-pose.txt".format(im_id))
            pose = np.loadtxt(pose_path, skiprows=1)
            R = pose[:3, :3]
            t = pose[:3, 3] * 1000.0  # convert unit from mm to m

            depth = mmcv.imread(depth_path, "unchanged")
            mask = (depth > 0).astype(np.uint8)

            bbox_obj = mask2bbox_xywh(mask)
            x1, y1, w, h = bbox_obj
            if filter_invalid:
                if h <= 1 or w <= 1:
                    num_instances_without_valid_box += 1
                    continue
            area = mask.sum()
            if area < 3:  # filter out too small or nearly invisible instances
                num_instances_without_valid_segmentation += 1
                continue

            code_path = osp.join(code_root, f"{im_id}-id.pkl")
            assert osp.exists(code_path), code_path
            inst = {
                "obj_id": obj_id,
                "bbox": bbox_obj,  # xyhw
                "rotation": R,
                "trans": t,
                "mask_visib_file": None,  # the mask need to be computed from the depth
                "mask_file": None,
                "code_path": code_path,
            }
            record["annotations"] = [inst]
            dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        print(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    if num_instances_without_valid_box > 0:
        print(
            "Filtered out {} instances without valid box. "
            "There might be issues in your dataset generation process.".format(num_instances_without_valid_box)
        )

    print(
        "loaded dataset dicts, num_images: {}, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start)
    )
    mmcv.dump(dataset_dicts, cache_path, protocol=4)
    print("Dumped dataset_dicts to {}".format(cache_path))
    return dataset_dicts


LM_13_OBJECTS = [
    "ape",
    "benchvise",
    "camera",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone",
]  # no bowl, cup


def get_lm_13_train_cfg(datasets_root):
    lm_13_train = dict(
        name="lm_13_train",
        dataset_root=osp.join(datasets_root, "BOP_DATASETS/lm/"),
        models_root=osp.join(datasets_root, "BOP_DATASETS/lm/models"),
        objs=LM_13_OBJECTS,  # selected objects
        ann_files=[
            osp.join(datasets_root, "BOP_DATASETS/lm/image_set/{}_{}.txt".format(_obj, "train"))
            for _obj in LM_13_OBJECTS
        ],
        image_prefixes=[
            osp.join(datasets_root, "BOP_DATASETS/lm/test/{:06d}".format(lm_full_obj2id[_obj]))
            for _obj in LM_13_OBJECTS
        ],
        code_prefixes=[
            osp.join(datasets_root, "BOP_DATASETS/lm/test/code_images/{:06d}".format(lm_full_obj2id[_obj]))
            for _obj in LM_13_OBJECTS
        ],
        height=480,
        width=640,
        cache_dir=osp.join(datasets_root, ".cache"),
        num_to_load=-1,
        filter_scene=True,
        filter_invalid=True,
    )
    return lm_13_train


def get_lm_13_test_cfg(datasets_root):
    lm_13_test = dict(
        name="lm_13_test",
        dataset_root=osp.join(datasets_root, "BOP_DATASETS/lm/"),
        models_root=osp.join(datasets_root, "BOP_DATASETS/lm/models"),
        objs=LM_13_OBJECTS,
        ann_files=[
            osp.join(datasets_root, "BOP_DATASETS/lm/image_set/{}_{}.txt".format(_obj, "test"))
            for _obj in LM_13_OBJECTS
        ],
        # NOTE: scene root
        image_prefixes=[
            osp.join(datasets_root, "BOP_DATASETS/lm/test/{:06d}").format(lm_full_obj2id[_obj])
            for _obj in LM_13_OBJECTS
        ],
        code_prefixes=[
            osp.join(datasets_root, "BOP_DATASETS/lm/test/code_images/{:06d}".format(lm_full_obj2id[_obj]))
            for _obj in LM_13_OBJECTS
        ],
        height=480,
        width=640,
        cache_dir=osp.join(datasets_root, ".cache"),
        num_to_load=-1,
        filter_scene=True,
        filter_invalid=False,
    )
    return lm_13_test


def get_lm_imgn_13_train_1k_per_obj_cfg(datasets_root):
    lm_imgn_13_train_1k_per_obj = dict(
        name="lm_imgn_13_train_1k_per_obj",  # BB8 training set
        dataset_root=osp.join(datasets_root, "lm_imgn/"),
        models_root=osp.join(datasets_root, "BOP_DATASETS/lm/models"),
        objs=LM_13_OBJECTS,  # selected objects
        ann_files=[
            osp.join(datasets_root, "lm_imgn/image_set/{}_{}.txt".format("train", _obj)) for _obj in LM_13_OBJECTS
        ],
        image_prefixes=[osp.join(datasets_root, "lm_imgn/imgn") for _obj in LM_13_OBJECTS],
        code_prefixes=[osp.join(datasets_root, "lm_imgn/code_images/") for _obj in LM_13_OBJECTS],
        cam=np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]),
        height=480,
        width=640,
        cache_dir=osp.join(datasets_root, ".cache"),
        n_per_obj=1000,  # 1000 per class
        filter_scene=True,
        filter_invalid=False,
    )
    return lm_imgn_13_train_1k_per_obj


def get_lm_data_dicts(dataset_name, datasets_root):
    if dataset_name == "lm_13_train":
        data_cfg = get_lm_13_train_cfg(datasets_root)
        data_dicts = get_lm_13_dicts(data_cfg)
    elif dataset_name == "lm_13_test":
        data_cfg = get_lm_13_test_cfg(datasets_root)
        data_dicts = get_lm_13_dicts(data_cfg)
    elif dataset_name == "lm_imgn_13_train_1k_per_obj":
        data_cfg = get_lm_imgn_13_train_1k_per_obj_cfg(datasets_root)
        data_dicts = get_imgn_dicts(data_cfg)
    else:
        raise ValueError("dataset {} not supported".format(dataset_name))
    return data_dicts
