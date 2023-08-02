''' compute the FPS points of the model '''
import os.path

import numpy as np
import argparse
from plyfile import PlyData
import mmcv

lm_obj_name_obj_id = {
    'ape': 1,
    'benchvise': 2,
    'bowl': 3,
    'camera': 4,
    'can': 5,
    'cat': 6,
    'cup': 7,
    'driller': 8,
    'duck': 9,
    'eggbox': 10,
    'glue': 11,
    'holepuncher': 12,
    'iron': 13,
    'lamp': 14,
    'phone': 15,
}
lmo_obj_name_obj_id = {
    'ape': 1,
    'can': 5,
    'cat': 6,
    'driller': 8,
    'duck': 9,
    'eggbox': 10,
    'glue': 11,
    'holepuncher': 12,
}
ycbv_obj_name_obj_id = {
    'master_chef_can': 1,
    'cracker_box': 2,
    'sugar_box': 3,
    'tomato_soup_can': 4,
    'mustard_bottle': 5,
    'tuna_fish_can': 6,
    'pudding_box': 7,
    'gelatin_box': 8,
    'potted_meat_can': 9,
    'banana': 10,
    'pitcher_base': 11,
    'bleach_cleanser': 12,
    'bowl': 13,
    'mug': 14,
    'power_drill': 15,
    'wood_block': 16,
    'scissors': 17,
    'large_marker': 18,
    'large_clamp': 19,
    'extra_large_clamp': 20,
    'foam_brick': 21,
}
dataset_obj_name_obj_id = {
    "lm": lm_obj_name_obj_id,
    "lmo": lmo_obj_name_obj_id,
    "ycbv": ycbv_obj_name_obj_id
}

def farthest_point_sample_init_center(xyz, npoint):
    ''' compute the FPS points of the given 3D points
    Args:
        xyz: point cloud data, shape (N, 3)
        npoint: number of FPS points
    '''
    num_xyz = xyz.shape[0]
    # first compute the center (as average of the min and max coordinates)
    xyz_max = xyz.max(axis=0)
    xyz_min = xyz.min(axis=0)
    xyz_center = (xyz_max + xyz_min) / 2
    xyz_extent = np.linalg.norm(xyz_max - xyz_min)
    farthest_xyz = xyz_center
    fps_xyz = np.zeros((npoint, 3))
    fps_ids = []
    distances_to_set = np.ones(num_xyz) * xyz_extent * 10
    for sample_id in range(npoint):
        distances = np.linalg.norm((xyz - farthest_xyz), axis=1)
        mask = distances < distances_to_set
        distances_to_set[mask] = distances[mask]
        farthest_id = np.argmax(distances_to_set)
        farthest_xyz = xyz[farthest_id, :]
        # update the record
        fps_ids.append(farthest_id)
        fps_xyz[sample_id, :] = farthest_xyz
    return fps_ids, fps_xyz

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate the FPS points for the CAD model")
    parser.add_argument('-dataset', choices=['lmo', 'ycbv', 'lm'])
    parser.add_argument('-npoint_log2', type=int, default=12, help="log2 of the number of FPS points")
    args = parser.parse_args()
    dataset = args.dataset
    npoint = int(2 ** args.npoint_log2)

    fps_output_root = "../datasets/BOP_DATASETS/{}/fps_202212".format(dataset)
    if not os.path.exists(fps_output_root):
        os.makedirs(fps_output_root)
    # loop the dataset to generate the FPS points
    obj_name_obj_id = dataset_obj_name_obj_id[dataset]
    for obj_name, obj_id in obj_name_obj_id.items():
        # load the vertices
        ply_path = "../datasets/BOP_DATASETS/{}/models/obj_{:06d}.ply".format(dataset, obj_id)
        ply_data = PlyData.read(ply_path)
        ply_x = ply_data['vertex']['x']
        ply_y = ply_data['vertex']['y']
        ply_z = ply_data['vertex']['z']
        ply_xyz = np.zeros((len(ply_x), 3))
        ply_xyz[:, 0] = ply_x
        ply_xyz[:, 1] = ply_y
        ply_xyz[:, 2] = ply_z
        print("[{}|{}] xyz.shape {}".format(dataset, obj_id, ply_xyz.shape))
        # obtain the FPS points
        fps_ids, fps_xyz = farthest_point_sample_init_center(ply_xyz, npoint)
        # save the FPS points
        fps_output_path = os.path.join(fps_output_root, "obj_{:06d}.pkl".format(obj_id))
        fps_output_dict = {"npoint": npoint, "id": fps_ids, "xyz": fps_xyz}
        mmcv.dump(fps_output_dict, fps_output_path)
        print("FPS points saved to {}".format(fps_output_path))
