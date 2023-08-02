import argparse
import os

# for YCBV configs, change:
# 1. dataset_name = ycbv
# 2. second_dataset_ratio = 0.875
# 3. Detection_reaults = detection_results/ycbv/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_ycbv_real_pbr_8e_test_keyframe.json

parser = argparse.ArgumentParser("convert the config of lmo to ycbv")
parser.add_argument("-cfg", type=str, help="name of the config, without .txt")
args = parser.parse_args()

# load LMO config
lmo_cfg_path = "config/lmo/{}.txt".format(args.cfg)
with open(lmo_cfg_path, 'r') as f:
    lmo_cfg_lines = f.readlines()

# convert to YCBV config
ycbv_cfg_path = "config/ycbv/{}.txt".format(args.cfg)
if os.path.exists(ycbv_cfg_path):
    print("YCBV config already exists: ", ycbv_cfg_path)
else:
    with open(ycbv_cfg_path, 'w') as f:
        for line in lmo_cfg_lines:
            if line.startswith("dataset_name = "):
                new_line = "dataset_name = ycbv\n"
            elif line.startswith("second_dataset_ratio = "):
                new_line = "second_dataset_ratio = 0.875\n"
            elif line.startswith("Detection_reaults = "):
                new_line = "Detection_reaults = detection_results/ycbv/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_ycbv_real_pbr_8e_test_keyframe.json\n"
            else:
                new_line = line
            f.write(new_line)
    print("config saved to: ", ycbv_cfg_path)
