''' determine the overall visibility of a model
reference: Zhang, Eugene, and Greg Turk. "Visibility-guided simplification." IEEE Visualization, 2002. VIS 2002.. IEEE, 2002.
'''
import numpy as np
from scipy.spatial import ConvexHull
import mmcv
import argparse
import sys
sys.path.append("../../bop_toolkit")
from bop_toolkit_lib import inout

parser = argparse.ArgumentParser("compute overall visibility for a given object")
parser.add_argument('-dataset', default="lmo", choices=['lmo', 'ycbv'], help="dataset name")
parser.add_argument('-obj_id', type=int, help="object id")
parser.add_argument("-r", type=float, default=2.0, help="radius parameter in HPR operator")
args = parser.parse_args()
dataset = args.dataset
obj_id = args.obj_id

def compute_vis_hpr(points, viewpoint=None, radius_param=2.0):
    """ compute the visibility from viewpoint for each points
    Reference: Katz, Sagi, Ayellet Tal, and Ronen Basri. "Direct visibility of point sets." ACM SIGGRAPH 2007 papers. 2007. 24-es.
    Args:
        points: 3D coordinates (in camera space), shape: (n, d)
        radius_param: a param in Algorithm 1 to determine radius R of the sphere
    """
    num_points, dim = points.shape
    points_all = np.zeros((num_points+1, dim), dtype=points.dtype)  # the 0th point is the viewpoint
    points_centered = points - viewpoint if viewpoint is not None else points
    # determine a reasonable radius
    points_norm = np.linalg.norm(points_centered, axis=1)  # shape: (n,)
    radius = np.max(points_norm) * (10 ** radius_param)
    # print("radius: {}".format(radius))
    # compute the spherical flipping results (eq.1)
    points_all[1:, :] = points_centered + 2 * (radius - points_norm)[:, None] * (points_centered / points_norm[:, None])
    # compute the convex hull
    hull = ConvexHull(points_all)
    # print("vertices: {}".format(hull.vertices))
    visibility = np.zeros(num_points+1)
    visibility[hull.vertices] = 1.0
    visibility = visibility[1:]  # shape: (n,) (discard the viewpoint)
    return visibility

def transform_pts_Rt(pts, R, t):
  """Applies a rigid transformation to 3D points.

  :param pts: nx3 ndarray with 3D points.
  :param R: 3x3 ndarray with a rotation matrix.
  :param t: 3x1 ndarray with a translation vector.
  :return: nx3 ndarray with transformed 3D points.
  """
  assert (pts.shape[1] == 3)
  pts_t = R.dot(pts.T) + t.reshape((3, 1))
  return pts_t.T

# aux info
lmo_id2obj = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}
ycbv_id2obj = {
    1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
    11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
    15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
    18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}
if dataset == "lmo":
    print("compute overall visibility for LMO {}".format(lmo_id2obj[obj_id]))
elif dataset == "ycbv":
    print("compute overall visibility for YCBV {}".format(ycbv_id2obj[obj_id]))
# load vertices
mesh_path = "../datasets/BOP_DATASETS/{}/models/obj_{:06d}.ply".format(dataset, obj_id)
vertices = inout.load_ply(mesh_path)["pts"]
num_vert = vertices.shape[0]

sampled_pose_path = "../datasets/sampled_poses_2562.pkl"
sampled_pose_list = mmcv.load(sampled_pose_path)
trans = np.array([0., 0., 400.0]).reshape((3, 1))
vert_sum_visib_mask = np.zeros(num_vert)
num_pose = len(sampled_pose_list)
for i in range(num_pose):
    rot = sampled_pose_list[i]['R'].reshape((3, 3))
    vert_cam = transform_pts_Rt(vertices, rot, trans)
    visib_mask = compute_vis_hpr(vert_cam, radius_param=args.r)
    vert_sum_visib_mask += visib_mask
    print("processed pose {}/{} visible {}/{}".format(
        i+1, num_pose, int(np.sum(visib_mask)), num_vert
    ))

vert_mean_visib_mask = vert_sum_visib_mask / num_pose
if dataset == "lmo":
    print("compute overall visibility for LMO {}".format(lmo_id2obj[obj_id]))
elif dataset == "ycbv":
    print("compute overall visibility for YCBV {}".format(ycbv_id2obj[obj_id]))
print("min {}\nmax {}".format(vert_mean_visib_mask.min(), vert_mean_visib_mask.max()))
for i in range(1, 10):
    threshold = i * 0.1  # from 0.1 to 0.9
    below_ratio = np.mean(vert_mean_visib_mask < threshold)
    print("[below {:.2f}] {:.4f}".format(threshold, below_ratio))
