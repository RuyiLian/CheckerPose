from binary_code_helper.class_id_encoder_decoder import class_code_images_to_class_id_image
import numpy as np
import cv2
# import pyprogressivex

def load_dict_class_id_3D_points(path):
    total_numer_class = 0
    number_of_itration = 0

    dict_class_id_3D_points = {}
    with open(path, "r") as f:
        first_line = f.readline()
        total_numer_class_, divide_number_each_itration, number_of_itration_ = first_line.split(" ") 
        divide_number_each_itration = float(divide_number_each_itration)
        total_numer_class = float(total_numer_class_)
        number_of_itration = float(number_of_itration_)

        for line in f:
            line = line[:-1]
            code, x, y, z= line.split(" ")
            code = float(code)
            x = float(x)
            y = float(y)
            z = float(z)

            dict_class_id_3D_points[code] = np.array([x,y,z])

    return total_numer_class, divide_number_each_itration, number_of_itration, dict_class_id_3D_points

def mapping_pixel_position_to_original_position(pixels, Bbox, Bbox_Size):
    """
    The image was cropped and resized. This function returns the original pixel position
    input:
        pixels: pixel position after cropping and resize, which is a numpy array, N*(X, Y)
        Bbox: Bounding box for the cropping, minx miny width height
    """
    ratio_x = Bbox[2] / Bbox_Size
    ratio_y = Bbox[3] / Bbox_Size

    original_pixel_x = ratio_x*pixels[:,0] + Bbox[0]     
    original_pixel_y = ratio_y*pixels[:,1] + Bbox[1] 

    original_pixel_x = original_pixel_x.astype('int')
    original_pixel_y = original_pixel_y.astype('int')

    return np.concatenate((original_pixel_x.reshape(-1, 1), original_pixel_y.reshape(-1, 1)), 1)


def mapping_roi_uv_to_original_position(roi_u, roi_v, Bbox):
    """
    The image was cropped and resized. This function returns the original pixel position
    input:
        roi_u, roi_v: 2D position after cropping and resize, which is a numpy array, with shape (N,)
        Bbox: Bounding box for the cropping, minx miny width height
    """
    abs_u = Bbox[2] * roi_u + Bbox[0]
    abs_v = Bbox[3] * roi_v + Bbox[1]
    return np.concatenate((abs_u.reshape(-1, 1), abs_v.reshape(-1, 1)), 1)


def build_non_unique_2D_3D_correspondence(Pixel_position, class_id_image, dict_class_id_3D_points):
    Point_2D = np.concatenate((Pixel_position[1].reshape(-1, 1), Pixel_position[0].reshape(-1, 1)), 1)   #(npoint x 2)
    
    ids_for_searching = class_id_image[Point_2D[:, 1], Point_2D[:, 0]]

    Points_3D = np.zeros((Point_2D.shape[0],3))
    for i in range(Point_2D.shape[0]):
        if np.isnan(np.array(dict_class_id_3D_points[ids_for_searching[i]])).any():
            continue         
        Points_3D[i] = np.array(dict_class_id_3D_points[ids_for_searching[i]])

    return Point_2D, Points_3D


def build_unique_2D_3D_correspondence(Pixel_position, class_id_image, dict_class_id_3D_points):
    # if multiple 2D pixel match to a 3D vertex. For this vertex, its corres pixel will be the mean position of those pixels

    Point_2D = np.concatenate((Pixel_position[1].reshape(-1, 1), Pixel_position[0].reshape(-1, 1)), 1)   #(npoint x 2)
    ids_for_searching = class_id_image[Point_2D[:, 1], Point_2D[:, 0]]

    #build a dict for 3D points and all 2D pixel
    unique_3D_2D_corres = {}
    for i in range(Point_2D.shape[0]):      
        if ids_for_searching[i] in unique_3D_2D_corres.keys():
            unique_3D_2D_corres[ids_for_searching[i]].append(Point_2D[i])
        else:
            unique_3D_2D_corres[ids_for_searching[i]] = [Point_2D[i]]

    Points_3D = np.zeros((len(unique_3D_2D_corres),3))
    Points_2D = np.zeros((len(unique_3D_2D_corres),2))
    for counter, (key, value) in enumerate(unique_3D_2D_corres.items()):
        Points_3D[counter] = dict_class_id_3D_points[key]
        sum_Pixel_2D = np.zeros((1,2))
        for Pixel_2D in value:
            sum_Pixel_2D = sum_Pixel_2D + Pixel_2D
        unique_Pixel_2D = sum_Pixel_2D / len(value)
        Points_2D[counter] = unique_Pixel_2D

    return Points_2D, Points_3D


def get_class_id_image_validmask(class_id_image):
    mask_image = np.zeros(class_id_image.shape)
    mask_image[class_id_image.nonzero()]=1
    return mask_image


# for inference during training
# [shape] pred_xyz: (h, w, 3), xyz_centroid numpy array (3,), xyz_range is a scalar
def pred_corr_to_object_pose(mask_image, pred_xyz, Bbox, Bbox_Size, xyz_centroid, xyz_range, intrinsic_matrix=None,
                             use_progressivex=False):
    if intrinsic_matrix is None:
        intrinsic_matrix = np.zeros((3, 3))
        intrinsic_matrix[0, 0] = 572.4114  # fx
        intrinsic_matrix[1, 1] = 573.57043  # fy
        intrinsic_matrix[0, 2] = 325.2611  # cx
        intrinsic_matrix[1, 2] = 242.04899  # cy
        intrinsic_matrix[2, 2] = 1.0
    Points_2D = mask_image.nonzero()
    # find the 2D-3D correspondences and Ransac + PnP
    success = False
    rot = []
    tvecs = []
    if Points_2D[0].size != 0:
        Points_2D = np.concatenate((Points_2D[1].reshape(-1, 1), Points_2D[0].reshape(-1, 1)), 1)   #(#point, 2)
        Points_3D = pred_xyz[Points_2D[:, 1], Points_2D[:, 0]]  # shape: (#point, 3)
        Points_3D = Points_3D * xyz_range + xyz_centroid
        Original_Points_2D = mapping_pixel_position_to_original_position(Points_2D, Bbox, Bbox_Size)

        if len(Original_Points_2D) >= 6:
            success = True
            intrinsic_matrix = np.ascontiguousarray(intrinsic_matrix)
            if use_progressivex:
                import pyprogressivex
                coord_2d = np.ascontiguousarray(Original_Points_2D.astype(np.float64))
                coord_3d = np.ascontiguousarray(Points_3D.astype(np.float64))
                pose_ests, label = pyprogressivex.find6DPoses(
                    x1y1=coord_2d,
                    x2y2z2=coord_3d,
                    K=intrinsic_matrix.astype(np.float64),
                    threshold=2,
                    neighborhood_ball_radius=20,
                    spatial_coherence_weight=0.1,
                    maximum_tanimoto_similarity=0.9,
                    max_iters=400,
                    minimum_point_number=6,
                    maximum_model_number=1
                )
                if pose_ests.shape[0] != 0:
                    rot = pose_ests[0:3, :3]
                    tvecs = pose_ests[0:3, 3]
                    tvecs = tvecs.reshape((3, 1))
                else:
                    rot = np.zeros((3, 3))
                    tvecs = np.zeros((3, 1))
                    success = False
            else:
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(Points_3D.astype(np.float32),
                                                              Original_Points_2D.astype(np.float32), intrinsic_matrix,
                                                              distCoeffs=None,
                                                              reprojectionError=2, iterationsCount=150,
                                                              flags=cv2.SOLVEPNP_EPNP)
                rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
    return rot, tvecs, success


# for inference during training
# [shape] pred_xyz: (h, w, 3), xyz_centroid numpy array (3,), xyz_range is a scalar
def debug_pred_corr_to_object_pose(mask_image, pred_xyz, Bbox, Bbox_Size, xyz_centroid, xyz_range, intrinsic_matrix=None):

    if intrinsic_matrix is None:
        intrinsic_matrix = np.zeros((3, 3))
        intrinsic_matrix[0, 0] = 572.4114  # fx
        intrinsic_matrix[1, 1] = 573.57043  # fy
        intrinsic_matrix[0, 2] = 325.2611  # cx
        intrinsic_matrix[1, 2] = 242.04899  # cy
        intrinsic_matrix[2, 2] = 1.0
    Points_2D = mask_image.nonzero()
    # find the 2D-3D correspondences and Ransac + PnP
    success = False
    rot = []
    tvecs = []
    if Points_2D[0].size != 0:
        Points_2D = np.concatenate((Points_2D[1].reshape(-1, 1), Points_2D[0].reshape(-1, 1)), 1)   #(#point, 2)
        Points_3D = pred_xyz[Points_2D[:, 1], Points_2D[:, 0]]  # shape: (#point, 3)
        Points_3D = Points_3D * xyz_range + xyz_centroid
        Original_Points_2D = mapping_pixel_position_to_original_position(Points_2D, Bbox, Bbox_Size)

        if len(Original_Points_2D) >= 6:
            # debug using progressivex
            success = True
            coord_2d = np.ascontiguousarray(Original_Points_2D.astype(np.float32))
            coord_3d = np.ascontiguousarray(Points_3D.astype(np.float32))
            intrinsic_matrix = np.ascontiguousarray(intrinsic_matrix)
            # import pyprogressivex
            # pose_ests, label = pyprogressivex.find6DPoses(
            #     x1y1=coord_2d.astype(np.float64),
            #     x2y2z2=coord_3d.astype(np.float64),
            #     K=intrinsic_matrix.astype(np.float64),
            #     threshold=2,
            #     neighborhood_ball_radius=20,
            #     spatial_coherence_weight=0.1,
            #     maximum_tanimoto_similarity=0.9,
            #     max_iters=400,
            #     minimum_point_number=6,
            #     maximum_model_number=1
            # )
            # if pose_ests.shape[0] != 0:
            #     rot = pose_ests[0:3, :3]
            #     tvecs = pose_ests[0:3, 3]
            #     tvecs = tvecs.reshape((3, 1))
            # else:
            #     rot = np.zeros((3, 3))
            #     tvecs = np.zeros((3, 1))
            #     success = False

            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(Points_3D.astype(np.float32),
                                                          Original_Points_2D.astype(np.float32), intrinsic_matrix,
                                                          distCoeffs=None,
                                                          reprojectionError=2, iterationsCount=150,
                                                          flags=cv2.SOLVEPNP_EPNP)
            rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
    return rot, tvecs, success, Points_3D, Original_Points_2D


def CNN_outputs_to_object_pose(mask_image, class_code_image, Bbox, Bbox_Size, class_base=2, dict_class_id_3D_points=None, intrinsic_matrix=None,
                               use_progressivex=False, neighborhood_ball_radius=20, spatial_coherence_weight=0.1,
                               prog_max_iters=400):
    if intrinsic_matrix is None:
        intrinsic_matrix = np.zeros((3,3))

        intrinsic_matrix[0,0] = 572.4114             #fx
        intrinsic_matrix[1,1] = 573.57043            #fy
        intrinsic_matrix[0,2] = 325.2611             #cx
        intrinsic_matrix[1,2] = 242.04899            #cy
        intrinsic_matrix[2,2] = 1.0 

    class_id_image = class_code_images_to_class_id_image(class_code_image, class_base)
    Points_2D = mask_image.nonzero()
        
    # find the 2D-3D correspondences and Ransac + PnP
    build_2D_3D_correspondence = build_non_unique_2D_3D_correspondence
    success = False
    rot = []
    tvecs = []

    if Points_2D[0].size != 0:   
        Points_2D,  Points_3D = build_2D_3D_correspondence(Points_2D, class_id_image, dict_class_id_3D_points)
        # PnP needs atleast 6 unique 2D-3D correspondences to run
        # mapping the pixel position to its original position
      
        Original_Points_2D = mapping_pixel_position_to_original_position(Points_2D, Bbox, Bbox_Size)

        if len(Original_Points_2D) >= 6:
            success = True
            coord_2d = np.ascontiguousarray(Original_Points_2D.astype(np.float32))
            coord_3d = np.ascontiguousarray(Points_3D.astype(np.float32))
            intrinsic_matrix = np.ascontiguousarray(intrinsic_matrix)
            if use_progressivex:
                import pyprogressivex
                try:
                    pose_ests, label = pyprogressivex.find6DPoses(
                                                                x1y1=coord_2d.astype(np.float64),
                                                                x2y2z2=coord_3d.astype(np.float64),
                                                                K=intrinsic_matrix.astype(np.float64),
                                                                threshold=2,
                                                                neighborhood_ball_radius=neighborhood_ball_radius,
                                                                spatial_coherence_weight=spatial_coherence_weight,
                                                                maximum_tanimoto_similarity=0.9,
                                                                max_iters=prog_max_iters,
                                                                minimum_point_number=6,
                                                                maximum_model_number=1
                                                            )
                    if pose_ests.shape[0] != 0:
                        rot = pose_ests[0:3, :3]
                        tvecs = pose_ests[0:3, 3]
                        tvecs = tvecs.reshape((3,1))
                    else:
                        rot = np.zeros((3,3))
                        tvecs = np.zeros((3,1))
                        success = False
                except:  # use PnP when error occurs
                    print("progressive-x failed, for {} points.".format(len(Original_Points_2D)))
                    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(Points_3D.astype(np.float32),
                                                                  Original_Points_2D.astype(np.float32),
                                                                  intrinsic_matrix, distCoeffs=None,
                                                                  reprojectionError=2, iterationsCount=150,
                                                                  flags=cv2.SOLVEPNP_EPNP)
                    rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
            
            else:
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(Points_3D.astype(np.float32),
                                                            Original_Points_2D.astype(np.float32), intrinsic_matrix, distCoeffs=None,
                                                            reprojectionError=2, iterationsCount=150, flags=cv2.SOLVEPNP_EPNP)
                rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
    return rot, tvecs, success


def point_visib_uv_to_object_pose(point_visib_mask, point_roi_u, point_roi_v, Bbox, Bbox_Size, p3d_unnorm,
                                  intrinsic_matrix=None, use_progressivex=False, neighborhood_ball_radius=20,
                                  spatial_coherence_weight=0.1, prog_max_iters=400):
    if intrinsic_matrix is None:
        intrinsic_matrix = np.zeros((3, 3))
        intrinsic_matrix[0, 0] = 572.4114  # fx
        intrinsic_matrix[1, 1] = 573.57043  # fy
        intrinsic_matrix[0, 2] = 325.2611  # cx
        intrinsic_matrix[1, 2] = 242.04899  # cy
        intrinsic_matrix[2, 2] = 1.0

    success = False
    rot = []
    tvecs = []

    Original_Points_2D = mapping_roi_uv_to_original_position(roi_u=point_roi_u, roi_v=point_roi_v, Bbox=Bbox)
    point_visib_mask = point_visib_mask.astype(bool)
    Original_Points_2D = Original_Points_2D[point_visib_mask]
    Original_Points_3D = p3d_unnorm[point_visib_mask]


    if len(Original_Points_2D) >= 6:
        success = True
        coord_2d = np.ascontiguousarray(Original_Points_2D.astype(np.float32))
        coord_3d = np.ascontiguousarray(Original_Points_3D.astype(np.float32))
        intrinsic_matrix = np.ascontiguousarray(intrinsic_matrix)
        if use_progressivex:
            import pyprogressivex
            pose_ests, label = pyprogressivex.find6DPoses(
                x1y1=coord_2d.astype(np.float64),
                x2y2z2=coord_3d.astype(np.float64),
                K=intrinsic_matrix.astype(np.float64),
                threshold=2,
                neighborhood_ball_radius=neighborhood_ball_radius,
                spatial_coherence_weight=spatial_coherence_weight,
                maximum_tanimoto_similarity=0.9,
                max_iters=prog_max_iters,
                minimum_point_number=6,
                maximum_model_number=1
            )
            if pose_ests.shape[0] != 0:
                rot = pose_ests[0:3, :3]
                tvecs = pose_ests[0:3, 3]
                tvecs = tvecs.reshape((3, 1))
            else:
                rot = np.zeros((3, 3))
                tvecs = np.zeros((3, 1))
                success = False

        else:
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(Original_Points_3D.astype(np.float32),
                                                          Original_Points_2D.astype(np.float32), intrinsic_matrix,
                                                          distCoeffs=None,
                                                          reprojectionError=2, iterationsCount=150,
                                                          flags=cv2.SOLVEPNP_EPNP)
            rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
    return rot, tvecs, success
