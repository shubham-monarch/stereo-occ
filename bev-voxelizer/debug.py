#! /usr/bin/env python3

import open3d as o3d
import cv2
import os
from tqdm import tqdm

from bev_generator import BEVGenerator
from logger import get_logger
import numpy as np
from helpers import mono_to_rgb_mask, crop_pcd, R_to_rvec, get_zed_camera_params, cam_extrinsics
from read_write_model import qvec2rotmat, read_images_binary


logger = get_logger("debug")

def calculate_rotation_matrix(vector):
    # Step 1: Normalize the vector
    a, b, c = vector
    norm = np.sqrt(a**2 + b**2 + c**2)
    v_norm = np.array([a, b, c]) / norm
    
    # Step 2: Calculate the angles between the vector and the coordinate axes
    cos_theta_x = v_norm[0]
    cos_theta_y = v_norm[1]
    cos_theta_z = v_norm[2]
    
    # Step 3: Create the rotation matrix
    rotation_matrix = np.array([
        [cos_theta_x, cos_theta_y, cos_theta_z],
        [cos_theta_x, cos_theta_y, cos_theta_z],
        [cos_theta_x, cos_theta_y, cos_theta_z]
    ])
    
    return rotation_matrix

def project_points_with_cv2(points_3d, intrinsics_matrix, image_size):
    # Ensure the points are in the required shape (N, 1, 3)
    points_3d = points_3d.reshape(-1, 1, 3)
    
    # Define rotation (rvec) and translation (tvec) as zero since points are already in the camera frame
    rvec = np.zeros((3, 1))  # No rotation
    tvec = np.zeros((3, 1))  # No translation
    
    # Distortion coefficients (assuming no distortion, use None)
    dist_coeffs = np.zeros((4, 1))  # No distortion

    # Project points to 2D
    image_points, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsics_matrix, dist_coeffs)
    
    # Convert the result to (N, 2) for convenience
    image_points = image_points.reshape(-1, 2)

    # Clip the points to stay within the image boundaries
    image_points[:, 0] = np.clip(image_points[:, 0], 0, image_size[1] - 1)
    image_points[:, 1] = np.clip(image_points[:, 1], 0, image_size[0] - 1)
    
    return image_points


if __name__ == "__main__":  


    # ================================================
    # CASE 5: check image size
    # ================================================
    
    # img_path = "data/train-data/0/_left.jpg"  # Specify the path to your image
    # image = cv2.imread(img_path)
    # if image is not None:
    #     height, width, _ = image.shape
    #     logger.info(f"Image size: {width}x{height}")
    # else:
    #     logger.error("Failed to read the image.")


    # ================================================
    # CASE 4: testing pointcloud to camera projection
    # ================================================
   
    # images = read_images_binary("debug/dense-reconstruction/images.bin")

    # sparse_image_keys = images.keys()
    # sparse_image_keys = sorted(sparse_image_keys)

    # logger.info(f"================================================")
    # logger.info(f"sparse_image_keys: {sparse_image_keys}")
    # logger.info(f"================================================\n")

    # # for idx,key in enumerate(sparse_image_keys):
    # #     logger.info(f"================================================")
    # #     logger.info(f"idx: {idx} key: {key}")
    # #     left_img = images[key]
    # #     logger.info(f"left_img.name: {left_img.name}")
    # #     logger.info(f"================================================\n")

    # # exit()
    # num_sparse_images = len(sparse_image_keys)
    
    # logger.info(f"================================================")
    # logger.info(f"num_sparse_images: {num_sparse_images}")
    # logger.info(f"================================================\n")

    # left_img = images[sparse_image_keys[76]]

    # logger.info(f"================================================")
    # logger.info(f"left_img.name: {left_img.name}")
    # logger.info(f"================================================\n")

    # C_l = cam_extrinsics(left_img)
    # pcd_input = o3d.t.io.read_point_cloud("debug/dense-reconstruction/dense.ply")
    
    # # Extract point cloud positions and colors
    # points = pcd_input.point['positions'].numpy()
    # colors = pcd_input.point['colors'].numpy()  # Extract colors

    # logger.info(f"================================================")
    # logger.info(f"points.shape: {points.shape}")
    # logger.info(f"================================================\n")  

    # tvec_i = C_l[:3, 3]  # Extract translation vector from camera extrinsics
    # rvec_i = cv2.Rodrigues(C_l[:3, :3])[0]  # Extract rotation vector from rotation matrix

    # camera_matrix = np.array([[1090.536, 0, 954.99],
    #                            [0, 1090.536, 523.12],
    #                            [0, 0, 1]], dtype=np.float32)  # Camera intrinsic parameters

    # # Project points to camera
    # projected_points, _ = cv2.projectPoints(points, rvec=rvec_i, tvec=tvec_i, 
    #                                          cameraMatrix=camera_matrix, distCoeffs=None)
    
    # logger.info(f"================================================")
    # logger.info(f"projected_points.shape: {projected_points.shape}")
    # logger.info(f"================================================\n")

    # # projected_points = project_points_with_cv2(points, camera_matrix, (640, 720))
  
    # projected_points = projected_points.reshape(-1, 2)  # Reshape for OpenCV
    # img = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Create a blank image

    
    # for idx, (point, color) in enumerate(zip(projected_points, colors)):
    #     x, y = int(point[0]), int(point[1])
    #     if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
    #         color_bgr = (int(color[2]), int(color[1]), int(color[0]))
    #         cv2.circle(img, (x, y), 3, color_bgr, -1)
    #         # 
    #         # if idx == 0:
    #             # logger.info(f"================================================")
    #             # logger.info(f"color: {color}")
    #             # logger.info(f"color_bgr: {color_bgr}")
    #             # logger.info(f"================================================\n")
    #         # 
    # cv2.imshow("Projected Points", img)
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    
    # ================================================
    # CASE 3: testing bev_generator.updated_camera_extrinsics()
    # ================================================
    
    # bev_generator = BEVGenerator()
    # ground_id = bev_generator.LABELS["NAVIGABLE_SPACE"]["id"]
    
    # pcd_input = o3d.t.io.read_point_cloud("debug/left-segmented-labelled.ply")
    # pcd_rectified = bev_generator.tilt_rectification(pcd_input)

    # camera_matrix = np.array([[1093.2768, 0, 964.989],
    #                            [0, 1093.2768, 569.276],
    #                            [0, 0, 1]], dtype=np.float32)  # Replace fx, fy, cx, cy with actual camera parameters
    # dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Assuming no lens distortion

    # # Extract point cloud positions
    # points = pcd_input.point['positions'].numpy()

    # # Project points to camera
    # projected_points, _ = cv2.projectPoints(points, rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)), 
    #                                          cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

    # # Convert projected points to 2D
    # projected_points_2d = projected_points.reshape(-1, 2)
    # logger.info(f"Projected Points: {projected_points_2d}")


    # # normal vectors
    # n_i, _ = bev_generator.get_class_plane(pcd_input, ground_id)
    # n_f, _ = bev_generator.get_class_plane(pcd_rectified, ground_id)

    # # axis angles
    # axis_angles_i = bev_generator.axis_angles(n_i)
    # axis_angles_f = bev_generator.axis_angles(n_f)
    
    # logger.info(f"================================================")    
    # logger.info(f"axis_angles_i: {axis_angles_i}")
    # logger.info(f"axis_angles_f: {axis_angles_f}")
    # logger.info(f"================================================\n")


    # # rotation matrix
    # # R_i, _ = cv2.Rodrigues(np.array([axis_angles_i[0], axis_angles_i[1], axis_angles_i[2]]))
    # # R_f, _ = cv2.Rodrigues(np.array([axis_angles_f[0], axis_angles_f[1], axis_angles_f[2]]))
    
    # # yaw, pitch, roll
    # exit()

    # logger.info(f"================================================")
    # logger.info(f"x_i: {x_i:.2f}, y_i: {y_i:.2f}, z_i: {z_i:.2f}")
    # logger.info(f"y_i: {y_i:.2f}, p_i: {p_i:.2f}, r_i: {r_i:.2f}")
    # logger.info(f"================================================\n")

    # # [pitch, yaw, roll] calculations for n2
    # x_f, y_f, z_f = bev_generator.axis_angles(n_f)
    # R_f, _ = cv2.Rodrigues(np.array([x_f, y_f, z_f]))
    # y_f, p_f, r_f = bev_generator.rotation_matrix_to_ypr(R_f)

    # logger.info(f"================================================")
    # logger.info(f"x_f: {x_f:.2f}, y_f: {y_f:.2f}, z_f: {z_f:.2f}")
    # logger.info(f"y_f: {y_f:.2f}, p_f: {p_f:.2f}, r_f: {r_f:.2f}")
    # logger.info(f"================================================\n")

    # exit()
    # T_i = np.eye(4)
    # T_i[:3, :3] = R_i
    # T_i[:3, 3] = [0.0, 0, 1]

    # T_f = np.eye(4)
    # T_f[:3, :3] = R_f
    # T_f[:3, 3] = [0.0, 0, 1]
   
    # T = np.linalg.inv(T_f) @ T_i

    # # logger.info(f"================================================")
    # # logger.info(f"T: {T}")
    # # logger.info(f"================================================\n")

    # R = T[:3, :3]

    # rvec, _ = cv2.Rodrigues(R)
    # p, y, r = rvec.flatten()

    # p, y, r = np.degrees([p, y, r])

    # logger.info(f"================================================")
    # logger.info(f"p: {p:.2f}, y: {y:.2f}, r: {r:.2f}")
    # logger.info(f"================================================\n")

    # ================================================
    # CASE 2: testing CAMERA_PARAMS
    # ================================================

    # camera_params = get_zed_camera_params("debug/front_2024-06-05-09-14-54.svo")
    # logger.info(f"================================================")
    # logger.info(f"camera_params: {camera_params}")
    # logger.info(f"================================================\n")
    # exit()


    # # ================================================
    # # CASE 1: generate segmentation masks
    # # ================================================
    # vis = o3d.visualization.Visualizer()
    # bev_generator = BEVGenerator()
    
    # pcd_input = o3d.t.io.read_point_cloud("debug/left-segmented-labelled.ply")
    # pcd_rectified = bev_generator.tilt_rectification(pcd_input)
    
    # ground_id = bev_generator.LABELS["NAVIGABLE_SPACE"]["id"]
    
    # # ground plane normal => [original / rectified] pointcloud
    # # return [a, b, c, d]
    # n_i, _ = bev_generator.get_class_plane(pcd_input, ground_id)
    # n_f, _ = bev_generator.get_class_plane(pcd_rectified, ground_id)

    # # logger.info(f"================================================")
    # # logger.info(f"n_i.shape: {n_i.shape}")
    # # logger.info(f"n_f.shape: {n_f.shape}")
    # # logger.info(f"================================================\n")
    
    # # pitch, yaw, roll  => [original / rectified]
    # p_i, y_i, r_i = bev_generator.axis_angles(n_i)
    # p_f, y_f, r_f = bev_generator.axis_angles(n_f)

    # logger.info(f"================================================")
    # logger.info(f"[BEFORE RECTIFICATION] - Yaw: {y_i:.2f}, Pitch: {p_i:.2f}, Roll: {r_i:.2f}")
    # logger.info(f"[AFTER RECTIFICATION] - Yaw: {y_f:.2f}, Pitch: {p_f:.2f}, Roll: {r_f:.2f}")
    # logger.info(f"================================================\n")

    # # generate BEV
    # bev_pcd = bev_generator.generate_BEV(pcd_input)
    
    # logger.info(f"================================================")
    # logger.info(f"Number of points in bev_pcd: {len(bev_pcd.point['positions'].numpy())}")
    # logger.info(f"================================================\n")
    
    # # cropping params
    # crop_bb = {'x_min': -5, 'x_max': 5, 'z_min': 0, 'z_max': 10}
    
    # bev_pcd_cropped = crop_pcd(bev_pcd, crop_bb)

    # x_values = bev_pcd_cropped.point['positions'][:, 0].numpy()
    # y_values = bev_pcd_cropped.point['positions'][:, 1].numpy()
    # z_values = bev_pcd_cropped.point['positions'][:, 2].numpy()
    
    # logger.info(f"================================================")
    # logger.info(f"Range of x values: {x_values.min()} to {x_values.max()}")
    # logger.info(f"Range of y values: {y_values.min()} to {y_values.max()}")
    # logger.info(f"Range of z values: {z_values.min()} to {z_values.max()}")
    # logger.info(f"================================================\n")
    
    # seg_mask_mono = bev_generator.bev_to_seg_mask_mono(bev_pcd_cropped, 
    #                                                             nx = 400, nz = 400, 
    #                                                             bb = crop_bb)
    # seg_mask_rgb = mono_to_rgb_mask(seg_mask_mono)
    # cv2.imshow("seg_mask_rgb", seg_mask_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # logger.info(f"================================================")
    # logger.info(f"seg_mask_rgb.shape: {seg_mask_rgb.shape}")
    # logger.info(f"================================================\n")

    # output_path = "debug/seg-mask-rgb.png"
    
    # cv2.imwrite(output_path, seg_mask_rgb)

    # logger.info(f"================================================")
    # logger.info(f"Segmentation mask saved to {output_path}")
    # logger.info(f"================================================\n")

    # ================================================
    # visualization
    # ================================================
    # vis.create_window()
        
    # # Co-ordinate frame for vis window      
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame)
    
    # # Adding point clouds to visualizer
    # # vis.add_geometry(combined_pcd.to_legacy())
    # vis.add_geometry(bev_pcd_cropped.to_legacy())
    
    # view_ctr = vis.get_view_control()
    # view_ctr.set_front(np.array([0, -1, 0]))
    # view_ctr.set_up(np.array([0, 0, 1]))
    # # view_ctr.set_zoom(0.9)
    # view_ctr.set_zoom(4)
    
    # vis.run()
    # vis.destroy_window()

    # ================================================
    # CASE 0: testing bev_generator.pcd_to_seg_mask_mono()
    # ================================================

    # src_folder = "train-data"
    # dst_folder = "debug/output-seg-masks"
    # bev_generator = BEVGenerator()

    # crop_bb = {'x_min': -5, 'x_max': 5, 'z_min': 0, 'z_max': 10}
    # nx = 400
    # nz = 400

    # if not os.path.exists(dst_folder):
    #     os.makedirs(dst_folder)

    # left_segmented_labelled_files = []

    # total_files = sum(len(files) for _, _, files in os.walk(src_folder) if 'left-segmented-labelled.ply' in files)
    # with tqdm(total=total_files, desc="Processing files", ncols=100) as pbar:
    #     for root, dirs, files in os.walk(src_folder):
    #         for file in files:
    #             if file == 'left-segmented-labelled.ply':
    #                 file_path = os.path.join(root, file)
    #                 left_segmented_labelled_files.append(file_path)

    #                 try:
    #                     pcd_input = o3d.t.io.read_point_cloud(file_path)
    #                     seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask(pcd_input, 
    #                                                                                  nx=nx, nz=nz, 
    #                                                                                  bb=crop_bb)

    #                     output_rgb_path = os.path.join(dst_folder, f"seg-mask-rgb-{os.path.basename(root)}.png")
    #                     cv2.imwrite(output_rgb_path, seg_mask_rgb)
    #                 except Exception as e:
    #                     logger.error(f"Error processing {file_path}: {e}")
    
    #                 pbar.update(1)
