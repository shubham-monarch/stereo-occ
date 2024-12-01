#! /usr/bin/env python3

import open3d as o3d
import os
import random
import numpy as np
import cv2
from bev_voxelizer import BevVoxelizer
from utils.log_utils import get_logger

def list_base_folders(folder_path):
    base_folders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            base_folders.append(os.path.join(root, dir_name))
    return base_folders

def pcd_to_bev(
    pcd_path: str,  
    H: int = 480, 
    W: int = 640
) -> np.ndarray:

    '''Convert a 3D PCD => BEV => (H, W, 3) numpy array'''

    pcd_3D = o3d.t.io.read_point_cloud(pcd_path)
    
    bev_voxelizer = BevVoxelizer()
    pcd_BEV = bev_voxelizer.generate_bev_voxels(pcd_3D)

    x_coords = pcd_BEV.point['positions'][:, 0].numpy()
    z_coords = pcd_BEV.point['positions'][:, 2].numpy()
    colors = pcd_BEV.point['colors'].numpy()  # Assuming colors are in [0, 1] range

    # cropping bev points to a 20m x 10m area
    valid_indices = np.where(
        (x_coords >= -10) & (x_coords <= 10) & 
        (z_coords >= 0) & (z_coords <= 20)
    )[0]

    x_coords = x_coords[valid_indices]
    z_coords = z_coords[valid_indices]
    colors = colors[valid_indices]

    x_min, x_max = x_coords.min(), x_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()

    x_scaled = ((x_coords - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
    z_scaled = ((z_coords - z_min) / (z_max - z_min) * (H - 1)).astype(np.int32)

    # rgb to bgr
    colors_bgr = colors[:, [2, 1, 0]]

    # Create a blank color image
    image_BEV = np.zeros((H, W, 3), dtype=np.uint8)

    # Plot the points on the image with their original colors
    for x, z, color in zip(x_scaled, z_scaled, colors_bgr):
        image_BEV[H - z - 1, x] = color  # Invert z to match image coordinates

    return image_BEV


if __name__ == "__main__":
    logger = get_logger()

    segmented_pcd_dir = "/home/skumar/ssd/2024_06_06_utc"
    segmented_bev_dir = "train-data"
    
    segmented_pcd_folders = list_base_folders(segmented_pcd_dir)
    
    random_folder = random.choice(segmented_pcd_folders)
    # random_folder = "/home/skumar/ssd/2024_06_06_utc/svo_files/front_2024-06-05-09-43-13.svo/246_to_388/frame-378" 
    
    logger.info(f"=================================")
    logger.info(f"Processing {random_folder}")
    logger.info(f"=================================\n")
    
    random_pcd_file = os.path.join(random_folder, "left-segmented-labelled.ply")

    bev_image = pcd_to_bev(random_pcd_file)
    
    output_path = os.path.join(random_folder, "bev_image.png")
    cv2.imwrite(output_path, bev_image)
    
    logger.info(f"Saved BEV image to {output_path}")

    # cv2.imshow("BEV Image", bev_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite("bev_image.png", bev_image)


   