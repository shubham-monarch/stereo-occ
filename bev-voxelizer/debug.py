#! /usr/bin/env python3

import open3d as o3d
import matplotlib.pyplot as plt

from bev_generator import BEVGenerator
from logger import get_logger
import numpy as np
from helpers import visualize_pcd

logger = get_logger("debug")

if __name__ == "__main__":  

    # ================================================
    # CASE 1: Crop point clouds and save them to disk
    # ================================================
    # src_pcd_path = "debug/left-segmented-labelled.ply"
    # dst_pcd_path = "debug/cropped_pointcloud.ply"
    
    # bev_voxelizer = BevVoxelizer()
    
    # src_pcd = o3d.t.io.read_point_cloud(src_pcd_path)
    # src_pcd = bev_voxelizer.tilt_rectification(src_pcd)
    
    # crop_pointcloud(src_pcd, dst_pcd_path)
   
    
    # ================================================
    # CASE 2: generate segmentation masks
    # ================================================
    bev_generator = BEVGenerator()
    
    pcd_input = o3d.t.io.read_point_cloud("debug/left-segmented-labelled.ply")
    pcd_rectified = bev_generator.tilt_rectification(pcd_input)
    
    ground_id = bev_generator.LABELS["NAVIGABLE_SPACE"]["id"]
    
    # ground plane normal => [original / rectified] pointcloud
    # return [a, b, c, d]
    n_i, _ = bev_generator.get_class_plane(pcd_input, ground_id)
    n_f, _ = bev_generator.get_class_plane(pcd_rectified, ground_id)

    # logger.info(f"================================================")
    # logger.info(f"n_i.shape: {n_i.shape}")
    # logger.info(f"n_f.shape: {n_f.shape}")
    # logger.info(f"================================================\n")
    
    # pitch, yaw, roll  => [original / rectified]
    p_i, y_i, r_i = bev_generator.axis_angles(n_i)
    p_f, y_f, r_f = bev_generator.axis_angles(n_f)

    logger.info(f"================================================")
    logger.info(f"[BEFORE RECTIFICATION] - Yaw: {y_i:.2f}, Pitch: {p_i:.2f}, Roll: {r_i:.2f}")
    logger.info(f"[AFTER RECTIFICATION] - Yaw: {y_f:.2f}, Pitch: {p_f:.2f}, Roll: {r_f:.2f}")
    logger.info(f"================================================\n")

    # generate BEV
    bev_pcd = bev_generator.generate_BEV(pcd_rectified)
    # visualize_pcd(bev_pcd)

    # crop BEV
    valid_indices = np.where(
        # # y <= 0
        # (pcd_rectified.point['positions'][:, 1].numpy() <= 0) & 
        
        # x between -3 and 3
        (pcd_rectified.point['positions'][:, 0].numpy() >= -3) & 
        (pcd_rectified.point['positions'][:, 0].numpy() <= 3) & 
        
        # z between 0 and 15
        (pcd_rectified.point['positions'][:, 2].numpy() >= 0) & 
        (pcd_rectified.point['positions'][:, 2].numpy() <= 15)
    )[0]

    logger.info(f"================================================")
    logger.info(f"len(valid_indices): {len(valid_indices)}")
    logger.info(f"================================================\n")
    
    bev_pcd_cropped = bev_pcd.select_by_index(valid_indices)        
    visualize_pcd(bev_pcd_cropped)
    
    # mask_mono = pcd_to_segmentation_mask_mono(bev_pcd)