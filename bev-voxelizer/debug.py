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
    # CASE 1: generate segmentation masks
    # ================================================
    vis = o3d.visualization.Visualizer()
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
    bev_pcd = bev_generator.generate_BEV(pcd_input)
    
    logger.info(f"================================================")
    logger.info(f"Number of points in bev_pcd: {len(bev_pcd.point['positions'].numpy())}")
    logger.info(f"================================================\n")
    
    # visualize_pcd(bev_pcd)

    valid_indices = np.where(

        # x between -3 and 3
        (bev_pcd.point['positions'][:, 0].numpy() >= -3) & 
        (bev_pcd.point['positions'][:, 0].numpy() <= 3) & 

        # z between 0 and 15
        (bev_pcd.point['positions'][:, 2].numpy() >= 0) & 
        (bev_pcd.point['positions'][:, 2].numpy() <= 15)

    )[0]

    logger.info(f"================================================")
    logger.info(f"len(valid_indices): {len(valid_indices)}")
    logger.info(f"================================================\n")
    
    bev_pcd_cropped = bev_pcd.select_by_index(valid_indices)        
    



    # visualization
    vis.create_window()
        
    # Co-ordinate frame for vis window      
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    
    # Adding point clouds to visualizer
    # vis.add_geometry(combined_pcd.to_legacy())
    vis.add_geometry(bev_pcd_cropped.to_legacy())
    
    view_ctr = vis.get_view_control()
    view_ctr.set_front(np.array([0, -1, 0]))
    view_ctr.set_up(np.array([0, 0, 1]))
    # view_ctr.set_zoom(0.9)
    view_ctr.set_zoom(4)
    
    vis.run()
    vis.destroy_window()

