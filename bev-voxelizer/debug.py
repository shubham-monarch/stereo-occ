#! /usr/bin/env python3

import open3d as o3d
import cv2

from bev_generator import BEVGenerator
from logger import get_logger
import numpy as np
from helpers import mono_to_rgb_mask, crop_pcd

logger = get_logger("debug")

if __name__ == "__main__":  

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
    # CASE 2: generate segmentation masks
    # ================================================
    pcd_input = o3d.t.io.read_point_cloud("debug/left-segmented-labelled.ply")
    bev_generator = BEVGenerator()
    
    crop_bb = {'x_min': -5, 'x_max': 5, 'z_min': 0, 'z_max': 10}
    
    seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask_mono(pcd_input, 
                                                                     nx = 400, nz = 400, 
                                                                     bb = crop_bb)

    cv2.imshow("seg_mask_rgb", seg_mask_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()