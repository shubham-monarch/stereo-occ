#! /usr/bin/env python3
import open3d as o3d  
import open3d.core as o3c
import shutil
import os
import logging, coloredlogs
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np  
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import cv2

# custom modules
# from utils.log_utils import get_logger
# from bev_voxelizer import BevVoxelizer
# from utils.data_generator import count_unique_labels, pcd_to_segmentation_mask_mono, mono_to_rgb_mask, get_label_colors_from_yaml
from logger import get_logger
from bev_generator import BEVGenerator

if __name__ == "__main__":
    
    # src_folder = "pcd-files/vineyards/RJM"
    
    vis = o3d.visualization.Visualizer()
    logger = get_logger("main")
    
    
    # segmented_pcd_folder = "bev-voxelizer/train-data/0/left-segmented-labelled.ply"
    # segmented_pcd_folder_files = os.listdir(segmented_pcd_folder)
    
    # random.seed(0)
    # file = random.choice(segmented_pcd_folder_files)
    # file  = "vineyards_RJM_15.ply"
    # file = "vineyards_RJM_38.ply"

    
    # file =  "train-data/0/left-segmented-labelled.ply"
    file = "debug/left-segmented-labelled.ply"

    # logger.warning(f"=================================")        
    # logger.warning(f"Processing {file}")
    # logger.warning(f"=================================\n")
    
    # # pcd_input = o3d.t.io.read_point_cloud(os.path.join(segmented_pcd_folder, file))
    # pcd_input = o3d.t.io.read_point_cloud(file)
    
    # bev_voxelizer = BevVoxelizer()
    # combined_pcd = bev_voxelizer.generate_bev_voxels(pcd_input)

    # bev_array = pcd_to_segmentation_mask_mono(combined_pcd)

    # # Save BEV array as image
    # output_path = "seg-mask-mono.png"
    # cv2.imwrite(output_path, bev_array)
    # logger.info(f"Saved BEV image to {output_path}")


    # num_unique_labels, unique_labels = count_unique_labels(bev_array)
    # logger.info(f"=================================")
    # logger.info(f"Number of unique labels: {num_unique_labels}")
    # logger.info(f"Unique labels: {unique_labels}")
    # logger.info(f"=================================\n")

  
    # # Load colors from yaml file
    # label_colors_bgr, label_colors_rgb = get_label_colors_from_yaml("Mavis.yaml")

    # logger.info(f"=================================")
    # logger.info(f"label_colors_bgr: {label_colors_bgr}")
    # logger.info(f"=================================\n")

    # seg_mask_rgb = mono_to_rgb_mask(bev_array, label_colors_bgr)

    # output_path = "seg-mask-rgb.png"
    # cv2.imwrite(output_path, seg_mask_rgb)
    # logger.info(f"Saved BEV image to {output_path}")

    # cv2.imshow("Segmentation Mask RGB", seg_mask_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    try:
        logger.warning(f"=================================")        
        logger.warning(f"Processing {file}")
        logger.warning(f"=================================\n")
                
        # pcd_input = o3d.t.io.read_point_cloud(os.path.join(segmented_pcd_folder, file))
        pcd_input = o3d.t.io.read_point_cloud(file)
        
        bev_generator = BEVGenerator()
        combined_pcd = bev_generator.generate_BEV(pcd_input)

        logger.info(f"=================================")
        logger.info(f"Shape of points in combined_pcd: {combined_pcd.point['positions'].shape}")
        logger.info(f"=================================\n")

        valid_indices = np.where(
            
            # x between -3 and 3
            (combined_pcd.point['positions'][:, 0].numpy() >= -3) & 
            (combined_pcd.point['positions'][:, 0].numpy() <= 3) & 
            
            # z between 0 and 15
            (combined_pcd.point['positions'][:, 2].numpy() >= 0) & 
            (combined_pcd.point['positions'][:, 2].numpy() <= 15)
        
        )[0]

        combined_pcd_cropped = combined_pcd.select_by_index(valid_indices)

        # output_file_path = os.path.join(segmented_pcd_folder, "combined_output.ply")
        # o3d.t.io.write_point_cloud("combined_output.ply", combined_pcd)
        # logger.info(f"Saved combined point cloud to combined_output.ply")
        
        
        vis.create_window()
        
        # Co-ordinate frame for vis window      
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
        # Adding point clouds to visualizer
        # vis.add_geometry(combined_pcd.to_legacy())
        vis.add_geometry(combined_pcd_cropped.to_legacy())
        
        view_ctr = vis.get_view_control()
        view_ctr.set_front(np.array([0, -1, 0]))
        view_ctr.set_up(np.array([0, 0, 1]))
        # view_ctr.set_zoom(0.9)
        view_ctr.set_zoom(4)
        
        vis.run()
        vis.destroy_window()
    except Exception as e:
        logger.error(f"Error processing {file}: {e}")

