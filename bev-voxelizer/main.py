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

# custom modules
from utils import io_utils
from bev_voxelizer import BevVoxelizer

# TO-DO
# - priority based collapsing
# - crop pointcloud to bounding boxs
# - hidden point removal 
# - farthest point sampling
# - checkout bev-former voxelizer
# - statistical outlier removal
# - refactor compute_tilt_matrix()
# - make project_to_ground_plane more robust

# LOGGING SETUP
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
coloredlogs.install(level='INFO', logger=logger, force=True)



def bev_images_from_foler(src_folder: str, bev_images_folder: str):
    '''
    Generate BEV images for all pointclouds in the source folder
    '''
    os.makedirs(bev_images_folder, exist_ok=True)          
    vis = o3d.visualization.Visualizer()
  
    for file_name in tqdm(np.random.permutation(os.listdir(src_folder)), desc="Processing files"):
        if "_treasury" in file_name and file_name.endswith(".ply"):
            try:
                logger.warning(f"=================================")        
                logger.warning(f"Processing {file_name}")
                logger.warning(f"=================================\n")
                
                src_path = os.path.join(src_folder, file_name)
                pcd_input = o3d.t.io.read_point_cloud(src_path)
                
                bev_voxelizer = BevVoxelizer()
                combined_pcd = bev_voxelizer.generate_bev_voxels(pcd_input)
                
                avg_x = combined_pcd.point['positions'][:, 0].mean().item()
                avg_y = combined_pcd.point['positions'][:, 1].mean().item()
                avg_z = combined_pcd.point['positions'][:, 2].mean().item()
                logger.info(f"Average x: {avg_x}, Average y: {avg_y}, Average z: {avg_z}")

                combined_pcd.point['positions'][:, 2] += 350
                # combined_pcd.point['positions'][:, 0] -= 500
                # Save the combined point cloud
                # combined_pcd_path = os.path.join(bev_pcd_folder, file_name)
                # o3d.t.io.write_point_cloud(combined_pcd_path, combined_pcd)
                
                vis.create_window()
                
                # Co-ordinate frame for vis window    
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
                vis.add_geometry(coordinate_frame)
                
                # Adding point clouds to visualizer
                vis.add_geometry(combined_pcd.to_legacy())
                
                view_ctr = vis.get_view_control()
                view_ctr.set_front(np.array([0, -1, 0]))
                view_ctr.set_up(np.array([0, 0, 1]))
                view_ctr.set_zoom(0.1)
                
                # Capture the screen and save it as an image
                vis.poll_events()
                vis.update_renderer()
                image_path = os.path.join(bev_images_folder, file_name.replace(".ply", ".png"))
                vis.capture_screen_image(image_path)
                time.sleep(3)  # Wait for 1 second
                vis.destroy_window()
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
            finally:
                vis.destroy_window()
            # break


# if __name__ == "__main__":
    
#     src_folder = "pcd-files/vineyards/"
#     # bev_images_folder = "bev-images-test"
    
    
#     bev_samples_folder = "bev-demo/bev-samples"
#     bev_files = os.listdir(bev_samples_folder)
#     bev_files_ply = [file.replace(".png", ".ply") for file in bev_files]
#     print(bev_files_ply)

#     file_name = "vineyards_gallo_216.ply"
    
#     # os.makedirs(bev_images_folder, exist_ok=True)          
#     vis = o3d.visualization.Visualizer()
  
#     src_path = os.path.join(src_folder, file_name)
#     pcd_input = o3d.t.io.read_point_cloud(src_path)
    
#     bev_voxelizer = BevVoxelizer()
#     combined_pcd = bev_voxelizer.generate_bev_voxels(pcd_input)
    
    
#     vis.create_window()
    
#     # Co-ordinate frame for vis window    
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4, origin=[0, 0, 0])
#     vis.add_geometry(coordinate_frame)
    
#     # Adding point clouds to visualizer
#     vis.add_geometry(combined_pcd.to_legacy())
    
#     view_ctr = vis.get_view_control()
#     view_ctr.set_front(np.array([0, -1, 0]))
#     view_ctr.set_up(np.array([0, 0, 1]))
#     view_ctr.set_zoom(0.5)
    
#     vis.run()
#     vis.destroy_window()



if __name__ == "__main__":
    
    src_folder = "pcd-files/vineyards/"
    # bev_images_folder = "bev-images-test"
    
    vis = o3d.visualization.Visualizer()
    
    bev_samples_folder = "bev-demo/bev-samples"
    bev_files = os.listdir(bev_samples_folder)
    
    for file in bev_files:
        try:
            file_name = file.replace(".png", ".ply")
            logger.warning(f"=================================")        
            logger.warning(f"Processing {file_name}")
            logger.warning(f"=================================\n")
                    
            src_path = os.path.join(src_folder, file_name)
            pcd_input = o3d.t.io.read_point_cloud(src_path)
            
            bev_voxelizer = BevVoxelizer()
            combined_pcd = bev_voxelizer.generate_bev_voxels(pcd_input)
            
            
            vis.create_window()
            
            # Co-ordinate frame for vis window    
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)
            
            # Adding point clouds to visualizer
            vis.add_geometry(combined_pcd.to_legacy())
            
            view_ctr = vis.get_view_control()
            view_ctr.set_front(np.array([0, -1, 0]))
            view_ctr.set_up(np.array([0, 0, 1]))
            view_ctr.set_zoom(0.5)
            
            vis.run()
            vis.destroy_window()
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
        # break
