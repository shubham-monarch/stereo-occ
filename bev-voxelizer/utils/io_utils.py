#! /usr/bin/env python3

import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import shutil
import time
import math
from typing import List
import random
import json
import sys
from pathlib import Path
from tqdm import tqdm
import coloredlogs


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, force=True)

def collect_s3_pointclouds(
    source_folder: Path,
    dest_folder: Path = None
) -> str:
    
    '''
    aws utils to collect pointclouds from s3
    '''
    
    base_folder = os.path.dirname(source_folder)
    src_folder_name = os.path.basename(source_folder)

    if dest_folder is None: 
        dest_folder = os.path.join(base_folder, f"segmented-{src_folder_name}")

    create_folders([dest_folder])
    
    # Walk through all directories and subdirectories
    total_items = sum([len(files) for _, _, files in os.walk(source_folder)])
    for i, (root, dirs, files) in enumerate(tqdm(os.walk(source_folder), total=total_items, desc="Processing files")):
        if "left-segmented-labelled.ply" in files:
            source_file = os.path.join(root, "left-segmented-labelled.ply")
            
            if os.path.exists(source_file):
                # Create a unique filename based on the subfolder structure
                destination_file = os.path.join(dest_folder, f"{i}.ply")
                shutil.copy(source_file, destination_file)
            
            else:
                logger.warning(f"Source file not found: {source_file}")
                continue
            
    return dest_folder  

def delete_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logging.error(f"Error {e} while deleting {file_path}")
        else:
            logging.info(f"The file {file_path} does not exist.")

def delete_folders(folders):
	for folder_path in folders:
			# logging.debug(f"Deleting the old files in {folder_path}")
			if os.path.exists(folder_path):
				try: 
					shutil.rmtree(folder_path)
				except Exception as e :
					logging.error(f"Error {e} while deleting {folder_path}")
					# time.sleep(1)  # wait for 1 second before retrying
			
def create_folders(folders):
	for path in folders:
		os.makedirs(path, exist_ok=True)
		# logging.warning(f"Created the {path} folder!")
            



# def bev_images_from_folder(src_folder: str, bev_images_folder: str):
#     '''
#     Generate BEV images for all pointclouds in the source folder
#     '''
#     os.makedirs(bev_images_folder, exist_ok=True)          
#     vis = o3d.visualization.Visualizer()
  
#     for file_name in tqdm(np.random.permutation(os.listdir(src_folder)), desc="Processing files"):
#         if "_treasury" in file_name and file_name.endswith(".ply"):
#             try:
#                 logger.warning(f"=================================")        
#                 logger.warning(f"Processing {file_name}")
#                 logger.warning(f"=================================\n")
                
#                 src_path = os.path.join(src_folder, file_name)
#                 pcd_input = o3d.t.io.read_point_cloud(src_path)
                
#                 bev_voxelizer = BevVoxelizer()
#                 combined_pcd = bev_voxelizer.generate_bev_voxels(pcd_input)
                
#                 avg_x = combined_pcd.point['positions'][:, 0].mean().item()
#                 avg_y = combined_pcd.point['positions'][:, 1].mean().item()
#                 avg_z = combined_pcd.point['positions'][:, 2].mean().item()
#                 logger.info(f"Average x: {avg_x}, Average y: {avg_y}, Average z: {avg_z}")

#                 combined_pcd.point['positions'][:, 2] += 350
#                 # combined_pcd.point['positions'][:, 0] -= 500
#                 # Save the combined point cloud
#                 # combined_pcd_path = os.path.join(bev_pcd_folder, file_name)
#                 # o3d.t.io.write_point_cloud(combined_pcd_path, combined_pcd)
                
#                 vis.create_window()
                
#                 # Co-ordinate frame for vis window    
#                 coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
#                 vis.add_geometry(coordinate_frame)
                
#                 # Adding point clouds to visualizer
#                 vis.add_geometry(combined_pcd.to_legacy())
                
#                 view_ctr = vis.get_view_control()
#                 view_ctr.set_front(np.array([0, -1, 0]))
#                 view_ctr.set_up(np.array([0, 0, 1]))
#                 view_ctr.set_zoom(0.1)
                
#                 # Capture the screen and save it as an image
#                 vis.poll_events()
#                 vis.update_renderer()
#                 image_path = os.path.join(bev_images_folder, file_name.replace(".ply", ".png"))
#                 vis.capture_screen_image(image_path)
#                 time.sleep(3)  # Wait for 1 second
#                 vis.destroy_window()
#             except Exception as e:
#                 logger.error(f"Error processing {file_name}: {e}")
#             finally:
#                 vis.destroy_window()
#             # break