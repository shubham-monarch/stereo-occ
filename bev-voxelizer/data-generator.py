#! /usr/bin/env python3

import numpy as np
import boto3
import os
import random
from tqdm import tqdm
import shutil
import open3d as o3d
import cv2

from logger import get_logger
from bev_generator import BEVGenerator



class DataGenerator:
    '''Class to generate training data for the SBEVNet model'''
    
    def __init__(self):
        self.logger = get_logger("data-generator")
 

    def s3_data_to_model_data(self, s3_data_dir = None, model_data_dir = None):
        
        assert s3_data_dir is not None, "s3_data_dir is required"
        assert model_data_dir is not None, "model_data_dir is required"

        # walk through s3 data directory
        for root, dirs, files in os.walk(s3_data_dir):
            for file in files:
                if file == 'left-segmented-labelled.ply':
                    try:
                        
                        file_path = os.path.join(root, file)
                        pcd_input = o3d.t.io.read_point_cloud(file_path)
                    
                        bev_generator = BEVGenerator()
                    
                        # mask dimensions (400 * 400)
                        nx , ny = 400, 400
                        
                        # crop bounding box
                        crop_bb = {'x_min': -5, 'x_max': 5, 'z_min': 0, 'z_max': 10}

                        # mono / rgb segmentation masks
                        seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask(pcd_input,
                                                                                    nx, ny,
                                                                                    crop_bb)
                        
                        # camera extrinsics
                        camera_extrinsics = bev_generator.get_updated_camera_extrinsics(pcd_input)

                        rel_path = os.path.relpath(root, s3_data_dir)
                        dest_dir = os.path.join(model_data_dir, rel_path)
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        # ================================================
                        # copy left and right images
                        # ================================================
                        for img_file in ['left.jpg', 'right.jpg']:
                            try:
                                src_file = os.path.join(root, img_file)
                                if os.path.exists(src_file):
                                    dest_file = os.path.join(dest_dir, img_file)
                                    shutil.copy2(src_file, dest_file)
                            except (IOError, OSError) as e:
                                self.logger.error(f"Failed to copy {img_file}: {str(e)}")

                        # ================================================
                        # save segmentation masks
                        # ================================================
                        try:
                            cv2.imwrite(os.path.join(dest_dir, 'seg_mask_mono.png'), seg_mask_mono)
                            cv2.imwrite(os.path.join(dest_dir, 'seg_mask_rgb.png'), seg_mask_rgb)
                        except Exception as e:
                            self.logger.error(f"Failed to save segmentation masks: {str(e)}")

                        # ================================================
                        # save camera extrinsics
                        # ================================================
                        try:
                            np.save(os.path.join(dest_dir, 'cam_extrinsics.npy'), camera_extrinsics)
                        except Exception as e:
                            self.logger.error(f"Failed to save camera extrinsics: {str(e)}")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to process {file}: {str(e)}")
                        continue

    


    def fetch_data_from_s3(self, s3_uri, local_dir):
        '''Fetch data from s3 and save to local directory'''
        
        s3 = boto3.resource('s3')
        bucket_name, prefix = s3_uri.replace("s3://", "").split("/", 1)
        bucket = s3.Bucket(bucket_name)

        # Get all leaf folders
        leaf_folders = set()
        for obj in bucket.objects.filter(Prefix=prefix):
            leaf_folder = os.path.dirname(obj.key)
            if leaf_folder != prefix:
                leaf_folders.add(leaf_folder)

        self.logger.info(f"=========================")
        self.logger.info(f"Found {len(leaf_folders)} leaf folders")
        self.logger.info(f"=========================\n")
        
        # Sort leaf folders and create numbered local folders
        selected_folders = sorted(leaf_folders)

        # Initialize progress bars
        pbar_folders = tqdm(enumerate(selected_folders, 1), desc='Processing folders', unit='folder')

        for folder_num, folder in pbar_folders:
            # Create numbered local folder
            numbered_folder = os.path.join(local_dir, str(folder_num))
            os.makedirs(numbered_folder, exist_ok=True)
            
            # Get all objects in the selected folder
            objects = list(bucket.objects.filter(Prefix=folder))
            
            # Download each object
            for obj in objects:
                local_file_path = os.path.join(numbered_folder, os.path.basename(obj.key))
                bucket.download_file(obj.key, local_file_path)

        pbar_folders.close()

if __name__ == "__main__":
    
    data_generator = DataGenerator()
    
    s3_uri = "s3://occupancy-dataset/occ-dataset/vineyards/RJM/"

    # fetch data from s3
    data_generator.fetch_data_from_s3(s3_uri, "aws-data")

    # process s3 data and move to model-data folder
    data_generator.s3_data_to_model_data("aws-data", "model-data")