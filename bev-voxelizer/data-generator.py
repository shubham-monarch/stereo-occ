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
 

    def raw_data_to_model_data(self, raw_data_dir = None, model_data_dir = None):
        '''Process raw data and move to model-data folder'''

        assert raw_data_dir is not None, "raw_data_dir is required"
        assert model_data_dir is not None, "model_data_dir is required"        
        assert not (os.path.exists(model_data_dir) and os.listdir(model_data_dir)), "model_data_dir must be empty"


        # Create the target folder if it doesn't exist
        os.makedirs(model_data_dir, exist_ok=True)

        # Define the target subfolders
        left_folder = os.path.join(model_data_dir, 'left')
        right_folder = os.path.join(model_data_dir, 'right')
        seg_masks_mono_folder = os.path.join(model_data_dir, 'seg-masks-mono')
        seg_masks_rgb_folder = os.path.join(model_data_dir, 'seg-masks-rgb')
        cam_extrinsics_folder = os.path.join(model_data_dir, 'cam-extrinsics')


        # Create the target subfolders if they don't exist
        os.makedirs(left_folder, exist_ok=True)
        os.makedirs(right_folder, exist_ok=True)
        os.makedirs(seg_masks_mono_folder, exist_ok=True)
        os.makedirs(seg_masks_rgb_folder, exist_ok=True)
        os.makedirs(cam_extrinsics_folder, exist_ok=True)

        # Count total files for progress bar
        total_files = 0
        for root, dirs, files in os.walk(raw_data_dir):
            for file in files:
                if file.endswith('left.jpg') or \
                   file.endswith('right.jpg') or \
                   file.endswith('-mono.png') or \
                   file.endswith('-rgb.png') or \
                   file.endswith('cam-extrinsics.npy'):
                    total_files += 1
        
        self.logger.info(f"=========================")
        self.logger.info(f"Total files: {total_files}") 
        self.logger.info(f"=========================\n")

        with tqdm(total=total_files, desc="Organizing Images") as pbar:
            for root, dirs, files in os.walk(raw_data_dir):
                # Get folder number from root path
                folder_num = os.path.basename(root)
                if not folder_num.isdigit():
                    continue

                for file in files:
                    if file == 'left.jpg':
                        new_filename = f"{folder_num}__left.jpg"
                        shutil.copy(os.path.join(root, file), os.path.join(left_folder, new_filename))
                        pbar.update(1)
                    elif file == 'right.jpg':
                        new_filename = f"{folder_num}__right.jpg"
                        shutil.copy(os.path.join(root, file), os.path.join(right_folder, new_filename))
                        pbar.update(1)
                    elif file == 'seg_mask_mono.png':
                        new_filename = f"{folder_num}__seg-mask-mono.png"
                        shutil.copy(os.path.join(root, file), os.path.join(seg_masks_mono_folder, new_filename))
                        pbar.update(1)
                    elif file == 'seg_mask_rgb.png':
                        new_filename = f"{folder_num}__seg-mask-rgb.png"
                        shutil.copy(os.path.join(root, file), os.path.join(seg_masks_rgb_folder, new_filename))
                        pbar.update(1)
                    elif file == 'cam_extrinsics.npy':
                        new_filename = f"{folder_num}__cam-extrinsics.npy"
                        shutil.copy(os.path.join(root, file), os.path.join(cam_extrinsics_folder, new_filename))
                        pbar.update(1)


    def s3_data_to_raw_data(self, s3_data_dir = None, raw_data_dir = None):
        '''Process s3 data and move to model-data folder'''

        assert s3_data_dir is not None, "s3_data_dir is required"
        assert raw_data_dir is not None, "raw_data_dir is required"    
        assert not (os.path.exists(raw_data_dir) and os.listdir(raw_data_dir)), "raw_data_dir must be empty"



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
                        dest_dir = os.path.join(raw_data_dir, rel_path)
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
        
        assert not (os.path.exists(local_dir) and os.listdir(local_dir)), f"local_dir must be empty!"

        s3 = boto3.resource('s3')
        bucket_name, prefix = s3_uri.replace("s3://", "").split("/", 1)
        bucket = s3.Bucket(bucket_name)

        # Get all leaf folders and randomly select 200
        leaf_folders = set()
        for obj in bucket.objects.filter(Prefix=prefix):
            leaf_folder = os.path.dirname(obj.key)
            if leaf_folder != prefix:
                leaf_folders.add(leaf_folder)
        leaf_folders = set(random.sample(list(leaf_folders), min(200, len(leaf_folders))))

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
    
    # s3_uri = "s3://occupancy-dataset/occ-dataset/vineyards/RJM/"

    # fetch data from s3
    # data_generator.fetch_data_from_s3(s3_uri, "aws-data")

    # process s3 data and move to model-data folder
    # data_generator.s3_data_to_raw_data("aws-data", "model-data")

    # process raw data and move to model-data folder
    data_generator.raw_data_to_model_data("data/raw-data", "model-data")
    
