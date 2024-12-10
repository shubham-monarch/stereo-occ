#! /usr/bin/env python3

import numpy as np
import boto3
import os
import random
from tqdm import tqdm
import shutil

from logger import get_logger



class DataGenerator:
    '''Class to generate training data for the SBEVNet model'''
    
    def __init__(self):
        self.logger = get_logger("data-generator")


    def copy_images(self, src_dir, dst_dir):
         # Create model data directory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)
        
        # Walk through s3 data directory
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file in ['left.jpg', 'right.jpg']:
                    # Get relative path from s3_data_dir
                    rel_path = os.path.relpath(root, src_dir)
                    
                    # Create corresponding directory in model_data_dir
                    dest_dir = os.path.join(dst_dir, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Copy file
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)


    def s3_data_to_model_data(self, s3_data_dir = None, model_data_dir = None):
        '''Copy left and right images from s3 data to model data directory'''
        
        assert s3_data_dir is not None, "s3_data_dir is required"
        assert model_data_dir is not None, "model_data_dir is required"

        self.copy_images(s3_data_dir, model_data_dir)    
    


        

    def fetch_data_from_s3(self, s3_uri, local_dir):
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
    
    # s3_uri = "s3://occupancy-dataset/occ-dataset/vineyards/RJM/"
    # data_generator.fetch_data_from_s3(s3_uri, "aws-data")

    data_generator.s3_data_to_model_data("aws-data", "model-data")