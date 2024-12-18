#! /usr/bin/env python3

import boto3
import os
import random
from logger import get_logger
from typing import List
from tqdm import tqdm
import open3d as o3d
import numpy as np
import cv2

from bev_generator import BEVGenerator

class LeafFolder:
    def __init__(self, src_URI: str, dest_URI: str):
        '''
        :param src_URI: occ-dataset S3 URI
        :param dest_URI: bev-dataset S3 URI
        '''
        self.logger = get_logger("LeafFolder")
        self.src_URI = src_URI
        self.dest_URI = dest_URI
        self.s3 = boto3.client('s3')

        
        
        self.bev_generator = BEVGenerator()
    
    def process_folder(self):
        
        # ==================
        # 1. download left-segmented-labelled.ply
        # ==================
        pcd_path = self.download_segmented_pcd(self.src_URI)

        # ==================
        # 2. generate mono / RGB segmentation masks
        # ==================
        pcd = o3d.t.io.read_point_cloud(pcd_path)
        
        # mask dimensions
        nx, nz = 256, 256

        # z is depth, x is horizontal
        crop_bb = {'x_min': -2.5, 'x_max': 2.5, 'z_min': 0.0, 'z_max': 5}        
        
        seg_mask_mono, seg_mask_rgb = self.bev_generator.pcd_to_seg_mask(pcd,
                                                                         nx=256,nz=256,
                                                                         bb=crop_bb)
        
        # ==================
        # 3. upload mono / RGB segmentation masks
        # ==================
        self.upload_mask(seg_mask_mono, self.dest_URI + "seg-mask-mono.png")
        self.upload_mask(seg_mask_rgb, self.dest_URI + "seg-mask-rgb.png")
        
        
        # 2. copy imgL
        # 3. copy imgR
        # 4. upload seg_mono
        # 5. upload seg_rgb
        # 6. upload ipm_fea
        # 6. upload ipm_rgb



        pass

        
    def rescale_img(self, img_uri: str):
        pass


    def copy_imgL(self, imgL_uri: str):
        pass

    def copy_imgR(self, imgR_uri: str):
        pass

    def upload_png(self, seg_mask: np.ndarray):
        pass

    def upload_ipm_fea(self, ipm_fea_uri: str):
        pass

    def upload_ipm_rgb(self, ipm_seg_uri: str):
        pass

    def upload_file(self, src_path: str, dest_URI: str) -> bool:
        ''' Upload a file from src_path to dest_URI'''      
        
        try:
            bucket_name, key = dest_URI.replace("s3://", "").split("/", 1)
            self.s3.upload_file(src_path, bucket_name, key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to upload file {src_path} to {dest_URI}: {str(e)}")
            return False

    def download_file(self, src_URI: str, dest_folder: str) -> str:
        ''' Download a file from S3 to the dest_folder'''
        
        try:
            bucket_name, key = src_URI.replace("s3://", "").split("/", 1)
            file_name = key.split("/")[-1]
            
            os.makedirs(dest_folder, exist_ok=True)
            tmp_path = os.path.join(dest_folder, file_name)
            
            self.s3.download_file(bucket_name, key, tmp_path)
            return tmp_path
        except Exception as e:
            self.logger.error(f"Failed to download file from {src_URI} to {dest_folder}: {str(e)}")
            raise
    
    
    def upload_mask(self, mask: np.ndarray, mask_uri: str) -> bool:
        """Save mask as PNG and upload to S3"""
        
        os.makedirs("tmp-masks", exist_ok=True)
        tmp_path = os.path.join("tmp-masks", "tmp_mask.png")
        cv2.imwrite(tmp_path, mask)
        
        # upload to S3
        success = self.upload_file(tmp_path, mask_uri)
        
        # clean-up   
        if success:
            os.remove(tmp_path)
        
        return success


    def download_segmented_pcd(self, folder_URI: str) -> str:
        self.logger.info(f"=======================")
        self.logger.info(f"Downloading left-segmented-labelled.ply!")
        self.logger.info(f"=======================\n")
        
        segmented_pcd_uri = folder_URI + f"left-segmented-labelled.ply" 
        return self.download_file(segmented_pcd_uri, "tmp-pcd")
                

        

    def generate_bev(self, imgL_uri: str, imgR_uri: str):
        pass
    

class DataGeneratorS3:
    def __init__(self, src_uris: List[str] = None, dest_folder: str = None, index_json: str = None):    
        self.logger = get_logger("DataGeneratorS3")
     
    def get_target_folder_uri(self, src_uri: str, dest_folder:str = "bev-dataset"):
        ''' Make leaf-folder path relative to the bev-dataset folder '''
        return src_uri.replace("occ-dataset", dest_folder, 1)
        
   
    def get_leaf_folders(self, s3_uri: str) -> List[str]:
        """Get all leaf folders URI inside the given S3 URI"""
        
        # Parse S3 URI to get bucket and prefix
        s3_parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        prefix = s3_parts[1] if len(s3_parts) > 1 else ""
        
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Get all objects with the given prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        # Keep track of all folders and their parent folders
        all_folders = set()
        parent_folders = set()
        
        # Process all objects
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                # Get the full path
                path = obj['Key']
                
                # Skip the prefix itself
                if path == prefix:
                    continue
                    
                # Get all folder paths in this object's path
                parts = path.split('/')
                for i in range(len(parts)-1):
                    folder = '/'.join(parts[:i+1])
                    if folder:
                        all_folders.add(folder)
                        
                        # If this isn't the immediate parent, it's a parent folder
                        if i < len(parts)-2:
                            parent_folders.add(folder)
        
        # Leaf folders are those that aren't parents of other folders
        leaf_folders = all_folders - parent_folders
        
        # Convert back to S3 URIs
        leaf_folder_uris = [f"s3://{bucket_name}/{folder}" for folder in sorted(leaf_folders)]
        
        return leaf_folder_uris


    def generate_bev_dataset(self, src_uri: str, dest_folder: str = "bev-dataset"):
        ''' Generate a BEV dataset from the given S3 URI '''
        
        leaf_URIs = self.get_leaf_folders(src_uri)
        random.shuffle(leaf_URIs)
        
        for idx, leaf_URI in tqdm(enumerate(leaf_URIs), total=len(leaf_URIs), desc="Processing leaf URIs"):    
            target_folder = self.get_target_folder_uri(leaf_URI, dest_folder)
            leaf_folder = LeafFolder(leaf_URI, target_folder)
            leaf_folder.process_folder()


if __name__ == "__main__":
    # uri_list = [
    #     "s3://occupancy-dataset/occ-dataset/vineyards/gallo/",
    # ]

    logger = get_logger("__main__")
    
    data_generator_s3 = DataGeneratorS3()
    # src_URI = "s3://occupancy-dataset/occ-dataset/vineyards/RJM/"
    
    leaf_URI_src = \
        "s3://occupancy-dataset/" \
        "occ-dataset/" \
        "vineyards/gallo/" \
        "2024_06_07_utc/svo_files/front_2024-06-04-10-24-57.svo/1398_to_1540/frame-1400/"

    leaf_URI_dest = \
        "s3://occupancy-dataset/" \
        "bev-dataset/" \
        "vineyards/gallo/" \
        "2024_06_07_utc/svo_files/front_2024-06-04-10-24-57.svo/1398_to_1540/frame-1400/"
    
    # data_generator_s3.generate_bev_dataset(src_URI)
    
    logger.info(f"=======================")
    logger.info(f"leaf_URI_src: {leaf_URI_src}")
    logger.info(f"leaf_URI_dest: {leaf_URI_dest}")
    logger.info(f"=======================\n")

    leafFolder = LeafFolder(leaf_URI_src, leaf_URI_dest)
    leafFolder.process_folder()
