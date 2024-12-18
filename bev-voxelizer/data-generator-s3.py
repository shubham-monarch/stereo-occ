#! /usr/bin/env python3

import boto3
import os
import random
from logger import get_logger
from typing import List
from tqdm import tqdm

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
        
        # 1. download left-segmented-labelled.ply
        self.download_segmented_pcd()

        # 2. copy imgL
        # 3. copy imgR
        # 4. upload seg_mono
        # 5. upload seg_rgb
        # 6. upload ipm_fea
        # 6. upload ipm_rgb



        pass

    

    def copy_imgL(self, imgL_uri: str):
        pass

    def copy_imgR(self, imgR_uri: str):
        pass

    def upload_seg_mono(self, seg_mono_uri: str):
        pass

    def upload_seg_rgb(self, seg_rgb_uri: str):
        pass

    def upload_ipm_fea(self, ipm_fea_uri: str):
        pass

    def upload_ipm_rgb(self, ipm_seg_uri: str):
        pass

    def download_segmented_pcd(self):
        self.logger.info(f"=======================")
        self.logger.info(f"Downloading left-segmented-labelled.ply!")
        self.logger.info(f"=======================\n")
        
        segmented_pcd_uri = self.src_URI + f"left-segmented-labelled.ply" 
        bucket_name, key = segmented_pcd_uri.replace("s3://", "").split("/", 1)
                
        os.makedirs("tmp", exist_ok=True)
        path_tmp = os.path.join("tmp", "left-segmented-labelled.ply")
        
        self.s3.download_file(bucket_name, key, path_tmp)
        return path_tmp

        

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
    logger.info(f"=======================")

    leafFolder = LeafFolder(leaf_URI_src, leaf_URI_dest)
    leafFolder.process_folder()
