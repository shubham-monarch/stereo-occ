#! /usr/bin/env python3

import boto3
from tqdm import tqdm
import logging
import coloredlogs
import random
import os

# custom imports
from utils.aws_utils import download_s3_file

# LOGGING SETUP
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
coloredlogs.install(level='INFO', logger=logger, force=True)



def list_unique_segmented_pcds(s3_uri: str):
    """
    Takes an S3 URI for a folder and returns the S3 URIs of all 'left-segmented-labelled.ply' files inside 'frame-' sub-folders recursively.
    Ensures that all the selected 'frame-' folders have a unique parent folder.
    """
    s3 = boto3.client('s3')
    
    # Parse the S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")
    
    s3_uri = s3_uri[5:]
    bucket_name, prefix = s3_uri.split('/', 1)
    
    def list_frame_subfolders(bucket_name, prefix, pbar, parent_folders):
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        frame_files_list = []
        
        if 'CommonPrefixes' in response:
            subfolders = [common_prefix['Prefix'] for common_prefix in response['CommonPrefixes']]
            for subfolder in subfolders:
                pbar.update(1)
                parent_folder = subfolder.split('/')[-3]  # Get the parent folder name
                if subfolder.startswith(prefix + 'frame-') and parent_folder not in parent_folders:
                    file_key = subfolder + 'left-segmented-labelled.ply'
                    try:
                        s3.head_object(Bucket=bucket_name, Key=file_key)
                        frame_files_list.append(f"s3://{bucket_name}/{file_key}")
                        parent_folders.add(parent_folder)
                    except:
                        pass
                frame_files_list.extend(list_frame_subfolders(bucket_name, subfolder, pbar, parent_folders))
        
        return frame_files_list
    
    # Get the total number of subfolders for the progress bar
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    total_subfolders = sum(1 for _ in response.get('CommonPrefixes', []))
    
    with tqdm(total=total_subfolders, desc="Processing subfolders") as pbar:
        parent_folders = set()
        return list_frame_subfolders(bucket_name, prefix, pbar, parent_folders)

def get_parent_folder_s3_uri(s3_uri: str) -> str:
    """
    Takes an S3 URI of a file and returns the S3 URI of the parent folder of the file.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")
    
    s3_uri = s3_uri.rstrip('/')
    parent_folder_uri = '/'.join(s3_uri.split('/')[:-1]) + '/'
    
    return parent_folder_uri



if __name__ == "__main__":
    s3_uri = "s3://occupancy-dataset/occ-dataset/vineyards/gallo/"
    demo_folder = "friday-demo"    

    segmented_pointcloud_files = list_unique_segmented_pcds(s3_uri)
    # logger.info(f"Found {len(s3_folders)} folders")
    # for folder in s3_folders:
    #     logger.info(folder)
    
    segmented_pointcloud_folders = [get_parent_folder_s3_uri(file) for file in segmented_pointcloud_files]

    for folder in segmented_pointcloud_folders:
        logger.info(folder)


        # Select 50 random folders
        random_folders = random.sample(segmented_pointcloud_folders, min(50, len(segmented_pointcloud_folders)))

        s3_client = boto3.client('s3')
        bucket_name, _ = s3_uri.replace("s3://", "").split("/", 1)
        with tqdm(total=len(random_folders), desc="Processing folders") as pbar:
            for idx, folder in enumerate(random_folders, start=1):
                folder_name = folder.split('/')[-2] + "_" + str(idx)  # Create a unique folder name
                local_folder_path = os.path.join(demo_folder, folder_name)
                
                if not os.path.exists(local_folder_path):
                    os.makedirs(local_folder_path)
                pbar.update(1)
                        
            # List all files in the folder
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder[len("s3://") + len(bucket_name) + 1:])
            if 'Contents' in response:
                total_files = len(response['Contents'])
                for obj in response['Contents']:
                    file_key = obj['Key']
                    local_file_path = os.path.join(local_folder_path, os.path.basename(file_key))
                    download_s3_file(f"s3://{bucket_name}/{file_key}", local_file_path)