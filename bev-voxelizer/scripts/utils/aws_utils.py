#! /usr/bin/env python3

import boto3
import logging
import coloredlogs
from tqdm import tqdm

# LOGGING SETUP
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
coloredlogs.install(level='INFO', logger=logger, force=True)

def list_unique_frame_subfolders(s3_uri: str):
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
    
    # Get the total number of subfolders for the progress bar
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    total_subfolders = sum(1 for _ in response.get('CommonPrefixes', []))
    
    with tqdm(total=total_subfolders, desc="Processing subfolders") as pbar:
        parent_folders = set()
        return list_frame_subfolders(bucket_name, prefix, pbar, parent_folders)



def list_s3_subfolders(s3_uri: str):
    """
    Takes an S3 URI for a folder and returns the S3 URIs of all the sub-folders ending with .svo inside that folder recursively.
    """
    s3 = boto3.client('s3')
    
    # Parse the S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")
    
    s3_uri = s3_uri[5:]
    bucket_name, prefix = s3_uri.split('/', 1)
    
    def list_subfolders(bucket_name, prefix, pbar):
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        subfolders_list = []
        
        if 'CommonPrefixes' in response:
            subfolders = [common_prefix['Prefix'] for common_prefix in response['CommonPrefixes']]
            for subfolder in subfolders:
                pbar.update(1)
                if subfolder.endswith('.svo/'):
                    subfolders_list.append(f"s3://{bucket_name}/{subfolder}")
                subfolders_list.extend(list_subfolders(bucket_name, subfolder, pbar))
        
        return subfolders_list
    
    # Get the total number of subfolders for the progress bar
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    total_subfolders = sum(1 for _ in response.get('CommonPrefixes', []))
    
    with tqdm(total=total_subfolders, desc="Processing subfolders") as pbar:
        return list_subfolders(bucket_name, prefix, pbar)




if __name__ == "__main__":
    
    s3_uri = "s3://occupancy-dataset/occ-dataset/vineyards/treasury/"
    svo_folders = list_s3_subfolders(s3_uri)
    logger.info(svo_folders)

    folder = svo_folders[0]
    frame_folders = list_unique_frame_subfolders(folder)
    logger.info(frame_folders)

   