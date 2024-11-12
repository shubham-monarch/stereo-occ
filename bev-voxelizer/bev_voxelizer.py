import logging, coloredlogs
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import open3d.core as o3c


# LOGGING SETUP
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
coloredlogs.install(level='INFO', logger=logger, force=True)


class BevVoxelizer:
    def __init__(self):
        
        self.LABELS = {    
            "OBSTACLE":        {"id": 1, "priority": 1},
            "VINE_POLE":       {"id": 5, "priority": 2},
            "VINE_CANOPY":     {"id": 3, "priority": 3},
            "VINE_STEM":       {"id": 4, "priority": 4},
            "NAVIGABLE_SPACE": {"id": 2, "priority": 5},
            "TRACTOR_HOOD":    {"id": 7, "priority": 9999}
        }
    
    def filter_radius_outliers(self, pcd: o3d.t.geometry.PointCloud, nb_points: int, search_radius: float):
        '''
        Filter radius-based outliers from the point cloud
        '''
        _, ind = pcd.remove_radius_outliers(nb_points=nb_points, search_radius=search_radius)
        inliers = pcd.select_by_mask(ind)
        outliers = pcd.select_by_mask(ind, invert=True)
        return inliers, outliers

    def get_plane_model(self,pcd: o3d.t.geometry.PointCloud, class_label: int):
        '''
        returns [a,b,c,d]
        '''
        pcd_class = self.get_class_pointcloud(pcd, class_label)
        plane_model, inliers = pcd_class.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        return plane_model.numpy()
    
    def plot_plane_histogram(self,pcd: o3d.t.geometry.PointCloud):
    
        
        positions = pcd.point['positions'].numpy()
        x_values = positions[:, 0]
        y_values = positions[:, 1]
        z_values = positions[:, 2]

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        axs[0].hist(x_values, bins=50, color='r', alpha=0.7)
        axs[0].set_title('Histogram of X values')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(y_values, bins=50, color='g', alpha=0.7)
        axs[1].set_title('Histogram of Y values')
        axs[1].set_xlabel('Y')
        axs[1].set_ylabel('Frequency')

        axs[2].hist(z_values, bins=50, color='b', alpha=0.7)
        axs[2].set_title('Histogram of Z values')
        axs[2].set_xlabel('Z')
        axs[2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def align_normal_to_y_axis(self,normal_):
        '''
        Rotation matrix to align the normal vector to the y-axis
        '''
        y_axis = np.array([0, 1, 0])
        v = np.cross(normal_, y_axis)
        s = np.linalg.norm(v)
        c = np.dot(normal_, y_axis)
        I = np.eye(3)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        return R
    
    def generate_unified_bev_pcd(self, bev_collection):
        '''
        Generate a unified BEV pointcloud from a collection of label-specific BEV pointclouds
        '''
        # Vertically stack the point positions of the bev_collection pointclouds
        position_tensors = [pcd.point['positions'].numpy() for pcd in bev_collection]
        stacked_positions = o3c.Tensor(np.vstack(position_tensors), dtype=o3c.Dtype.Float32)
        
        # Vertically stack the point labels of the bev_collection pointclouds
        label_tensors = [pcd.point['label'].numpy() for pcd in bev_collection]
        stacked_labels = o3c.Tensor(np.vstack(label_tensors), dtype=o3c.Dtype.Int32)

        # Vertically stack the point colors of the bev_collection pointclouds
        color_tensors = [pcd.point['colors'].numpy() for pcd in bev_collection]
        stacked_colors = o3c.Tensor(np.vstack(color_tensors), dtype=o3c.Dtype.UInt8)
        
        # Create a unified point cloud
        map_to_tensors = {}
        map_to_tensors['positions'] = stacked_positions
        map_to_tensors['label'] = stacked_labels
        map_to_tensors['colors'] = stacked_colors
        
        combined_pcd = o3d.t.geometry.PointCloud(map_to_tensors)    
        return combined_pcd


    def axis_angles(self,vec):
        '''
        Calculate the angles between input vector and the coordinate axes
        '''
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        angle_x = np.arccos(np.dot(vec, x_axis) / np.linalg.norm(vec))
        angle_y = np.arccos(np.dot(vec, y_axis) / np.linalg.norm(vec))
        angle_z = np.arccos(np.dot(vec, z_axis) / np.linalg.norm(vec))
        
        return np.degrees(angle_x), np.degrees(angle_y), np.degrees(angle_z)

    def compute_tilt_matrix(self, pcd: o3d.t.geometry.PointCloud):
        '''
        Compute navigation-space tilt w.r.t y-axis
        '''
        normal, _ = self.get_class_plane(pcd, self.LABELS["NAVIGABLE_SPACE"]["id"])
        R = self.align_normal_to_y_axis(normal)
        return R

    def get_class_pointcloud(self, pcd: o3d.t.geometry.PointCloud, class_label: int):
        '''
        Returns class-specific point cloud
        '''
        mask = pcd.point["label"] == class_label
        pcd_labels = pcd.select_by_index(mask.nonzero()[0])
        return pcd_labels

    def get_class_plane(self,pcd: o3d.t.geometry.PointCloud, class_label: int):
        '''
        Get the inliers / normal vector for the labelled pointcloud
        '''
        pcd_class = self.get_class_pointcloud(pcd, class_label)
        plane_model, inliers = pcd_class.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        [a, b, c, d] = plane_model.numpy()
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal) 
        return normal, inliers
    
    def rotation_matrix_to_ypr(self,R):
        '''
        Convert rotation matrix to yaw, pitch, roll
        '''
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = 0

        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def generate_bev_voxels(self, pcd_input: o3d.t.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
        # pcd tilt correction
        R = self.compute_tilt_matrix(pcd_input)
        
        yaw, pitch, roll = self.rotation_matrix_to_ypr(R)

        logger.warning(f"=================================")    
        logger.warning(f"Yaw: {yaw:.2f} degrees, Pitch: {pitch:.2f} degrees, Roll: {roll:.2f} degrees")
        logger.warning(f"=================================")    
        
        # sanity check
        normal, _ = self.get_class_plane(pcd_input, self.LABELS["NAVIGABLE_SPACE"]["id"])
        normal_ = np.dot(normal, R.T)
        angles = self.axis_angles(normal_)
        
        logger.info(f"=================================")    
        logger.info(f"axis_angles: {angles}")
        logger.info(f"Ground plane makes {angles} degrees with axes!")
        logger.info(f"=================================\n")

        # angle between normal and y-axis should be close to 0 / 180 degrees
        if not np.isclose(angles[1], 0, atol=1) and np.isclose(angles[1], 180, atol=1):
            logger.error(f"=================================")    
            logger.error(f"Error: angles_transformed[1] is {angles[1]}, but it should be close to 0 degrees. Please check the tilt correction!")
            logger.error(f"=================================\n")
            exit(1)


        pcd_corrected = pcd_input.clone()
        pcd_corrected.rotate(R, center=(0, 0, 0))

        # filtering unwanted labels => [vegetation, tractor-hood, void, sky]
        valid_labels = np.array([label["id"] for label in self.LABELS.values()])
        valid_mask = np.isin(pcd_corrected.point['label'].numpy(), valid_labels)
        
        pcd_filtered = pcd_corrected.select_by_mask(valid_mask.flatten())
        original_points = len(pcd_corrected.point['positions'])
        filtered_points = len(pcd_filtered.point['positions'])
        reduction_percentage = ((original_points - filtered_points) / original_points) * 100    
        
        unique_labels = np.unique(pcd_filtered.point['label'].numpy())
        
        logger.info(f"=================================")    
        logger.info(f"Before filtering: {original_points}")
        logger.info(f"After filtering: {filtered_points}")
        logger.info(f"Reduction %: {reduction_percentage:.2f}%")
        logger.info(f"Unique labels in pcd_filtered: {unique_labels}")
        logger.info(f"=================================\n")
        
        # class-wise point cloud extraction
        pcd_canopy = self.get_class_pointcloud(pcd_filtered, self.LABELS["VINE_CANOPY"]["id"])
        pcd_pole = self.get_class_pointcloud(pcd_filtered, self.LABELS["VINE_POLE"]["id"])
        pcd_stem = self.get_class_pointcloud(pcd_filtered, self.LABELS["VINE_STEM"]["id"])
        pcd_obstacle = self.get_class_pointcloud(pcd_filtered, self.LABELS["OBSTACLE"]["id"])
        pcd_navigable = self.get_class_pointcloud(pcd_filtered, self.LABELS["NAVIGABLE_SPACE"]["id"])
        
        # num-points for each class
        total_points = len(pcd_filtered.point['positions'])
        canopy_points = len(pcd_canopy.point['positions'])
        pole_points = len(pcd_pole.point['positions'])
        stem_points = len(pcd_stem.point['positions'])
        obstacle_points = len(pcd_obstacle.point['positions'])
        navigable_points = len(pcd_navigable.point['positions'])

        # % points for each class
        canopy_percentage = (canopy_points / total_points) * 100
        pole_percentage = (pole_points / total_points) * 100
        stem_percentage = (stem_points / total_points) * 100
        obstacle_percentage = (obstacle_points / total_points) * 100
        navigable_percentage = (navigable_points / total_points) * 100

        logger.info(f"=================================")    
        logger.info(f"Total points: {total_points}")
        logger.info(f"Canopy points: {canopy_points} [{canopy_percentage:.2f}%]")
        logger.info(f"Pole points: {pole_points} [{pole_percentage:.2f}%]")
        logger.info(f"Stem points: {stem_points} [{stem_percentage:.2f}%]")
        logger.info(f"Obstacle points: {obstacle_points} [{obstacle_percentage:.2f}%]")
        logger.info(f"Navigable points: {navigable_points} [{navigable_percentage:.2f}%]")
        logger.info(f"=================================\n")

        # downsampling label-wise pointcloud
        down_pcd = pcd_filtered.voxel_down_sample(voxel_size=0.01)
        down_canopy = pcd_canopy.voxel_down_sample(voxel_size=0.01)
        down_navigable = pcd_navigable.voxel_down_sample(voxel_size=0.01)
        
        # NOT DOWN-SAMPLING [obstacle, stem, pole]
        down_obstacle = pcd_obstacle.clone()
        down_stem = pcd_stem.clone()
        down_pole = pcd_pole.clone()

        down_total_points = len(down_pcd.point['positions'].numpy())
        down_canopy_points = len(down_canopy.point['positions'])
        down_pole_points = len(down_pole.point['positions'])
        down_stem_points = len(down_stem.point['positions'])
        down_obstacle_points = len(down_obstacle.point['positions'])
        down_navigable_points = len(down_navigable.point['positions'])

        total_reduction_pct = (total_points - down_total_points) / total_points * 100 if total_points != 0 else 0
        canopy_reduction_pct = (canopy_points - down_canopy_points) / canopy_points * 100 if canopy_points != 0 else 0
        pole_reduction_pct = (pole_points - down_pole_points) / pole_points * 100 if pole_points != 0 else 0
        stem_reduction_pct = (stem_points - down_stem_points) / stem_points * 100 if stem_points != 0 else 0
        obstacle_reduction_pct = (obstacle_points - down_obstacle_points) / obstacle_points * 100 if obstacle_points != 0 else 0
        navigable_reduction_pct = (navigable_points - down_navigable_points) / navigable_points * 100 if navigable_points != 0 else 0
        
        logger.info(f"=================================")    
        logger.info(f"[AFTER DOWNSAMPLING]")
        logger.info(f"Total points: {down_total_points} [-{total_reduction_pct:.2f}%]")
        logger.info(f"Canopy points: {down_canopy_points} [-{canopy_reduction_pct:.2f}%]")
        logger.info(f"Pole points: {down_pole_points} [-{pole_reduction_pct:.2f}%]")
        logger.info(f"Stem points: {down_stem_points} [-{stem_reduction_pct:.2f}%]")
        logger.info(f"Obstacle points: {down_obstacle_points} [-{obstacle_reduction_pct:.2f}%]")
        logger.info(f"Navigable points: {down_navigable_points} [-{navigable_reduction_pct:.2f}%]")
        logger.info(f"=================================\n")
        
        # radius-based outlier removal
        rad_filt_pole = down_pole if len(down_pole.point['positions']) == 0 else self.filter_radius_outliers(down_pole, nb_points=16, search_radius=0.05)[0]
        rad_filt_stem = down_stem if len(down_stem.point['positions']) == 0 else self.filter_radius_outliers(down_stem, nb_points=16, search_radius=0.05)[0]
        # rad_filt_obstacle = down_obstacle if len(down_obstacle.point['positions']) == 0 else self.filter_radius_outliers(down_obstacle, nb_points=16, search_radius=0.05)[0]
        rad_filt_obstacle = down_obstacle if len(down_obstacle.point['positions']) == 0 else self.filter_radius_outliers(down_obstacle, nb_points=10, search_radius=0.05)[0]
        # rad_filt_navigable, _ = filter_radius_outliers(down_navigable, nb_points=16, search_radius=0.05)
        # rad_filt_canopy, _ = filter_radius_outliers(down_canopy, nb_points=1, search_radius=0.1)
        
        rad_filt_pole_points = len(rad_filt_pole.point['positions'].numpy())
        rad_filt_stem_points = len(rad_filt_stem.point['positions'].numpy())
        rad_filt_obstacle_points = len(rad_filt_obstacle.point['positions'].numpy())
        # rad_filt_navigable_points = len(rad_filt_navigable.point['positions'].numpy())
        # rad_filt_canopy_points = len(rad_filt_canopy.point['positions'].numpy())
        
        pole_reduction_pct = (down_pole_points - rad_filt_pole_points) / down_pole_points * 100 if down_pole_points != 0 else 0
        stem_reduction_pct = (down_stem_points - rad_filt_stem_points) / down_stem_points * 100 if down_stem_points != 0 else 0
        obstacle_reduction_pct = (down_obstacle_points - rad_filt_obstacle_points) / down_obstacle_points * 100 if down_obstacle_points != 0 else 0
        # navigable_reduction_pct = (down_navigable_points - rad_filt_navigable_points) / down_navigable_points * 100 if down_navigable_points != 0 else 0
        # canopy_reduction_pct = (down_canopy_points - rad_filt_canopy_points) / down_canopy_points * 100
        
        logger.info(f"=================================")    
        logger.info(f"[AFTER RADIUS-BASED OUTLIER REMOVAL]")
        logger.info(f"Pole points: {rad_filt_pole_points} [-{pole_reduction_pct:.2f}%]")
        logger.info(f"Stem points: {rad_filt_stem_points} [-{stem_reduction_pct:.2f}%]")
        logger.info(f"Obstacle points: {rad_filt_obstacle_points} [-{obstacle_reduction_pct:.2f}%]")
        # logger.info(f"Canopy points: {rad_filt_canopy_points} [-{canopy_reduction_pct:.2f}%]")
        # logger.info(f"Navigable points: {rad_filt_navigable_points} [-{navigable_reduction_pct:.2f}%]")
        logger.info(f"=================================\n")

        
        # projecting points to the navigable plane
        normal, inliers = self.get_class_plane(pcd_navigable, self.LABELS["NAVIGABLE_SPACE"]["id"])
        normal = normal / np.linalg.norm(normal)
        inliers_navigable = pcd_navigable.select_by_index(inliers)

        # self.plot_plane_histogram(inliers_navigable)
        
        # compute angle with y-axis
        angle_y = self.axis_angles(normal)[1]
        logger.info(f"Angle between normal and y-axis: {angle_y:.2f} degrees")

        # align normal with +y-axis if angle with y-axis is negative
        if angle_y < 0:
            normal = -normal

        
        inliers_positions = inliers_navigable.point['positions'].numpy()
        y_values = inliers_positions[:, 1]
        mean_y_value = float(np.mean(y_values))
        logger.info(f"Mean value of Y coordinates: {mean_y_value}")
        
        # label-wise BEV generations
        bev_navigable = inliers_navigable.clone()
        bev_navigable.point['positions'][:, 1] = float(mean_y_value)
        
        bev_obstacle = rad_filt_obstacle.clone()
        bev_obstacle.point['positions'][:, 1] = float(mean_y_value) - 0.1

        bev_pole = rad_filt_pole.clone()
        bev_pole.point['positions'][:, 1] = float(mean_y_value) - 0.09

        bev_stem = rad_filt_stem.clone()
        bev_stem.point['positions'][:, 1] = float(mean_y_value) - 0.08
        
        bev_canopy = down_canopy.clone()
        bev_canopy.point['positions'][:, 1] = float(mean_y_value) - 0.07

        bev_collection = [inliers_navigable, bev_canopy, bev_pole, bev_stem, bev_obstacle]
        combined_pcd = self.generate_unified_bev_pcd(bev_collection)
        return combined_pcd
