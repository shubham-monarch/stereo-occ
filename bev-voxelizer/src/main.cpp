
#include <iostream>
#include <memory>
#include <thread>

#include <open3d/Open3D.h>



int main(int argc, char *argv[]) {
    
    std::string ply_path = "../ply/segmented-1056_to_1198/1.ply"; 

    open3d::io::ReadPointCloudOption option;
    option.remove_nan_points = true;
    option.remove_infinite_points = true;
    option.print_progress = true;


    open3d::geometry::PointCloud pcd;
    open3d::io::ReadPointCloudFromPLY(ply_path, pcd, option);

    std::cout << "Number of points: " << pcd.points_.size() << std::endl;

    return 0;
}
