#include <iostream>
#include <open3d/Open3D.h>

int main(int argc, char *argv[]) {
    std::string ply_path = "1.ply";

    // Read the point cloud
    auto pcd = open3d::io::CreatePointCloudFromFile(ply_path);
    if (pcd == nullptr) {
        std::cerr << "Failed to read the PLY file." << std::endl;
        return 1;
    }

    // Convert to tensor
    // open3d::core::Tensor points = pcd->GetPointPositions();
    // open3d::core::Tensor colors = pcd->GetPointColors();

    // Create a tensor for labels
    open3d::core::Tensor labels;
    if (pcd->HasPoints()) {
        // Assuming labels are stored in the same order as points
        labels = open3d::core::Tensor(pcd->points_.size(), {3}, open3d::core::Dtype::UInt8);
        for (size_t i = 0; i < pcd->points_.size(); ++i) {
            labels[i] = static_cast<uint8_t>(pcd->points_[i][3]); // Assuming label is the 4th value
            std::cout << "Label: " << labels[i] << std::endl;
        }
    } else {
        std::cout << "No points found in the point cloud." << std::endl;
    }

    // Combine all attributes into a single tensor
    // std::vector<open3d::core::Tensor> attributes = {points, colors};
    // if (labels.NumElements() > 0) {
    //     attributes.push_back(labels);
    // }
    // open3d::core::Tensor all_data = open3d::core::Tensor::Concat(attributes, 1);

    // // Print information about the tensor
    // std::cout << "Tensor shape: " << all_data.GetShape().ToString() << std::endl;
    // std::cout << "Tensor dtype: " << all_data.GetDtype().ToString() << std::endl;

    // // Print the first few rows
    // std::cout << "First few rows:" << std::endl;
    // std::cout << all_data.Slice(0, 0, 5).ToString() << std::endl;

    return 0;
}