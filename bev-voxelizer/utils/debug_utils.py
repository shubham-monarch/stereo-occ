import matplotlib.pyplot as plt

''' Debugging utilities for bev-voxelizer '''


def plot_bev_y_values(bev_point_clouds: list) -> None:
    ''' Plot y-values for each BEV point cloud '''
    
    y_values = []
    for bev_point_cloud in bev_point_clouds:
        y_values.append(bev_point_cloud.point['positions'].numpy()[:, 1])

    # Plot y-values for each BEV point cloud on the same plot
    plt.figure(figsize=(10, 8))
    colors = ['r.', 'g.', 'b.', 'c.', 'm.']
    for y_value, color in zip(y_values, colors):
        plt.plot(y_value, color, alpha=0.7)

    plt.title('Y values for all BEV point clouds')
    plt.xlabel('Index')
    plt.ylabel('Y Value')
    plt.tight_layout()
    plt.show()


def plot_bev_scatter(bev_collection: list) -> None:
    ''' Create a scatter plot of BEV points (x,z) for each class with different scales for x and y axes '''
    
    plt.figure(figsize=(10, 8))
    
    # Plot each BEV class with different colors and labels
    colors = [
        (48/255, 94/255, 173/255),  # Navigable (BGR: [173, 94, 48])
        (117/255, 171/255, 68/255), # Canopy (BGR: [68, 171, 117])
        (148/255, 119/255, 121/255),# Pole (BGR: [121, 119, 148])
        (174/255, 122/255, 162/255),# Stem (BGR: [162, 122, 174])
        (228/255, 4/255, 246/255)   # Obstacle (BGR: [246, 4, 228])
    ]
    labels = ['Navigable', 'Canopy', 'Pole', 'Stem', 'Obstacle']
    
    for bev, color, label in zip(bev_collection, colors, labels):
        positions = bev.point['positions'].numpy()
        plt.scatter(positions[:, 0] , positions[:, 2] / 10.0, color=color, label=label, alpha=0.5, s=1)
    
    plt.xlabel('X (centimeters)')
    plt.ylabel('Z (decameters)')
    plt.title('Bird\'s Eye View Point Distribution')
    plt.legend()
    plt.grid(True)
    
    # Set range for x and y axis
    plt.xlim(-30, 30)  # Example range for x-axis in centimeters
    plt.ylim(0, 20)    # Example range for y-axis in decameters
    
    # Save and show the plot
    plt.savefig('bev_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()