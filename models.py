import numpy as np
import open3d as o3d

class ImageFeatures:
    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors

class CameraInfo:
    def __init__(self, rgb_frame, depth_map, confidence_map, focal_length):
        self.rgb_frame = rgb_frame
        self.depth_map = depth_map
        self.confidence_map = confidence_map
        self.focal_length = focal_length

        self.K = np.array([[focal_length, 0, rgb_frame.shape[0] // 2],
                           [0, focal_length, rgb_frame.shape[1] // 2],
                           [0, 0, 1]])

class SceneData:
    def __init__(self, image_features: ImageFeatures, camera_info: CameraInfo):
        self.image_features = image_features
        self.camera_info = camera_info

    def as_point_cloud(self, confidence_threshold=0.95):
        height, width, _ = self.camera_info.rgb_frame.shape

        # Generate pixel grid
        u = np.arange(width)
        v = np.arange(height)
        uu, vv = np.meshgrid(u, v)

        # Compute 3D points
        X = (uu - width // 2) * self.camera_info.depth_map / self.camera_info.focal_length
        Y = (vv - height // 2) * self.camera_info.depth_map / self.camera_info.focal_length
        Z = self.camera_info.depth_map

        # Flatten arrays for Open3D processing
        points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        colors = self.camera_info.rgb_frame.reshape(-1, 3)

        # Flatten confidence map
        conf_flat = self.camera_info.confidence_map.flatten()
        conf_flat_normalized = conf_flat / np.max(conf_flat) # Normalize to [0, 1]

        # Mask out invalid points (e.g., where depth is zero or negative)
        valid_mask = Z.flatten() > 0 & (conf_flat_normalized > confidence_threshold)
        points = points[valid_mask]
        colors = colors[valid_mask]

        # Create an Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd


        