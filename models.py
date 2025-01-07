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

    def get_valid_keypoints_and_descriptors(self, confidence_threshold=0.65):
        """
        Filters keypoints and descriptors based on valid depth and confidence values.
        Returns only valid keypoints and corresponding descriptors.
        """
        # extract keypoints and descriptors
        keypoints = self.image_features.keypoints
        descriptors = self.image_features.descriptors

        # initialize lists for valid keypoints and descriptors
        valid_keypoints = []
        valid_descriptors = []

        # loop through keypoints
        for i, keypoint in enumerate(keypoints):
            u, v = int(keypoint[0]), int(keypoint[1])

            # check depth and confidence
            depth = self.camera_info.depth_map[v, u]
            confidence = self.camera_info.confidence_map[v, u]

            if depth > 0 and (confidence / np.max(self.camera_info.confidence_map)) > confidence_threshold:
                valid_keypoints.append(keypoint)
                valid_descriptors.append(descriptors[i])

        # convert valid descriptors to a numpy array
        valid_descriptors = np.array(valid_descriptors, dtype=np.float32)

        return valid_keypoints, valid_descriptors

    def as_point_cloud(self, confidence_threshold=0.95):
        height, width, _ = self.camera_info.rgb_frame.shape

        # generate pixel grid
        u = np.arange(width)
        v = np.arange(height)
        uu, vv = np.meshgrid(u, v)

        # compute 3D points
        X = (uu - width // 2) * self.camera_info.depth_map / self.camera_info.focal_length
        Y = (vv - height // 2) * self.camera_info.depth_map / self.camera_info.focal_length
        Z = self.camera_info.depth_map

        # flatten arrays for Open3D processing
        points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        colors = self.camera_info.rgb_frame.reshape(-1, 3)

        # flatten confidence map
        conf_flat = self.camera_info.confidence_map.flatten()
        conf_flat_normalized = conf_flat / np.max(conf_flat) # Normalize to [0, 1]

        # mask out invalid points (e.g., where depth is zero or negative)
        valid_mask = Z.flatten() > 0 & (conf_flat_normalized > confidence_threshold)
        points = points[valid_mask]
        colors = colors[valid_mask]

        # create an open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd


        