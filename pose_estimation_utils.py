import numpy as np

def compute_transformation(matches, keypoints1, keypoints2):
    """Estimate the rotation and translation between two frames using Procrustes Analysis."""
    matched_kp1 = keypoints1[matches[:, 0]]
    matched_kp2 = keypoints2[matches[:, 1]]

    # Compute centroids
    centroid1 = np.mean(matched_kp1, axis=0)
    centroid2 = np.mean(matched_kp2, axis=0)

    # Center the points
    kp1_centered = matched_kp1 - centroid1
    kp2_centered = matched_kp2 - centroid2

    # Compute rotation matrix using SVD
    H = kp1_centered.T @ kp2_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation matrix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation vector
    t = centroid2 - R @ centroid1

    return R, t