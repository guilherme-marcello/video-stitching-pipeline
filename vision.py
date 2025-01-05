import numpy as np
from scipy.linalg import svd

def warp_image(image, H, output_size) -> np.array:
    """
    Warp an image using a homography matrix H.
    """

    height, width = output_size
    warped_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Compute the inverse of the homography matrix for backward warping
    H_inv = np.linalg.inv(H)

    # Iterate over every pixel in the output image
    for y in range(height):
        for x in range(width):
            # Transform (x, y) in the output image to (x', y') in the input image, that's why we use H_inv..!
            homogeneous_point = np.array([x, y, 1])
            transformed_point = np.dot(H_inv, homogeneous_point)
            transformed_point /= transformed_point[2]  # Normalize by z (homogeneous scale)

            x_prime, y_prime = transformed_point[:2]
            x_prime, y_prime = int(x_prime), int(y_prime)

            # Check if the transformed point is within bounds of the input image
            if 0 <= x_prime < image.shape[1] and 0 <= y_prime < image.shape[0]:
                warped_image[y, x] = image[y_prime, x_prime]

    return warped_image


def warp_bounding_box(box, H) -> np.array:
    # extract corner points
    points = np.array([
        [box[0], box[1], 1],  # Bottom-left corner
        [box[2], box[1], 1],  # Bottom-right corner
        [box[2], box[3], 1],  # Top-right corner
        [box[0], box[3], 1]   # Top-left corner
    ])
    
    # apply transformation matrix H
    transformed_points = (H @ points.T).T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2].reshape(-1, 1)  # Normalize

    # Get new bounding box
    x_min = int(transformed_points[:, 0].min())
    y_min = int(transformed_points[:, 1].min())
    x_max = int(transformed_points[:, 0].max())
    y_max = int(transformed_points[:, 1].max())
    return np.array([x_min, y_max, x_max, y_min])

def compute_homography(src_pts, dst_pts):
    """
    Compute the homography matrix H that maps src_pts to dst_pts.
    
    Parameters:
    src_pts: np.ndarray (N, 2)
        Source points in the original image
    dst_pts: np.ndarray (N, 2)
        Destination points in the aerial view
    
    Returns:
    H: np.ndarray (3, 3)
        Homography matrix
    """
    num_points = src_pts.shape[0]
    assert num_points >= 4, "At least 4 points are required to compute a homography."

    # Construct the matrix A for Ah = 0
    A = []
    for i in range(num_points):
        x, y = src_pts[i]
        x_prime, y_prime = dst_pts[i]
        A.append([x, y, 1, 0, 0, 0, -x_prime*x, -x_prime*y, -x_prime])
        A.append([0, 0, 0, x, y, 1, -y_prime*x, -y_prime*y, -y_prime])
    A = np.array(A)

    # Solve Ah = 0 using Singular Value Decomposition (SVD)
    # right singular vectors (At.A) -> Vt
    U, S, Vt = svd(A)
    h = Vt[-1]  # The solution is the last row of Vt
    H = h.reshape((3, 3))  # Reshape h into a 3x3 matrix
    H /= H[2,2]
    return H