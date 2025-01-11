import argparse
from scipy.io import loadmat, savemat

from video import EnrichedVideo, EnrichedFrame
from vision import compute_homography

class Config:
    def __init__(self, kp, map, i):
        self.keypoint_matches = kp
        self.google_maps_image = map
        self.input_dir = i
        

def parse_args() -> Config:
    """
    Parses command-line arguments for the video processing pipeline.

    This function defines and parses command-line arguments 
    necessary for the video processing pipeline.

    Args:
        None

    Returns:
        Config: An object containing the parsed arguments config.
            - keypoint_matches (-kp): Path to the keypoint matches file. 
                                     Defaults to "kp_gmaps.mat".
            - google_maps_image (-map): Path to the Google Maps image. 
                                       Defaults to "gmaps_alamedaIST.png".
            - input_dir (-v): Directory containing video frames and YOLO detections.. 
                                    Defaults to the current directory. 
    """
    parser = argparse.ArgumentParser(description="PIV 1.1: Process video frames and YOLO detections.")
    parser.add_argument("--keypoint_matches", "-kp", type=str, default="kp_gmaps.mat",
                        help="Path to keypoint matches file.")
    parser.add_argument("--google_maps_image", "-map",  type=str, default="gmaps_alamedaIST.png",
                        help="Path to Google Maps image.")
    parser.add_argument("--input_dir", "-i", type=str, default=".",
                        help="Directory containing video frames and YOLO detections.")
    return parser.parse_args()

def main(config: Config):
    # load detections and frames as a "video" object
    print(f"Loading files in {config.input_dir}, this may take a while...")
    video = EnrichedVideo.from_input_directory(config.input_dir)

    print(f"Loaded {len(video)} frames ({video.get_number_of_frames_with_detections()} with detections)!")

    # load keypoint matches
    matches_raw = loadmat(config.keypoint_matches) 

    # this matches are regarding first image img_0001.jpg -> maps image
    matches = matches_raw["kp_gmaps"]

    # load source points
    src_points = matches[:, :2]
    
    # load dst points
    dst_points = matches[:, 2:]

    # compute homography!
    H = compute_homography(src_points, dst_points)
    savemat("homography.mat", {"H": H})

    for frame in video:
        frame: EnrichedFrame
        warped_frame = frame.transform(H)
        warped_frame.export()    

if __name__ == "__main__":
    config = parse_args()
    main(config)