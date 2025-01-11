# Video Processing Pipeline with Homography and YOLO Detection

## Overview

This project processes video frames with YOLO object detection and computes a homography matrix to map detections onto a reference aerial image (e.g., a Google Maps image). It includes functionality for image warping, bounding box transformations, and frame-by-frame video processing.

## Features

- **Input video parsing**: Handles image frames and YOLO detections together.
- **Homography Calculation**: Computes a transformation matrix between a source and destination image.
- **Warping**: Warps image frames and bounding boxes using the homography matrix.
- **Exporting**: Exports processed frames and detection data for further analysis.

---

## Assumptions

- **Single Camera**: This pipeline assumes the use of a single, static camera. This simplifies the computation of the homography matrix, as the scene and reference frame remain consistent throughout the video.
- **Static Camera**: A static camera ensures that the relationship between the video frames and the reference aerial image does not change, which is critical for accurate homography calculations.
- **No Feature Detection**: This pipeline does not perform feature detection. Instead, it focuses on applying precomputed keypoint matches to compute the homography matrix.
- **No OpenCV**: The pipeline is implemented without using OpenCV, leveraging libraries like NumPy and SciPy for matrix operations and image transformations.

---

## Project Structure

### Files

- `main.py`:
  - Entry point for the pipeline.
  - Parses command-line arguments and coordinates the video processing workflow.
  
- `video.py`:
  - Contains classes for enriched video and frame processing.
  - Handles loading, exporting, and visualization of frames and their associated YOLO detections.

- `vision.py`:
  - Provides functions for homography computation, image warping, and bounding box transformation.

- `files.py`:
  - Utility functions for handling file operations, such as extracting numbers from filenames to match frames with YOLO detections.

- `LICENSE`:
  - Licensing details for the project.

---

## Dependencies

- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib

To install the required dependencies, run:

```bash
pip install numpy scipy matplotlib
```

---

## Usage

1. **Prepare Input Data**:
   - Ensure all input frames are named `img_<frame_number>.jpg`.
   - Ensure YOLO detection outputs are named `yolo_<frame_number>.mat`.
   - Place these files in a directory.

2. **Run the Pipeline**:
   Use the following command to process the video frames:

   ```bash
   python main.py -kp <keypoint_matches_file> -map <google_maps_image> -i <input_directory>
   ```

   - `-kp`: Path to the keypoint matches file (default: `kp_gmaps.mat`).
   - `-map`: Path to the Google Maps image (default: `gmaps.png`).
   - `-i`: Input directory containing frames and YOLO detections (default: `.`).

3. **Output**:
   - Warped frames and detection data will be exported to the output directory.

---

## How It Works

1. **Load Video Frames**:
   - Frames and YOLO detection outputs are matched by their filenames and loaded into `EnrichedFrame` objects.

2. **Compute Homography**:
   - A homography matrix is calculated using keypoint matches between the first frame and a reference aerial image.

3. **Warp Frames**:
   - Frames and bounding boxes are transformed using the computed homography matrix.

4. **Export Results**:
   - Processed frames and detection outputs are saved to the specified directory for further analysis.

---

## License

This project is licensed under the terms specified in the `LICENSE` file.