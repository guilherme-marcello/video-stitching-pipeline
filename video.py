import os
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# our code
import files
import vision

class EnrichedFrame:
    """
    This class represents a frame with yolo detections (if any).
    """

    def __init__(self, frame, detections_bb, number=-1):
        self.frame = frame
        self.detections_bb = detections_bb
        self.number = number

    @classmethod
    def from_path(cls, image_path: str, yolo_path: str, number: int):
        frame = cv2.imread(image_path)
        detections_bb = loadmat(yolo_path)["xyxy"] if yolo_path else [] # bounding boxes
        return cls(frame, detections_bb, number)
    
    def get_number(self) -> int:
        return self.number
    
    def _export_detections(self, output_path: str) -> None:
        """
        Export the detections to a .mat file.
        
        Args:
        output_path: str
            Path to the output .mat file.
        """
        if self.has_detections():
            savemat(output_path, {"xyxy": self.detections_bb})

    def _export_frame(self, output_path: str) -> None:
        """
        Export the frame to an image file.
        
        Args:
        output_path: str
            Path to the output image file.
        """
        cv2.imwrite(output_path, self.frame)

    def export(self, output_dir: str) -> None:
        """
        Export the frame and detections to the output directory.
        
        Args:
        output_dir: str
            Path to the output directory
        """
        self._export_detections(os.path.join(output_dir, f"yolo_{self.number}.mat"))
        self._export_frame(os.path.join(output_dir, f"img_{self.number}.jpg"))

    def transform(self, H):
        """
        Warp the frame using the homography matrix H.
        
        Args:
        H: np.ndarray (3, 3)
            Homography matrix
        
        Returns:
            EnrichedFrame: A new EnrichedFrame object with the warped frame and detections.
        """
        height, width = self.frame.shape[:2]

        warped_frame = vision.warp_image(self.frame, H, (height, width))
        warped_detections = [vision.warp_bounding_box(box, H) for box in self.detections_bb]
        return EnrichedFrame(frame=warped_frame, detections_bb=warped_detections, number=self.number)


    def has_detections(self):
        return len(self.detections_bb) > 0
    
    def show(self):
        """
        Show the frame with the detections.
        """
        frame = self.frame.copy()
        for box in self.detections_bb:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {self.number}")
        plt.show()

class EnrichedVideo:
    """
    This class represents video with yolo detections.
    """
    def __init__(self, frames: list[EnrichedFrame]):
        self.frames = frames
        self.frames = sorted(self.frames, key=lambda frame: frame.get_number())

    @classmethod
    def from_input_directory(cls, input_dir: str):
        _filenames = os.listdir(input_dir)

        # list and order all img_0 in the input directory saving it to a list. do the same for files starting with "yolo"
        frames_path = {
            files.extract_number(filename): filename
            for filename in _filenames if filename.startswith("img_")
        }

        yolo_detections_path = {
            files.extract_number(filename): filename
            for filename in _filenames if filename.startswith("yolo_")
        }

        # now we need to match frames with yolo detections. if no objects were detected in some frame, there will be no YOLO file for that frame.
        frames = [
            EnrichedFrame.from_path(
                image_path=os.path.join(input_dir, frame_path),
                yolo_path=os.path.join(input_dir, yolo_detections_path[frame_number]) if frame_number in yolo_detections_path else None,
                number=frame_number
            )
            for frame_number, frame_path in frames_path.items()
        ]

        return cls(frames)

    def get_number_of_frames_with_detections(self):
        return sum([1 if frame.has_detections() else 0 for frame in self.frames])

    def __getitem__(self, idx) -> EnrichedFrame:
        return self.frames[idx]

    def __len__(self):
        return len(self.frames)
    
    def __iter__(self) -> EnrichedFrame:
        return iter(self.frames)
        
