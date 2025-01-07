from models import ImageFeatures, CameraInfo, SceneData


class KPFileParser:
    def _extract_img_number(string):
        import re

        match = re.search(r'img(\d+)', string)
        if match:
            return int(match.group(1)) - 1 # to be easier, let's start from 0..!
        else:
            return None

    @classmethod
    def parse(cls, raw_loaded_kp) -> list[ImageFeatures]:
        keypoints_dict = dict()
        for key in raw_loaded_kp.keys():
            # this removes the header and version keys, etc
            if key.startswith("_"): 
                continue

            keypoints_dict[cls._extract_img_number(key)] = raw_loaded_kp[key][0][0]

        # here the keypoints_dict contains a map of img_number -> load keypoint file for this img
        loaded_keypoints = sorted(keypoints_dict.items(), key=lambda x: x[0]) 
        sorted_loaded_keypoints = list(map(lambda x: x[1], loaded_keypoints)) # sort by img number, making it a list

        # with the loaded keypoint files sorted by img number, we can now extract the keypoints and descriptors
        parsed_features = list(map(lambda x: ImageFeatures(x[0], x[1]), sorted_loaded_keypoints))
        return parsed_features
    
class CamInfoFileParser:
    @classmethod
    def parse(cls, raw_loaded_cams_info) -> list[CameraInfo]:
        cams_info = raw_loaded_cams_info["cams_info"]
        images_info = [cam_info[0][0][0] for cam_info in cams_info]
        parsed_cams = list(
            map(lambda x: CameraInfo(
                rgb_frame=x[0], depth_map=x[1], confidence_map=x[2], focal_length=x[3][0][0]
            ), images_info)
        )
        return parsed_cams
    
class ScenesDataParser:
    @classmethod
    def parse(cls, raw_loaded_kp, raw_loaded_cams_info) -> list[SceneData]:
        images_features = KPFileParser.parse(raw_loaded_kp)
        cams = CamInfoFileParser.parse(raw_loaded_cams_info)
        parsed = list(map(lambda x: SceneData(x[0], x[1]), zip(images_features, cams)))
        return parsed