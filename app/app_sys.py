################################################################
# System Libraries                                             #
# - Manages paths to asset and other system related functions  #
################################################################
import os


class AppSys:
    def __init__(self):
        """ Project Home Path """
        self.PATH_HOME = os.path.dirname(os.path.abspath(__file__))

        self.Default_Video = '1StarChoss.mp4'
        basename = self.Default_Video.split('.')[0]
        self.Default_Video_trimmed = basename + '_trimmed.mp4'
        self.Default_Video_masked = basename + '_trimmed_masked.mp4'
        self.Default_Video_kp = basename + '_trimmed_masked_kp.mp4'

        """ Asset Paths """
        self.PATH_ASSET = os.path.join(self.PATH_HOME, 'asset')
        # All data
        # Raw video data
        self.PATH_ASSET_RAW = os.path.join(self.PATH_ASSET, 'raw')
        # Depth map data
        self.PATH_ASSET_DEPTH = os.path.join(self.PATH_ASSET, 'depth')
        # Masked data
        self.PATH_ASSET_MSK = os.path.join(self.PATH_ASSET, 'masked')
        # Key points detectetion
        self.PATH_ASSET_KP = os.path.join(self.PATH_ASSET, 'kp')


        # Keypoint detection data
        self.PATH_ASSET_KP_DETECT = os.path.join(self.PATH_ASSET, 'KP_detect')
        # Keypoint detection annotated data
        self.PATH_ASSET_KP_DETECT_ANNOTATED = os.path.join(self.PATH_ASSET_KP_DETECT, 'MB2024_annotated_CVAT')
        # Hold detection data
        self.PATH_ASSET_HOLD_DETECT = os.path.join(self.PATH_ASSET, 'hold_detect')



        # Background cleaning data
        self.PATH_ASSET_PREP_MSK = os.path.join(self.PATH_ASSET, 'prep_mask')
        self.PATH_ASSET_PREP_MSK_TEMP = os.path.join(self.PATH_ASSET_PREP_MSK, 'temp')
        self.PATH_ASSET_PREP_MSK_TRAIN = os.path.join(self.PATH_ASSET_PREP_MSK, 'images', 'train')
        self.PATH_ASSET_PREP_MSK_VAL = os.path.join(self.PATH_ASSET_PREP_MSK, 'images', 'val')
        self.PATH_ASSET_PREP_MSK_LBL = os.path.join(self.PATH_ASSET_PREP_MSK, 'labels')
        self.PATH_ASSET_PREP_MSK_LBL_TRAIN = os.path.join(self.PATH_ASSET_PREP_MSK_LBL, 'train')
        self.PATH_ASSET_PREP_MSK_LBL_VAL = os.path.join(self.PATH_ASSET_PREP_MSK_LBL, 'val')
        self.PATH_ASSET_PREP_MSK_YAML = os.path.join(self.PATH_ASSET_PREP_MSK, 'data.yaml')
        # Key point detection data
        self.PATH_ASSET_PREP_KP = os.path.join(self.PATH_ASSET, 'prep_kp')
        self.PATH_ASSET_PREP_KP_TEMP = os.path.join(self.PATH_ASSET_PREP_KP, 'temp')
        self.PATH_ASSET_PREP_KP_TRAIN = os.path.join(self.PATH_ASSET_PREP_KP, 'images', 'train')
        self.PATH_ASSET_PREP_KP_VAL = os.path.join(self.PATH_ASSET_PREP_KP, 'images', 'val')
        self.PATH_ASSET_PREP_KP_LBL = os.path.join(self.PATH_ASSET_PREP_KP, 'labels')
        self.PATH_ASSET_PREP_KP_LBL_TRAIN = os.path.join(self.PATH_ASSET_PREP_KP_LBL, 'train')
        self.PATH_ASSET_PREP_KP_LBL_VAL = os.path.join(self.PATH_ASSET_PREP_KP_LBL, 'val')
        self.PATH_ASSET_PREP_KP_YAML = os.path.join(self.PATH_ASSET_PREP_KP, 'data.yaml')

        """ Model Paths """
        # Pre-trained
        self.PATH_MODELS = os.path.join(self.PATH_HOME, 'models')
        self.PATH_MODEL_YOLOV11_POSE = os.path.join(self.PATH_MODELS, 'yolo11n-pose.pt')
        self.PATH_MODEL_YOLOV11_SEG = os.path.join(self.PATH_MODELS, 'yolo11n-seg.pt')
        self.PATH_MODEL_YOLOV12_SEG = os.path.join(self.PATH_MODELS, 'yolo12n-seg.pt')

        # Model for wall keypoint detection
        self.PATH_KP_DETECT = os.path.join(self.PATH_HOME, 'KP_detect')

        self.PATH_TOOL = os.path.join(self.PATH_HOME, 'MVP')

if __name__ == '__main__':
    app_sys = AppSys()
    attrs = app_sys.__dir__()
    for attr in attrs:
        if not attr.startswith('_'):
            print(f'{attr}:    {getattr(app_sys, attr)},    Exists? - {os.path.exists(getattr(app_sys, attr))}')
