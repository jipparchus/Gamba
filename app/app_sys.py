################################################################
# System Libraries                                             #
# - Manages paths to asset and other system related functions  #
################################################################
import os


class AppSys:
    def __init__(self):
        """ Project Home Path """
        self.PATH_HOME = os.path.dirname(os.path.abspath(__file__))

        """ Asset Paths """
        # All data
        self.PATH_ASSET = os.path.join(self.PATH_HOME, 'asset')
        # Raw video data
        self.PATH_ASSET_RAW = os.path.join(self.PATH_ASSET, 'raw')
        # Keypoint detection data
        self.PATH_ASSET_KP_DETECT = os.path.join(self.PATH_ASSET, 'KP_detect')
        # Keypoint detection annotated data
        self.PATH_ASSET_KP_DETECT_ANNOTATED = os.path.join(self.PATH_ASSET_KP_DETECT, 'MB2024_annotated_CVAT')
        # Hold detection data
        self.PATH_ASSET_HOLD_DETECT = os.path.join(self.PATH_ASSET, 'hold_detect')

        """ Model Paths """
        # Pre-trained
        self.PATH_MODELS = os.path.join(self.PATH_HOME, 'models')
        self.PATH_MODEL_YOLOV11_POSE = os.path.join(self.PATH_MODELS, 'yolo11n-pose.pt')
        self.PATH_MODEL_YOLOV11_SEG = os.path.join(self.PATH_MODELS, 'yolo11n-seg.pt')

        # Model for wall keypoint detection
        self.PATH_KP_DETECT = os.path.join(self.PATH_HOME, 'KP_detect')


if __name__ == '__main__':
    app_sys = AppSys()
    attrs = app_sys.__dir__()
    for attr in attrs:
        if not attr.startswith('_'):
            print(f'{attr}:    {getattr(app_sys, attr)},    Exists? - {os.path.exists(getattr(app_sys, attr))}')
