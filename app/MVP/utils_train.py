import os
import random
import sys

import cv2
import numpy as np
from app_sys import AppSys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils_predict import standardize_fsize

path_current = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.split(path_current)[0])

app_sys = AppSys()

"""
Image Augmentation
"""

neffect_max = 4
param_filtering = 1
method_filtering = 'percent'
generate = True
split = True
coco2yolo = False
yolo_config_yaml = False
annotate = False
# Instance of the AppSys
app_sys = AppSys()

class Effect:
    """
    Frame modifications
    """
    def __init__(self):
        self.dict_effects = {
            'blur': self.blur,
            'sharp': self.sharp,
            }
        """
        yolo training has...
        - hsv_h
        - hsv_s
        - hsv_v
        - degrees
        - translate
        - scale
        - shear
        - perspective
        - flipud
        - fliplr
        - mixup
        - copy_paste
        - crop_fraction
        """

    def apply_effects(self, img, effect):
        """
        Apply a effect
        """
        _img = img
        _img = self.dict_effects[effect](_img)
        _img = _img.astype('uint8')
        return _img
    
    # Image sharpness
    def blur(self, img):
        size = random.randint(21, 31)
        if size % 2 == 0:
            size = size - 1
        img = cv2.blur(img, (size, size))
        return img

    def sharp(self, img):
        # Binary image
        size = random.randint(15, 21)
        # size = random.randint(15, 39)
        if size % 2 == 0:
            size = size - 1
        # size = 31
        img_bi_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bi = cv2.adaptiveThreshold(img_bi_, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_bi = cv2.morphologyEx(img_bi, cv2.MORPH_CLOSE, kernel, iterations=1)
        img_bi = cv2.morphologyEx(img_bi, cv2.MORPH_OPEN, kernel, iterations=1)
        img_bi = self.sobel(img_bi)
        img_bi = img_bi / np.max(img_bi) * 255
        img_bi = np.where(img_bi==0, 1, 1.25)
        return img_bi * img_bi_

    def sobel_horizontal(self, img):
        kernel_sobel_h = np.array([[-1, -2, -1],
                                    [0, 0, 0], 
                                    [1, 2, 1]])
        image_output = cv2.filter2D(img, -1, kernel_sobel_h)
        return image_output

    def sobel_vertical(self, img):

        kernel_sobel_v = np.array([[-1, 0, 1],
                                    [-2, 0, 2], 
                                    [-1, 0, 1]])
        image_output = cv2.filter2D(img, -1, kernel_sobel_v)
        return image_output

    def sobel(self, img):
        image_h = self.sobel_horizontal(img)
        image_v = self.sobel_vertical(img)
        image_output = np.sqrt(image_h ** 2 + image_v ** 2)
        return image_output

class SampleImage():
    """
    A class to extract random frames from a video and apply random effects to them.
    """
    def __init__(self, video, saveto) -> None:
        # Number of frames to be extracted from each video

        self.vpath = video
        self.vname = os.path.split(self.vpath)[1]
        # Depth video
        self.vpath_depth = os.path.join(app_sys.PATH_ASSET_DEPTH, f'{self.vname[:-4]}_vis.mp4')
        self.vname_depth = os.path.split(self.vpath_depth)[1]
        # Directpry to save the augmented frames
        self.saveto = saveto
        print('save to: ', self.saveto)
        os.makedirs(self.saveto, exist_ok=True)

        # Open the video and get the number of frames
        self.vidcap = cv2.VideoCapture(self.vpath)
        self.num_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.num_frames)
        
        self.vidcap_depth = cv2.VideoCapture(self.vpath_depth)
        self.num_frames_depth = int(self.vidcap_depth.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.num_frames_depth)
        

    def get_frame(self, nf, method, depth=False):
        """ Extract the nf-th frame / random frame of the video """
        # If selecting random frames, overwite nf.
        if method == 'random':
            nf = random.randint(0, self.num_frames - 1)
        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, nf)
        success, img = self.vidcap.read()

        # extract from the depth video as well.
        if depth:
            self.vidcap_depth.set(cv2.CAP_PROP_POS_FRAMES, nf)
            success_depth, img_depth = self.vidcap_depth.read()
        
            if success & success_depth:
                return img, img_depth
        else:
            return img

    def get_modified_frames(self, arg, method='percent', lis_effects=['original', 'depth', 'blur', 'sharp']):
        """
        Parameters:
            arg: - number of frames to extract per video for method == 'random'
                 - percent of the number of frames to extract per video for method == 'percent'
            methodf: 'random' or 'percent'
        """
        # Instance of Effect class
        effect = Effect()
        if method == 'random':
            if min(0, arg) < 0:
                arg = 0
            elif max(self.num_frames, arg) > self.num_frames:
                arg = self.num_frames
            list_nfs = np.arange(arg)
        else:
            if min(0, arg) < 0:
                arg = 0
            elif max(100, arg) > 100:
                arg = 100
            list_nfs = np.unique(np.linspace(0, self.num_frames-1, num=int(self.num_frames*arg/100)).astype(int))

        # Extraction
        for i in list_nfs:
            if 'depth' in lis_effects:
                img, img_depth = self.get_frame(i, method, depth=True)
                if (img is None) | (img_depth is None):
                    continue
            else:
                img = self.get_frame(i, method, depth=False)
                if img is None:
                    continue
            
            for eff in lis_effects:
                if eff == 'depth':
                    img_depth_std = standardize_fsize(img_depth, target_size=640)
                    cv2.imwrite(os.path.join(self.saveto, f"{self.vname.split('.')[0]}_{i}_{eff}.jpg"), img_depth_std)
                else:
                    if eff in 'original':
                        pass
                    else:
                        img = effect.apply_effects(img, eff)
                    img_std = standardize_fsize(img, target_size=640)
                    cv2.imwrite(os.path.join(self.saveto, f"{self.vname.split('.')[0]}_{i}_{eff}.jpg"), img_std)
                
            


def rename(lis_imgs, base=0):
    # Rename images in the list of image names
    imgs = [f for f in lis_imgs if f.endswith('.jpg')]
    imgs_shuffled = shuffle(imgs, random_state=1)
    lis_new_names = []
    for e, i in enumerate(imgs_shuffled):
        index = base + e
        img_name_new = f'{index:05d}.jpg'
        print(i, img_name_new)
        lis_new_names.append(img_name_new)
    return lis_new_names

