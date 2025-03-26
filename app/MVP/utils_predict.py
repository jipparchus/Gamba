import os
import pickle
import sys

import cv2
import numpy as np
import torch

path_current = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.split(path_current)[0])

from app_sys import AppSys

app_sys = AppSys()

"""
Others
"""

def gpu_info():
    print('CUDA version? - ', torch.__version__)
    print('CUDA available? - ', torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
        print(torch.cuda.memory_summary())
    return torch.cuda.is_available()


"""
Model Prediction
"""

def standardize_fsize(img, target_size=640):
    """
    Standardize the frame size to 640x640 pixels for YOLOv11.
    Resize while keeping the aspect ratio. Padding the frame with black pixels.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    # resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(img, (new_w, new_h))
    # Create a blank canvas with padding
    # If grayscale image
    if len(img.shape) == 2:
        padded_img = np.ones((target_size, target_size), dtype=np.uint8) * 128  # Gray padding
    else:
        padded_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 128  # Gray padding
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    padded_img[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

    return padded_img

def standardize_fsize_coord_conversion(video_path, target_size=640, coords=(0,0)):
    """
    coordinates conversion. from coords after standardising to that of the before.
    """
    x, y = coords
    video = cv2.VideoCapture(video_path)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    # New origin
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    x -= pad_left
    y -= pad_top
    x /= scale
    y /= scale
    return (round(x), round(y))


def process_frame_seg(frame, model, class2keep, conf_lvl, bg_color, isgpu):
    """
    Mask out all areas not listed in class2keep and replace them with a gray background.
    Object listed in color_class dictionary will be replaced by a single color
    Args:
        frame: Input frame (BGR)
        model: segmentation model
        class_to_keep: List of class ID to keep
        conf_lvl: Confidence level
        bg_color: Background color (BGR)
    Returns:
        Processed frame
    """
    frame = standardize_fsize(frame)
    # Run YOLO segmentation
    if isgpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
    results = model(frame, device=device)
    masks = results[0].masks.data  # Segmentation masks
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs
    
    # Initialize empty mask
    final_mask = torch.zeros_like(masks[0])
    # final_mask = masks.detach().cpu().numpy()
    # masks.detach().cpu().numpy()

    # Merge only the mask
    for i in range(len(class_ids)):
        if class_ids[i] in class2keep:
            final_mask += masks[i]
                


    # Convert mask to numpy array
    mask_np = final_mask.cpu().numpy()
    mask_np = (mask_np > conf_lvl).astype('uint8') * 255  # Binarize mask

    # Resize mask to match frame size
    mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

    # Apply mask to the image
    result = frame.copy()
    result[mask_np == 0] = bg_color  # Replace non-wall areas with gray

    return result, mask_np


def masked_video(video, model, **kwargs):
    """
    Create a video showing only selected class objects. Other area is set to be grey by default.
    Args:
        video
        model
    Kwargs:
        fourcc
        saveas
        class_to_keep: List of class ID to keep
        bg_color: Background color (BGR)
    """
    fourcc = kwargs.pop('fourcc', cv2.VideoWriter_fourcc(*'mp4v'))
    saveas = kwargs.pop('saveas', 'output_video.mp4')
    class2keep = kwargs.pop('class2keep', [0,1])
    conf_lvl = kwargs.pop('conf_lvl', 0.5)
    bg_color = kwargs.pop('bg_color', (128, 128, 128))
    isgpu = gpu_info()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(round(video.get(cv2.CAP_PROP_FPS)))
    print(f'W: {w}, H: {h}, FPS: {fps}')
    # list of gray masks
    lis_masks = []
    # processed video
    out = cv2.VideoWriter(saveas, fourcc, fps, (640, 640))
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        processed, mask_gray = process_frame_seg(frame, model, class2keep, conf_lvl, bg_color, isgpu)
        out.write(processed)
        lis_masks.append(mask_gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()
    pickle.dump(np.array(lis_masks), open(saveas.replace('mp4','pkl'), 'wb'))

def annotated_video(video, model, conf_lvl=0.5):
    isgpu = gpu_info()
    if isgpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
    return model(os.path.join(app_sys.PATH_ASSET_RAW, video), conf=conf_lvl, device=device, save=True)