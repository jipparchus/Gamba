"""
Trim the head and tail of the video


https://stackoverflow.com/questions/68198575/how-can-i-displaymy-console-output-in-tkinter
"""

import os
import subprocess
import sys
import threading
import time
import tkinter as tk

# from dataclasses import dataclass
from logging import DEBUG, FileHandler, Formatter, getLogger

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

path_current = os.path.dirname(os.path.abspath('__file__'))
os.path.split(path_current)[0]
sys.path.append('/workspaces/MoonClimbers/app')

from app_sys import AppSys
from utils import VideoData, get_log_message

app_sys = AppSys()


"""
Logging Config
"""
formatter = Formatter('%(asctime)s -%(name)s - %(levelname)s - %(message)s')
file_handler = FileHandler('test.log')
file_handler.setFormatter(formatter)

main_logger = getLogger(__name__)
main_logger.addHandler(file_handler)
main_logger.setLevel(DEBUG)


direc_assets = app_sys.PATH_ASSET_RAW
plt.rcParams.update({'font.size': 120})
matplotlib.use("svg")
lock = threading.Lock()


class Tab1(tk.Frame):
    """
    1. Select video
    2. Trim head and tail off wall
    3. Cut
    4. If mistake, back to the top.
    5. save
    """

    def __init__(self, master=None):
        super().__init__(master)
        self.direc_assets = direc_assets
        self.direc_video = os.path.join(self.direc_assets, app_sys.Default_Video)
        self.video = VideoData(path=self.direc_video)
        self.create_frames()
        self.create_widgets()
        self.init_video_canvas()
        self.pack()


    def create_frames(self):
        # main_logger.info(get_log_message('func', 'Tab1.create_frames'))
        self.frame_top = tk.Frame(self)
        self.frame_top.pack(side=tk.TOP)
        self.frame_mid = tk.Frame(self, relief=tk.SOLID, bd=5, width=20)
        self.frame_mid.pack(side=tk.TOP)
        self.frame_btm = tk.Frame(self)
        self.frame_btm.pack(side=tk.TOP)

        """ frames in frame_top """
        self.frame_top_top = tk.Frame(self.frame_top, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_top.pack(side=tk.TOP)
        self.frame_top_btm = tk.Frame(self.frame_top, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_btm.pack(side=tk.TOP)

        """ frames in frame_mid """
        self.canvas_w = 500
        self.canvas_h = 500
        self.frame_mid_canvas = tk.Canvas(self.frame_mid, width=self.canvas_w, height=self.canvas_h)
        self.frame_mid_canvas.pack(side=tk.TOP)

        """ frames in frame_btm """
        self.frame_btm_ = tk.Frame(self.frame_btm, width=20, height=10)
        self.frame_btm_.pack(side=tk.TOP)

    def create_widgets(self):
        # main_logger.info(get_log_message('func', 'Tab1.create_widgets'))
        """
        frame_top
        """

        """ Video selection """
        frame = self.frame_top_top

        lbl = tk.Label(frame, text='Trim: ')
        lbl.grid(column=0, row=0)

        self.btn_file_select = tk.Button(frame, text='Select Video', command=self.get_video)
        self.btn_file_select.grid(column=1, row=0)

        self.lbl_direc_video = tk.Label(frame, text=f'{self.direc_video}')
        self.lbl_direc_video.grid(column=1, row=1)

        """ Play buttom & Slider """
        frame = self.frame_top_btm

        lbl = tk.Label(frame, text='start')
        lbl.grid(column=0, row=0, padx=1, pady=1)
        lbl = tk.Label(frame, text='end')
        lbl.grid(column=0, row=1, padx=1, pady=1)

        self.scl_start = tk.Scale(frame,
                                  from_=0,
                                  to=self.video.total_frames - 2,
                                  resolution=1,
                                  length=200,
                                  width=10,
                                  orient=tk.HORIZONTAL,
                                  command=self.update_sframe)
        self.scl_start.set(0)
        self.scl_start.grid(column=1, row=0, padx=1, pady=1)

        self.scl_end = tk.Scale(frame,
                                from_=1,
                                to=self.video.total_frames - 1,
                                resolution=1,
                                length=200,
                                width=10,
                                orient=tk.HORIZONTAL,
                                command=self.update_eframe)
        self.scl_end.set(self.video.total_frames - 1)
        self.scl_end.grid(column=1, row=1, padx=1, pady=1)

        lbl = tk.Label(frame, text='Save as ')
        lbl.grid(column=1, row=2, padx=2, pady=2)
        self.vname_trimmed = tk.StringVar()
        self.vname_trimmed.set(f"{self.video.path.split('.')[0]}_trimmed.mp4")
        self.txt_saveas = tk.Label(frame, textvariable=self.vname_trimmed)
        self.txt_saveas.grid(column=1, row=3, padx=1, pady=1)

        self.btn_cut = tk.Button(frame, text='Cut', command=self.cut_video)
        self.btn_cut.grid(column=1, row=4, padx=2, pady=1)

        """
        frame_btm
        """
        frame = self.frame_btm_
        """ Message Box """
        self.message = tk.Label(frame, text='Trim the head and tail of the video')
        self.message.grid(column=0, row=0)

        self.btn_depth = tk.Button(frame, text='Video-Depth-Anything', command=self.save_depth_video)
        self.btn_depth.grid(column=0, row=1)

    """
    Utils
    """

    def draw_on_canvas(self, frame):
        for thread in threading.enumerate():
            main_logger.debug(get_log_message('threading', f'{thread.name}'))
        main_logger.info(get_log_message('func', 'Tab1.draw_on_canvas'))
        frame_h, frame_w, _ = frame.shape
        ss = max(frame_h, frame_w)
        frame_h = int(frame_h / ss * self.canvas_h)
        frame_w = int(frame_w / ss * self.canvas_w)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_w, frame_h))
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        # self.frame_mid_canvas.create_image(0, 0, image=img, anchor=tk.NW)
        image_width = img.width()
        image_height = img.height()
        self.frame_mid_canvas.create_image((self.frame_mid_canvas.winfo_width() / 2 - image_width / 2), (self.frame_mid_canvas.winfo_height() / 2 - image_height / 2), image=img, anchor=tk.NW)
        self.frame_mid_canvas.image = img

    def draw_on_canvas_specific(self, ff):
        main_logger.info(get_log_message('func', 'Tab1.draw_on_canvas_specific'))
        # ret, frame = self.video.vcap.read()
        ret, frame = self.video.set_start_frame(ff)
        if ret:
            self.draw_on_canvas(frame)

    def widget_switching(self):
        for ww in [self.btn_file_select, self.btn_cut, self.scl_start, self.scl_end]:
            if ww['state'] == tk.NORMAL:
                ww['state'] = tk.DISABLED
            else:
                ww['state'] = tk.NORMAL
            main_logger.debug(get_log_message('var', f"{ww} --- {ww['state']}"))
        main_logger.info(get_log_message('func', 'Tab1.widget_switching'))

    """
    Widget action
    """

    """ Select Video """

    def get_video(self):
        thread_get_video = threading.Thread(target=self.target_get_video)
        thread_get_video.start()

    def target_get_video(self):
        # main_logger.info(get_log_message('func', 'Tab1.target_get_video'))
        # When nothing has been selected, set back to the initial values
        self.direc_video = tk.filedialog.askopenfilename(filetypes=[(self.direc_assets, '*.mp4')], title='Select a Video')
        self.lbl_direc_video.configure(text=f'{self.direc_video}')
        self.vname_trimmed.set(f"{self.direc_video.split('.')[0]}_trimmed.mp4")
        # Update info
        self.video = VideoData(path=self.direc_video)
        self.create_widgets()

    """ Play Video """

    def cut_video(self):
        self.thread_play_video = threading.Thread(target=self.target_cut_video)
        self.thread_play_video.start()

    def target_cut_video(self):
        # Disable
        self.widget_switching()
        self.save_trimed_video()
        self.widget_switching()

    def save_trimed_video(self):
        try:
            self.message.configure(text=f'{self.vname_trimmed.get()}\nSaving......')
            # Create video writer object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.vname_trimmed.get(), fourcc, self.video.fps, (self.video.width, self.video.height))
            self.video.vcap.set(cv2.CAP_PROP_POS_FRAMES, self.scl_start.get())
            for ff in range(self.scl_start.get(), self.scl_end.get()):
                ret, frame = self.video.vcap.read()
                if ret:
                    out.write(frame)
            out.release()
            self.message.configure(text=f'{self.vname_trimmed.get()}\nhas been saved!')
        except Exception as e:
            self.message.configure(text=f'{e} has occured')

    """ Set Start and End Frame """

    def update_sframe(self, event):
        # main_logger.info(get_log_message('func', 'Tab1.update_sframe'))
        thread_update_sframe = threading.Thread(target=self.target_update_sframe)
        thread_update_sframe.start()

    def update_eframe(self, event):
        # main_logger.info(get_log_message('func', 'Tab1.update_eframe'))
        thread_update_eframe = threading.Thread(target=self.target_update_eframe)
        thread_update_eframe.start()

    def scales_rules(self):
        """
        # start frame < # end frame
        """
        # main_logger.info(get_log_message('func', 'Tab1.scales_rules'))
        ss = self.scl_start.get()
        ee = self.scl_end.get()
        if ss >= ee:
            self.scl_start.set(ee - 1)
        else:
            pass

    def target_update_sframe(self):
        # main_logger.info(get_log_message('func', 'Tab1.target_update_sframe'))
        self.btn_cut['state'] = tk.DISABLED
        self.scales_rules()
        ff = self.scl_start.get()
        self.draw_on_canvas_specific(ff)
        self.btn_cut['state'] = 'normal'

    def target_update_eframe(self):
        # main_logger.info(get_log_message('func', 'Tab1.target_update_eframe'))
        self.btn_cut['state'] = 'disabled'
        self.scales_rules()
        ff = self.scl_end.get()
        self.draw_on_canvas_specific(ff)
        self.btn_cut['state'] = 'normal'

    def init_video_canvas(self):
        """
        Initialise the canvas for video display
        """
        self.img = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((self.canvas_h, self.canvas_w))))
        self.frame_mid_canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

    """
    Video Depth Estimation
    """
    def save_depth_video(self):
        self.btn_depth.config(state='disabled')
        lisdir = os.listdir(app_sys.PATH_TOOL)
        # Clone Video-Depth-Anything repository
        if 'Video-Depth-Anything' not in lisdir:
            try:
                process_clone = subprocess.Popen(['bash', './clone_vda.sh'], stdout=subprocess.PIPE)
                process_clone.wait()
                process_init = subprocess.Popen(['bash', './init_vda.sh'], stdout=subprocess.PIPE)
                process_init.wait()
            except RuntimeError:
                print('RuntimeError: Clone the Video-Depth-Anything model')
        # Use Video-Depth-Anything small or large model
        print('#'*30)
        print(f'Input video: {self.video.name}')
        print('model: vits')
        print('Start the process ...')
        try:
            t1 = time.time()
            process = subprocess.Popen(['bash', './get_depth_video.sh', self.video.name, 'vits'], stdout=subprocess.PIPE)
            for line in process.stdout:
                print(line, end='')
            process.wait()
            t2 = time.time()
            print(f'Video-Depth-Anything finished the proicess! ({round((t2-t1)/60, 1)} sec)')
            print('#'*30)
        except RuntimeError:
            print('Error in video depth estimation.')
        self.btn_depth.config(state='normal')

