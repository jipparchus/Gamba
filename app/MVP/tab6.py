"""
Contacts on the wall
"""

import os
import pickle
import sys
import threading
import tkinter as tk

# from dataclasses import dataclass
from logging import DEBUG, FileHandler, Formatter, getLogger

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk

path_current = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.split(path_current)[0])

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



class Tab6(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.direc_assets = direc_assets
        self.direc_video = os.path.join(self.direc_assets, app_sys.Default_Video)
        self.video = VideoData(path=self.direc_video)
        self.post_init()

    def post_init(self):
        self.direc_saveas = str(self.direc_video.split('.')[0]) + '.pkl'
        self.playing = False

        self.tt = np.arange(self.video.total_frames)
        self.s0 = np.zeros_like(self.tt)
        self.s1 = np.zeros_like(self.tt)
        self.s2 = np.zeros_like(self.tt)
        self.s3 = np.zeros_like(self.tt)
        self.list_status = [self.s0, self.s1, self.s2, self.s3]

        self.create_frames()
        self.create_widgets()
        self.init_video_canvas()
        self.init_graph_canvas()
        self.pack()

    def create_frames(self):
        # main_logger.info(get_log_message('func', 'Tab1.create_frames'))
        self.frame_top = tk.Frame(self, width=380)
        self.frame_top.pack(side=tk.TOP)
        self.frame_mid = tk.Frame(self, relief=tk.SOLID, bd=5, width=380)
        self.frame_mid.pack(side=tk.TOP)
        self.frame_btm = tk.Frame(self, relief=tk.SOLID, bd=1, width=380)
        self.frame_btm.pack(side=tk.TOP)
        self.frame_footer = tk.Frame(self, relief=tk.SOLID, bd=1, width=380)
        self.frame_footer.pack(side=tk.TOP)

        """ frames in frame_top """
        self.frame_top_top = tk.Frame(self.frame_top, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_top.pack(side=tk.TOP)
        self.frame_top_btm = tk.Frame(self.frame_top, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_btm.pack(side=tk.TOP)

        """ frames in frame_mid """
        self.canvas_w_video = 380
        self.canvas_h_video = 380
        self.frame_mid_canvas = tk.Canvas(self.frame_mid, width=self.canvas_w_video, height=self.canvas_h_video)
        self.frame_mid_canvas.pack(side=tk.TOP)

        """ frames in frame_btm """
        self.canvas_w_graph = 100
        self.canvas_h_graph = 100
        self.frame_btm_canvas = tk.Canvas(self.frame_btm, width=self.canvas_w_graph, height=self.canvas_h_graph)
        self.frame_btm_canvas.pack(side=tk.TOP)

        """ frames in frame_footer """
        self.frame_footer_ = tk.Frame(self.frame_footer, relief=tk.SOLID, bd=1, width=380)
        self.frame_footer_.pack(side=tk.TOP)

    def create_widgets(self):
        # main_logger.info(get_log_message('func', 'Tab1.create_widgets'))
        """
        frame_top
        """

        """ Video selection """
        frame = self.frame_top_top

        lbl = tk.Label(frame, text='Import the video to annotate: ')
        lbl.grid(column=0, row=0)

        self.btn_file_select = tk.Button(frame, text='Select Video', command=self.get_video)
        self.btn_file_select.grid(column=1, row=0)

        self.lbl_direc_video = tk.Label(frame, text=f'{self.direc_video}')
        self.lbl_direc_video.grid(column=0, row=1)

        """ Body part selection, Playback speed & Play """
        self.frame_top_btm_left = tk.Frame(self.frame_top_btm, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_btm_left.pack(side=tk.LEFT)
        self.frame_top_btm_right = tk.Frame(self.frame_top_btm, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_btm_right.pack(side=tk.LEFT)

        frame = self.frame_top_btm_left

        self.optn_parts = [
            'Hand_L',
            'Hand_R',
            'Foot_L',
            'Foot_R'
        ]
        colors = [
            'blue',
            'orange',
            'green',
            'red'
        ]
        self.dict_cols = {key: val for key, val in zip(self.optn_parts, colors)}
        self.part_selected = tk.StringVar()
        self.part_selected.set('Hand L')
        self.drp_parts = tk.OptionMenu(frame, self.part_selected, *self.optn_parts, command=self.on_select_part)
        self.drp_parts.grid(column=0, row=0, padx=1, pady=1)
        # Initialise the parts
        self.part = 'Hand L'
        self.idx_part = 0

        lbl = tk.Label(frame, text='Playback Speed:')
        lbl.grid(column=1, row=0, padx=1, pady=1)

        self.optn_speed = [
            1.0,
            1.0,
            1.0,
            1.0
        ]
        ss = tk.DoubleVar()
        ss.set(1.0)
        self.drp_speed = tk.OptionMenu(frame, ss, *self.optn_speed)
        self.drp_speed.grid(column=3, row=0, padx=1, pady=1)
        self.drp_speed['state'] = tk.DISABLED

        self.btn_play = tk.Button(frame, text='Play', command=self.play_video)
        self.btn_play.grid(column=4, row=0)

        frame = self.frame_top_btm_right

        # Checkbox to switch on/off record mode
        self.check_record_bool = tk.BooleanVar()
        self.check_record = tk.Checkbutton(frame, text='Record', variable=self.check_record_bool, onvalue=True)
        self.check_record.grid(column=0, row=0, padx=1, pady=1)
        self.check_record_bool.set(True)

        self.scl_contacting = tk.Scale(frame,
                                       from_=0,
                                       to=1,
                                       length=40,
                                       width=10,
                                       orient=tk.HORIZONTAL)
        self.scl_contacting.set(1)
        self.scl_contacting.grid(column=0, row=1, padx=1, pady=1)

        """
        frame_btm_btm
        """
        frame = self.frame_footer_
        """ Message Box """
        self.message = tk.Label(frame, text='Save the data')
        self.message.grid(column=0, row=0)

        self.btn_save = tk.Button(frame, text='Save', command=self.save_data)
        self.btn_save.grid(column=0, row=1)

    """
    Utils
    """

    def draw_on_canvas(self, frame):
        for thread in threading.enumerate():
            main_logger.debug(get_log_message('threading', f'{thread.name}'))
        main_logger.info(get_log_message('func', 'Tab1.draw_on_canvas'))
        frame_h, frame_w, _ = frame.shape
        ss = max(frame_h, frame_w)
        frame_h = int(frame_h / ss * self.canvas_h_video)
        frame_w = int(frame_w / ss * self.canvas_w_video)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_w, frame_h))
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        # self.frame_mid_canvas.create_image(0, 0, image=img, anchor=tk.NW)
        image_width = img.width()
        image_height = img.height()
        self.frame_mid_canvas.create_image((self.frame_mid_canvas.winfo_width() / 2 - image_width / 2), (self.frame_mid_canvas.winfo_height() / 2 - image_height / 2), image=img, anchor=tk.NW)
        self.frame_mid_canvas.image = img

    """
    Widget action
    """

    """ Select Video """

    def get_video(self):
        thread_get_video_t2 = threading.Thread(target=self.target_get_video)
        thread_get_video_t2.start()

    def target_get_video(self):
        # main_logger.info(get_log_message('func', 'Tab1.target_get_video'))
        # When nothing has been selected, set back to the initial values
        self.direc_video = tk.filedialog.askopenfilename(filetypes=[(self.direc_assets, '*.mp4')], title='Select a Video')
        self.lbl_direc_video.configure(text=f'{self.direc_video}')
        # Update info
        self.video = VideoData(path=self.direc_video)

        for widget in self.winfo_children():
            print(widget)
            widget.destroy()

        self.post_init()

    """ Part selection """

    def on_select_part(self, event):
        """
        Show the plot of the contact state of the part
        """
        self.part = self.part_selected.get()
        self.idx_part = self.optn_parts.index(self.part)

    """ Playback """

    def play_video(self):
        if not self.playing:
            # self.btn_play['state'] = tk.DISABLED
            self.playing = True
            thread_play_video_t2 = threading.Thread(target=self.target_play_video)
            thread_play_video_t2.start()
            # self.btn_play['state'] = tk.NORMAL

    def target_play_video(self):
        self.update_frames()

    def update_frames(self):
        ret, frame = self.video.vcap.read()
        frame_number = int(self.video.vcap.get(cv2.CAP_PROP_POS_FRAMES))
        if self.check_record_bool.get():
            self.record_contact(frame_number - 1)
        if ret:
            self.draw_on_canvas(frame)
            self.update_graph_plot(self.idx_part, frame_number - 1)
            self.frame_mid.after(int(1000 / self.video.fps), self.update_frames)
        else:
            main_logger.info(get_log_message('info', 'Video playback completed'))
            self.playing = False
            self.summary_graph_plot()
            self.video.vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def init_video_canvas(self):
        """
        Initialise the canvas for video display
        """
        self.img = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((380, 380))))
        self.frame_mid_canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.plot = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((380, 100))))
        self.frame_btm_canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

    """ Record the state of the parts """

    def record_contact(self, frame_number):
        self.list_status[self.idx_part][frame_number] = self.scl_contacting.get()

    """ Graph plots """

    def init_graph_canvas(self):
        """
        Initialise the canvas for contact plot display
        """
        self.fig, self.ax = plt.subplots(figsize=(38, 8), tight_layout={"pad": 0.5, "h_pad": 0, "w_pad": 0}, dpi=10)

        for s, name in zip(self.list_status, self.optn_parts):
            self.ax.plot(self.tt, s, label=name, lw=20)
        self.ax.set_xlim(1, self.video.total_frames)
        self.ax.set_ylim((-0.05, 1.05))
        self.ax.set_ylabel('State')
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=80, handletextpad=1, borderaxespad=0, borderpad=0)
        cc = FigureCanvasTkAgg(self.fig, self.frame_btm_canvas)
        cc.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # cc.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # cc.draw()

    def summary_graph_plot(self):
        self.ax.clear()
        self.ax.set_xlim((0, self.video.total_frames))
        self.ax.set_ylim((-0.05, 1.05))
        self.ax.set_ylabel('State')
        for s, name in zip(self.list_status, self.optn_parts):
            self.ax.plot(self.tt, s, label=name, lw=20)
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=80, borderaxespad=0, borderpad=0)
        self.fig.canvas.draw()

    def update_graph_plot(self, idx_part, frame_number):
        self.ax.clear()
        self.ax.set_xlim((0, self.video.total_frames))
        self.ax.set_ylim((-0.05, 1.05))
        self.ax.set_ylabel('State')
        # self.ax.plot(self.tt[:frame_number], self.list_status[idx_part][:frame_number], label=self.optn_parts[idx_part], lw=2)
        self.ax.plot(self.tt, self.list_status[idx_part], color=self.dict_cols[self.optn_parts[idx_part]], label=self.optn_parts[idx_part], lw=20)
        self.ax.axvline(x=frame_number, ymin=0, ymax=1, lw=10, color='black')
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=80, borderaxespad=0, borderpad=0)
        self.fig.canvas.draw()

    """ Save the recorded data """

    def save_data(self):
        dict_record = {key: val for key, val in zip(self.optn_parts, self.list_status)}
        with open(self.direc_saveas, mode='wb') as f:
            pickle.dump(dict_record, f)
        self.message.configure(text=self.direc_saveas)
