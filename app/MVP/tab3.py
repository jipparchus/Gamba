"""
Sampling frames, add effects, annotate, and train a model for the video to support annotation

Can add videos later.
Train models per video for the annotation.
labels and traing/validation images are accumulated in the same directory.
So the later, videos will be annotated based on the previous annotations.
"""

import os
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
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

path_current = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.split(path_current)[0])

from app_sys import AppSys
from utils import AnnotationObjects, VideoData, direc_exist_check, init_yolo_config
from utils_predict import masked_video
from utils_train import SampleImage, rename

app_sys = AppSys()


"""
Logging Config
"""
formatter = Formatter('%(asctime)s -%(name)s - %(levelname)s - %(message)s')
file_handler = FileHandler('tab2.log')
file_handler.setFormatter(formatter)

main_logger = getLogger(__name__)
main_logger.addHandler(file_handler)
main_logger.setLevel(DEBUG)


plt.rcParams.update({'font.size': 120})
matplotlib.use("svg")
lock = threading.Lock()



class Tab3(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.direc_assets = app_sys.PATH_ASSET_RAW
        self.saveto = app_sys.PATH_ASSET_PREP_MSK_TEMP
        direc_exist_check(self.saveto)
        self.listup_imgs()
        self.direc_video = os.path.join(self.direc_assets, app_sys.Default_Video)
        self.video = VideoData(path=self.direc_video)
        self.post_init()


    def post_init(self):
        # self.tt = np.arange(self.video.total_frames)
        # self.s0 = np.zeros_like(self.tt)
        # self.s1 = np.zeros_like(self.tt)
        # self.s2 = np.zeros_like(self.tt)
        # self.s3 = np.zeros_like(self.tt)
        # self.list_status = [self.s0, self.s1, self.s2, self.s3]

        self.create_frames()
        self.create_widgets()
        self.init_video_canvas()
        self.pack()

    def create_frames(self):
        # main_logger.info(get_log_message('func', 'Tab1.create_frames'))
        self.frame_top = tk.Frame(self, width=800)
        self.frame_top.pack(side=tk.TOP)
        self.frame_mid = tk.Frame(self, relief=tk.SOLID, bd=5, width=800)
        self.frame_mid.pack(side=tk.TOP)
        self.frame_btm = tk.Frame(self, relief=tk.FLAT, bd=1, width=800)
        self.frame_btm.pack(side=tk.TOP)

        """ frames in frame_top """
        self.frame_top_top = tk.Frame(self.frame_top, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_top.pack(side=tk.TOP)
        self.frame_top_btm = tk.Frame(self.frame_top, relief=tk.FLAT, bd=5, width=20)
        self.frame_top_btm.pack(side=tk.TOP)
        self.frame_top_btm_left = tk.Frame(self.frame_top_btm, relief=tk.FLAT, bd=5, width=20)
        self.frame_top_btm_left.pack(side=tk.LEFT)
        self.frame_top_btm_right = tk.Frame(self.frame_top_btm, relief=tk.FLAT, bd=5, width=20)
        self.frame_top_btm_right.pack(side=tk.LEFT)


        """ frames in frame_mid """
        self.frame_mid_top = tk.Frame(self.frame_mid, relief=tk.RIDGE, bd=5, width=20)
        self.frame_mid_top.pack(side=tk.TOP)
        self.canvas_w_video = 500
        self.canvas_h_video = 500
        self.frame_mid_canvas = tk.Canvas(self.frame_mid, width=self.canvas_w_video, height=self.canvas_h_video)
        self.frame_mid_canvas.pack(side=tk.TOP)


        """ frames in frame_btm """
        self.frame_btm_left = tk.Frame(self.frame_btm, relief=tk.SOLID, bd=1, width=100)
        self.frame_btm_left.pack(side=tk.LEFT)

        self.frame_btm_right = tk.Frame(self.frame_btm, relief=tk.SOLID, bd=2, width=600)
        self.frame_btm_right.pack(side=tk.LEFT)

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

        """ Sampling method selection, Parameter, Effects & Sample """
        self.frame_top_btm_left_top = tk.Frame(self.frame_top_btm_left, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_btm_left_top.pack(side=tk.TOP)
        self.frame_top_btm_left_btm = tk.Frame(self.frame_top_btm_left, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_btm_left_btm.pack(side=tk.TOP)

        frame = self.frame_top_btm_left_top

        self.frame_top_btm_left_top_left = tk.Frame(frame, relief=tk.RIDGE, bd=5, width=20)
        self.frame_top_btm_left_top_left.pack(side=tk.LEFT)
        self.frame_top_btm_left_top_right = tk.Frame(frame, relief=tk.RIDGE, bd=5, width=10)
        self.frame_top_btm_left_top_right.pack(side=tk.LEFT)

        frame = self.frame_top_btm_left_top_left
        self.optn_sample_mode = [
            'Random',
            'Percentage'
        ]

        self.sample_mode_selected = tk.StringVar()
        self.sample_mode_selected.set('Random')
        self.drp_parts = tk.OptionMenu(frame, self.sample_mode_selected, *self.optn_sample_mode)
        self.drp_parts.grid(column=0, row=0, padx=1, pady=1)
        self.idx_mode = 0

        lbl = tk.Label(frame, text='Param:')
        lbl.grid(column=1, row=0, padx=1, pady=1)
        self.sample_param = tk.Spinbox(frame, from_=1, to=100, width=3)
        self.sample_param.grid(column=2, row=0, padx=1, pady=1)
        
        self.effect_blur = tk.BooleanVar()
        self.effect_sharp = tk.BooleanVar()
        lbl = tk.Label(frame, text='Effects:')
        lbl.grid(column=0, row=1, padx=1, pady=1)
        lbl = tk.Label(frame, text='Blur')
        lbl.grid(column=1, row=1, padx=1, pady=1)
        self.checkbox_1 = tk.Checkbutton(frame, onvalue=1, offvalue=0, variable=self.effect_blur)
        self.checkbox_1.grid(column=2, row=1)
        lbl = tk.Label(frame, text='Sharp')
        lbl.grid(column=3, row=1, padx=1, pady=1)
        self.checkbox_2 = tk.Checkbutton(frame, onvalue=1, offvalue=0, variable=self.effect_sharp)
        self.checkbox_2.grid(column=4, row=1)

        frame = self.frame_top_btm_left_top_right
        self.btn_sample = tk.Button(frame, text='Sample', height=5, width=10, command=self.sample_frames)
        # self.btn_sample.grid(column=1, row=0)
        self.btn_sample.pack()

        frame = self.frame_top_btm_left_btm

        # self.btn_sample = tk.Button(frame, text='Sample', height=10, width=10, command=self.sample_frames)
        # self.btn_sample.grid(column=1, row=0)
        self.btn_load_anno = tk.Button(frame, text='Load Annotation', command=self.load_saved_annotation)
        self.btn_load_anno.grid(column=0, row=0)
        self.btn_load_anno.config(state=tk.DISABLED)
        self.btn_new_anno = tk.Button(frame, text='Create New Annotation', command=self.create_annotation, background='yellow')
        self.btn_new_anno.grid(column=2, row=0)

        """ List of nodes """
        self.frame_top_btm_right_1 = tk.Frame(self.frame_top_btm_right, relief=tk.FLAT, bd=5, width=20)
        self.frame_top_btm_right_1.pack(side=tk.TOP)
        frame = self.frame_top_btm_right_1

        self.listbox_title = tk.StringVar()
        self.listbox_title.set('List of Points')

        lbl = tk.Label(frame, textvariable=self.listbox_title)
        lbl.pack()

        self.frame_top_btm_right_2 = tk.Frame(self.frame_top_btm_right, relief=tk.FLAT, bd=5, width=20)
        self.frame_top_btm_right_2.pack(side=tk.TOP)
        frame = self.frame_top_btm_right_2

        self.listbox_nodes = tk.Listbox(frame, width=20, height=5)
        self.listbox_nodes.pack(side=tk.LEFT)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox_nodes.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox_nodes.yview)
        # When a node is selected
        self.listbox_nodes.bind('<<ListboxSelect>>', self.node_select)

        self.frame_top_btm_right_3 = tk.Frame(self.frame_top_btm_right, relief=tk.FLAT, bd=5, width=20)
        self.frame_top_btm_right_3.pack(side=tk.TOP)
        frame = self.frame_top_btm_right_3

        self.btn_delete_node_all = tk.Button(frame, text='Delete All', command=self.node_delete_all, background='red')
        self.btn_delete_node_all.pack(side=tk.LEFT)
        self.btn_delete_node = tk.Button(frame, text='Delete', command=self.node_delete)
        self.btn_delete_node.pack(side=tk.LEFT)


        """
        frame_mid
        """
        frame = self.frame_mid_top

        self.btn_pre = tk.Button(frame, text='Previous', bg='orange', command=self.frame_previous)
        self.btn_pre.grid(column=0, row=0)
        self.btn_next = tk.Button(frame, text='Next', bg='lightgreen', command=self.frame_next)
        self.btn_next.grid(column=1, row=0)
        # lbl = tk.Label(frame, text='Fr')
        # lbl.grid(column=2, row=0)
        self.nframe = tk.StringVar()
        nframe = 'Sample Frames!'
        self.listup_imgs()
        if len(self.imgs2annotate_original) > 0:
            nframe = f'1 / {len(self.imgs2annotate_original)}'
            self.draw_on_canvas(cv2.imread(os.path.join(self.saveto, self.imgs2annotate_original[0])))
        self.nframe.set(nframe)
        self.nframe_lbl = tk.Label(frame, textvariable=self.nframe, width=15)
        self.nframe_lbl.grid(column=3, row=0)

        self.optn_class_lbl = [
            'wall',
            'climber'
        ]

        lbl = tk.Label(frame, text='Label: ')
        lbl.grid(column=4, row=0)
        self.class_lbl_selected = tk.StringVar()
        self.class_lbl_selected.set('wall')
        self.class_lbl = tk.OptionMenu(frame, self.class_lbl_selected, *self.optn_class_lbl, command=self.class_lbl_onselection)
        self.class_lbl.grid(column=5, row=0)

        self.btn_annotation = tk.Button(frame, text='Start Annotation', bg='lightblue', command=self.start_annotation_thread)
        self.btn_annotation.grid(column=6, row=0)

        """
        frame_btm_left
        """
        frame = self.frame_btm_left
        """ Message Box """
        self.message = tk.Label(frame, text='Save the annotation')
        self.message.grid(column=0, row=0)
        self.btn_save = tk.Button(frame, text='Save Annotation', command=self.save_data, foreground='red')
        self.btn_save.grid(column=0, row=1)

        """
        frame_btm_right
        """
        frame = self.frame_btm_right

        self.lis_yolo_model = [
            'yolov11n-seg',
            'YOLO12-seg'
        ]
        self.yolo_model_selected = tk.StringVar()
        self.yolo_model_selected.set(self.lis_yolo_model[0])
        self.optn_yolo_model = tk.OptionMenu(frame, self.yolo_model_selected, *self.lis_yolo_model)
        self.optn_yolo_model.grid(column=0, row=0)

        # Epochs
        lbl = tk.Label(frame, text='Epochs:')
        lbl.grid(column=1, row=0)
        self.epochs = tk.IntVar(value=20)
        # self.epochs.set(20)
        self.epochs_spbx = tk.Spinbox(frame, from_=1, to=100, width=3, textvariable=self.epochs)
        self.epochs_spbx.grid(column=2, row=0)
        # Epochs
        lbl = tk.Label(frame, text='Batch:')
        lbl.grid(column=3, row=0)
        self.batch = tk.IntVar(value=2)
        # self.batch.set(2)
        self.batch_spbx = tk.Spinbox(frame, from_=1, to=100, width=3, textvariable=self.batch)
        self.batch_spbx.grid(column=4, row=0)

        self.optn_device = [
            'cpu',
            'cuda'
        ]
        self.device_selected = tk.StringVar()
        self.device_selected.set('cuda')
        self.drp_device = tk.OptionMenu(frame, self.device_selected, *self.optn_device)
        self.drp_device.grid(column=5, row=0)

        self.btn_train = tk.Button(frame, text='Train', command=self.train_model, background='red', foreground='white')
        self.btn_train.grid(column=6, row=0)

        # Which trained model to use for prediction
        self.lis_trained_model = list(os.listdir('./runs/segment/'))
        self.trained_model_selected = tk.StringVar()
        self.trained_model_selected.set(self.lis_trained_model[0])
        self.optn_trained_model = tk.OptionMenu(frame, self.trained_model_selected, *self.lis_trained_model)
        self.optn_trained_model.grid(column=0, row=1)

        self.btn_predict = tk.Button(frame, text='Predict', command=self.predict, background='blue', foreground='white')
        self.btn_predict.grid(column=6, row=1)

    """
    Utils
    """

    def draw_on_canvas(self, frame):
        frame_h, frame_w, _ = frame.shape
        ss = max(frame_h, frame_w)
        frame_h = int(frame_h / ss * self.canvas_h_video)
        frame_w = int(frame_w / ss * self.canvas_w_video)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.img_resized = cv2.resize(frame, (frame_w, frame_h))
        img = ImageTk.PhotoImage(image=Image.fromarray(self.img_resized))
        image_width = img.width()
        image_height = img.height()
        self.frame_mid_canvas.create_image((self.frame_mid_canvas.winfo_width() / 2 - image_width / 2), (self.frame_mid_canvas.winfo_height() / 2 - image_height / 2), image=img, anchor=tk.NW)
        self.frame_mid_canvas.image = img

    def draw_on_canvas_specific(self, ff):
        # ret, frame = self.video.vcap.read()
        ret, frame = self.video.set_start_frame(ff)
        if ret:
            self.draw_on_canvas(frame)

    """
    Widget action
    """

    """ Select Video """

    def get_video(self):
        thread_get_video_t2 = threading.Thread(target=self.target_get_video)
        thread_get_video_t2.start()

    def target_get_video(self):
        # When nothing has been selected, set back to the initial values
        self.direc_video = tk.filedialog.askopenfilename(filetypes=[(self.direc_assets, '*.mp4')], title='Select a Video')
        self.lbl_direc_video.configure(text=f'{self.direc_video}')
        # Update info
        self.video = VideoData(path=self.direc_video)
        self.draw_on_canvas_specific(1)

    """ Sampling Frames """

    def sample_frames(self):
        """
        Create blurred, sharpened and the original images from the selected video.
        """
        lis_effects = ['original']
        for i, j in zip([self.effect_blur.get(), self.effect_sharp.get()], ['blur', 'sharp']):
            if i:
                lis_effects.append(j)
        if self.direc_video.endswith('.mp4'):
            sample_ = SampleImage(os.path.join(app_sys.PATH_ASSET_RAW, self.direc_video), self.saveto)
            sample_.get_modified_frames(int(self.sample_param.get()), method=self.sample_mode_selected.get(), lis_effects=lis_effects)
        
        self.listup_imgs()
        self.nframe.set(f'1 / {len(self.imgs2annotate_original)}')
                
    def load_saved_annotation(self):
        print('load saved annotation')

    def create_annotation(self):
        """
        Create dictionary for annotation fraph data for each images
        """
        self.dict_annotation = {e: AnnotationObjects(img, self.optn_class_lbl) for e, img in enumerate(self.imgs2annotate_original)}
        
    def listup_imgs(self):
        """
        Only list up the original images. (Blurred and the sharpened images are annotated based on the original image's annotation)
        """
        
        print(os.listdir(self.saveto))
        self.imgs2annotate = sorted([dd for dd in os.listdir(self.saveto) if dd.endswith('.jpg')])
        # self.imgs2annotate = sorted([os.path.basename(dd) for dd in os.listdir(self.saveto) if dd.endswith('.png')])
        self.imgs2annotate_original = [i for i in self.imgs2annotate if 'original' in i]
        
    """ Move to different frames """

    def frame_previous(self):
        nframe = int(self.nframe.get().split(' / ')[0])
        nframe -= 1
        self.nframe_check(nframe)
        nframe = int(self.nframe.get().split(' / ')[0])
        print(os.path.join(self.saveto, self.imgs2annotate_original[nframe-1]))
        self.draw_on_canvas(cv2.imread(os.path.join(self.saveto, self.imgs2annotate_original[nframe-1])))
        self.update_field()
    
    def frame_next(self):
        nframe = int(self.nframe.get().split(' / ')[0])
        nframe += 1
        self.nframe_check(nframe)
        nframe = int(self.nframe.get().split(' / ')[0])
        print(os.path.join(self.saveto, self.imgs2annotate_original[nframe-1]))
        self.draw_on_canvas(cv2.imread(os.path.join(self.saveto, self.imgs2annotate_original[nframe-1])))
        self.update_field()
        

    def nframe_check(self, nframe:int):
        # Check the current frame number
        if nframe <= 0:
            nframe = 1
        elif nframe > len(self.imgs2annotate_original):
            nframe = len(self.imgs2annotate_original)
        
        # Current image to be annotated
        self.img2annotate_idx = nframe - 1
        self.img2annotate = self.imgs2annotate_original[nframe - 1]
        self.nframe.set(f'{nframe} / {len(self.imgs2annotate_original)}')

    def class_lbl_onselection(self):
        self.update_field()

    def init_video_canvas(self):
        """
        Initialise the canvas for video display
        """
        self.img = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((self.canvas_h_video, self.canvas_w_video))))
        self.frame_mid_canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
    
    def start_annotation_thread(self):
        thread_start_annotation = threading.Thread(target=self.start_annotation)
        thread_start_annotation.start()

    def start_annotation(self):
        is_ready = 'dict_annotation' in self.__dir__()
        if is_ready:
        # Change the mouse cusor
            self.frame_mid_canvas.bind("<Enter>", lambda event: self.frame_mid_canvas.config(cursor="crosshair"))
            self.frame_mid_canvas.bind("<Leave>", lambda event: self.frame_mid_canvas.config(cursor=""))
            # Left click to place a node
            self.frame_mid_canvas.bind("<Button-1>", self.on_lclick)
            # Double left click to select the node -> move or delete
            self.frame_mid_canvas.bind("<B1-Motion>", self.on_hold)
            # Press Entre to complete the polygon
            self.frame_mid_canvas.bind("<Button-3>", self.complete_polygon)

            self.frame_mid_canvas.focus_set()

            self.annotation_class = self.optn_class_lbl.index(self.class_lbl_selected.get())

            # self.list_nodes_yolo = []
    def update_field(self):
        """
        As move to another image to be annotated, load the existing annotations and
            - refresh the listbox
            - refresh the canvas
        """
        # Remove items from the listbox
        self.listbox_nodes.delete(0, tk.END)
        # load the nodes and edges
        if 'dict_annotation' in self.__dir__():
            self.edge2line()

    def edge2line(self):
        self.frame_mid_canvas.delete('all')
        self.draw_on_canvas(cv2.imread(os.path.join(self.saveto, self.img2annotate)))

        edges = self.dict_annotation[self.img2annotate_idx].get_all_edges(self.class_lbl_selected.get())
        lis_nodes = self.dict_annotation[self.img2annotate_idx].get_node_coords_all(self.class_lbl_selected.get())
        
        
        
        
        
        dict_node_coords = {name: data['data'] for name, data in lis_nodes}
        
        
        
        
        
        for edge in edges:
            p0_name, p1_name = edge
            
            p0_coords = dict_node_coords[p0_name]
            p1_coords = dict_node_coords[p1_name]
            self.frame_mid_canvas.create_line(p0_coords[0], p0_coords[1], p1_coords[0], p1_coords[1], fill='red', width=2, tag=f'{p0_name}_{p1_name}')

            for p_name, p_coords in dict_node_coords.items():
                self.frame_mid_canvas.create_oval(p_coords[0]-self.marker_radius, p_coords[1]-self.marker_radius,
                                                p_coords[0]+self.marker_radius, p_coords[1]+self.marker_radius,
                                                outline='red', width=2, tag=f"n{p_name}")


    def check_canvas_object(self):
        # Get the latest coordinates of the nodes with the object tag.
        cls = self.class_lbl_selected.get()

        dict_temp = {}
        print(self.frame_mid_canvas.find_all())
        for obj_id in self.frame_mid_canvas.find_all():
            tag = list(self.frame_mid_canvas.gettags(obj_id))
            if len(tag) > 0:
                tag = tag[0]
                if ('current' not in tag) & ('n' in tag):
                    coords = self.frame_mid_canvas.coords(obj_id)
                    xx = np.mean([coords[0], coords[2]]).astype(int)
                    yy = np.mean([coords[1], coords[3]]).astype(int)
                    dict_temp[tag] = (xx, yy)
                    self.dict_annotation[self.img2annotate_idx].update_node(cls, tag[1:], (xx, yy))
        # nframe, class_class_lbl, nodes
        self.listbox_title.set(f"Imgae {self.nframe.get().split(' / ')[0]} Class {self.annotation_class} Points")
        self.listbox_nodes.delete(0, tk.END)
        for node, xy in dict_temp.items():
            self.listbox_nodes.insert(tk.END, f'Point {node[1:]}: {xy}')

        

    def on_lclick(self, event):
        """
        Check if the click is held after certain time duration.
        If still held, do not treat as a click, but as a hold
        """
        threshold_duration = 200    # milliseconds
        self.is_hold = False
        self.master.after(threshold_duration, self.get_click_coords, event)
        
 

    def get_click_coords(self, event):
        """
        L-click to select nodes of polygon
        """
        self.marker_radius = 6
        if not self.is_hold:
            # self.frame_mid_canvas.create_oval(event.x-self.marker_radius, event.y-self.marker_radius,
            #                                 event.x+self.marker_radius, event.y+self.marker_radius,
            #                                 outline='red',
            #                                 width=2,
            #                                 tag=f"n{len([j['data'] for i, j in self.dict_annotation[self.img2annotate_idx].get_node_coords_all(self.class_lbl_selected.get())])}"
            #                                 )

            cls = self.class_lbl_selected.get()
            nn = str(len(self.dict_annotation[self.img2annotate_idx].get_node_coords_all(cls)))
            self.dict_annotation[self.img2annotate_idx].add_node(cls, nn, (event.x, event.y))

            self.edge2line()



            
            # self.list_nodes_yolo += [event.x/self.canvas_w_video, event.y/self.canvas_h_video]
            print([j['data'] for i, j in self.dict_annotation[self.img2annotate_idx].get_node_coords_all(self.class_lbl_selected.get())])
            
            if len([j['data'] for i, j in self.dict_annotation[self.img2annotate_idx].get_node_coords_all(self.class_lbl_selected.get())]) > 1:      
                nbr = self.dict_annotation[self.img2annotate_idx].find_neighbours(cls, nn)[0]
                # Tag for the edges on tkinter canvas
                tag = self.dict_annotation[self.img2annotate_idx].find_adj_edges(cls, nn)[nbr]['tag']

                p0 = self.dict_annotation[self.img2annotate_idx].get_node_coords(cls, nbr)
                p1 = self.dict_annotation[self.img2annotate_idx].get_node_coords(cls, nn)
                self.frame_mid_canvas.create_line(p0[0], p0[1], p1[0], p1[1], fill='red', width=2, tag=tag)
                # self.list_edges.append((p0, p1))
            
            self.check_canvas_object()
            

    def on_hold(self, event):
        """
        L-click hold to move the selected node
        """
        max_distance = self.marker_radius*3
        self.is_hold = True
        if len([j['data'] for i, j in self.dict_annotation[self.img2annotate_idx].get_node_coords_all(self.class_lbl_selected.get())]) > 0:
            # Find the closest node to the current coordinates
            distance = np.linalg.norm(np.array([j['data'] for i, j in self.dict_annotation[self.img2annotate_idx].get_node_coords_all(self.class_lbl_selected.get())]) - np.array((event.x, event.y)), axis=1)
            idx_min = np.argmin(distance)
            if distance[idx_min] <= max_distance:
                self.frame_mid_canvas.moveto(f'n{idx_min}', event.x-self.marker_radius, event.y-self.marker_radius)
                [j['data'] for i, j in self.dict_annotation[self.img2annotate_idx].get_node_coords_all(self.class_lbl_selected.get())][idx_min] = (event.x, event.y)
                self.update_lines(idx_min)
        
        self.check_canvas_object()

    def update_lines(self, idx_node):

        p1 = self.dict_annotation[self.img2annotate_idx].get_node_coords(self.class_lbl_selected.get(), str(idx_node))
        dict_adj = self.dict_annotation[self.img2annotate_idx].find_adj_edges(self.class_lbl_selected.get(), str(idx_node))
        nbrs = self.dict_annotation[self.img2annotate_idx].find_neighbours(self.class_lbl_selected.get(), str(idx_node))
        for nbr in nbrs:
            p0 = self.dict_annotation[self.img2annotate_idx].get_node_coords(self.class_lbl_selected.get(), nbr)
            tag = dict_adj[nbr]['tag']
            self.frame_mid_canvas.delete(tag)
            self.frame_mid_canvas.create_line(p0[0], p0[1], p1[0], p1[1], fill='red', width=2, tag=tag)

        
    def complete_polygon(self, event):
        """
        R-click to complete the polygon
        """
        if (len([j['data'] for i, j in self.dict_annotation[self.img2annotate_idx].get_node_coords_all(self.class_lbl_selected.get())])) < 2:
            pass
        else:
            # Connect the last and the first nodes
            lis_nodes = self.dict_annotation[self.img2annotate_idx].get_node_coords_all(self.class_lbl_selected.get())
            p0_name, p0_data = lis_nodes[0]
            p1_name, p1_data = lis_nodes[-1]
            self.dict_annotation[self.img2annotate_idx].close_polygon(self.class_lbl_selected.get())
            self.frame_mid_canvas.create_line(p0_data['data'][0], p0_data['data'][1], p1_data['data'][0], p1_data['data'][1], fill='red', width=2, tag=f'{p0_name}_{p1_name}')
            # self.list_edges.append((p0_data['data'], p1_data['data']))
        
        self.check_canvas_object()

    def node_select(self, event):
        # Find what node waws selected in the ListBox
        idx_selected = self.listbox_nodes.curselection()
        if idx_selected:
            self.node_selected = self.listbox_nodes.get(idx_selected)

        for obj_id in self.frame_mid_canvas.find_all():
            if len(list(self.frame_mid_canvas.gettags(obj_id))) > 0:
                if f"n{self.node_selected.split(':')[0].split(' ')[1]}" == list(self.frame_mid_canvas.gettags(obj_id))[0]:
                    self.frame_mid_canvas.itemconfig(obj_id, outline='red', width=7)
                elif 'n' in list(self.frame_mid_canvas.gettags(obj_id))[0]:
                    self.frame_mid_canvas.itemconfig(obj_id, outline='red', width=2)

    def node_delete(self):
        self.frame_mid_canvas.delete(f"n{self.node_selected.split(':')[0].split(' ')[1]}")
        dict_adj =  self.dict_annotation[self.img2annotate_idx].find_adj_edges(self.class_lbl_selected.get(), self.node_selected.split(':')[0].split(' ')[1])
        print(dict_adj)
        # Remove lines connected to the neighbours
        xy = []
        nodes = []

        # Assume that there are maximum two neighbours for polygon annotation
        for node, data in dict_adj.items():
            xx, yy = self.dict_annotation[self.img2annotate_idx].get_node_coords(self.class_lbl_selected.get(), node)
            xy.append([xx, yy])
            nodes.append(node)
            self.frame_mid_canvas.delete(data['tag'])

        # If two neighbours, 
        if len(dict_adj.items()) == 2:
            self.frame_mid_canvas.create_line(xy[0][0], xy[0][1], xy[1][0], xy[1][1], fill='red', width=2, tag=f'{nodes[0]}_{nodes[1]}')
        self.dict_annotation[self.img2annotate_idx].remove_node(self.class_lbl_selected.get(), self.node_selected.split(':')[0].split(' ')[1])
        self.check_canvas_object()


        # Clear selection
        self.listbox_nodes.selection_clear(0, tk.END)

    def node_delete_all(self):
        self.frame_mid_canvas.delete('all')
        self.draw_on_canvas(cv2.imread(os.path.join(self.saveto, self.img2annotate)))
        self.dict_annotation[self.img2annotate_idx].reset_graph()


    """ Save the recorded data """
    def save_data(self):
        thread_save = threading.Thread(target=self.save_data_thread)
        thread_save.start()

    def save_data_thread(self):
        try:
            
            print('#'*50)
            for nf in range(len(self.imgs2annotate_original)):
                img_original = self.imgs2annotate_original[nf].split('.')[0]
                # Apply the same annotation for the images with efffects
                img_basename = img_original.replace('original', '')
                lis_imgs = [i.split('.')[0] for i in self.imgs2annotate if (img_basename in i)]
                print('*' * 10)
                nx, ny, _ = self.img_resized.shape
                yolo_label = ''
                for e, cls in enumerate(self.optn_class_lbl):
                    if len(self.dict_annotation[nf].get_node_coords_all(cls)) > 0:
                        yolo_label += f'{e}'
                    for (node, dict_data) in self.dict_annotation[nf].get_node_coords_all(cls):
                        xx, yy = dict_data['data']
                        yolo_label += f' {xx/nx} {yy/ny}'
                    if e < len(self.optn_class_lbl) - 1:
                        yolo_label += '\n'
                print(yolo_label)
                direc_exist_check(app_sys.PATH_ASSET_PREP_MSK_LBL)
                # Save the same annotations for the originl and modified images
                print('Saved for:\n')
                for img in lis_imgs:
                    print(img)
                    with open(os.path.join(app_sys.PATH_ASSET_PREP_MSK_LBL, img + '.txt'), 'w') as f:
                        f.write(yolo_label)
            # Create data.yaml for trainig
            with open(os.path.join(app_sys.PATH_ASSET_PREP_MSK, 'data.yaml'), 'w') as f:
                f.write('names:\n')
                for cls in self.optn_class_lbl:
                    f.write(f'- {cls}\n')
                f.write(f'nc: {len(self.optn_class_lbl)}\n')
                f.write(f'path: {app_sys.PATH_ASSET_PREP_MSK}\n')
                f.write(f'train: {app_sys.PATH_ASSET_PREP_MSK_TRAIN}\n')
                f.write(f'val: {app_sys.PATH_ASSET_PREP_MSK_VAL}\n')
                print(f)
                
        except AttributeError as e:
            print(e)


    def split_train_val(self):
        """
        Split the images into training and validation sets
        """
        direc_imgs = app_sys.PATH_ASSET_PREP_MSK_TEMP
        # images to be annotated
        imgs = [f for f in os.listdir(direc_imgs) if f.endswith('.jpg')]
        train, valid = train_test_split(imgs, test_size=0.2, random_state=1)
        dict_data = dict(zip(['train', 'val'], [train, valid]))
        print(dict_data)
        for ndirec in dict_data.keys():
            # ndirec ... 'train' or 'val'
            # train / val directory
            new_direc = os.path.join(app_sys.PATH_ASSET_PREP_MSK, 'images', ndirec)
            direc_exist_check(new_direc)
            # Number of files exist in the directory already
            n_pre_exist = len(os.listdir(new_direc))
            print(f'{n_pre_exist} images exists in {ndirec} directory.')
            # for nfile, nfile_new in zip(dict_data[ndirec], rename(dict_data[ndirec], base=n_pre_exist)):
            for nfile in dict_data[ndirec]:
                # Number of files exist in the directory already.
                os.rename(os.path.join(app_sys.PATH_ASSET_PREP_MSK_TEMP, nfile), os.path.join(new_direc, nfile))
        
        """
        Split the labels into training and validation sets
        """
        direc_labels = app_sys.PATH_ASSET_PREP_MSK_LBL
        # Check the image names in the training/validation set
        dict_data = dict(zip(['train', 'val'], [train, valid]))
        for ndirec in dict_data.keys():
            new_direc = os.path.join(direc_labels, ndirec)
            direc_exist_check(new_direc)
            n_pre_exist = len(os.listdir(new_direc))
            print(f'{n_pre_exist} labels exists in {ndirec} directory.')
            for nfile in dict_data[ndirec]:
                f = nfile.split('.')[0] + '.txt'
                os.rename(os.path.join(direc_labels, f), os.path.join(new_direc, f))

    """ YOLO seg train """
    def train_model(self):
        if len(os.listdir(app_sys.PATH_ASSET_PREP_MSK_TEMP)) > 0:
            self.split_train_val()
            init_yolo_config('pre_mask')
        thread_train = threading.Thread(target=self.training_thread)
        thread_train.start()
    
    def training_thread(self):
        print(self.device_selected.get())
        model_selected = self.yolo_model_selected.get()
        if model_selected == 'yolov11n-seg':
            model = YOLO(app_sys.PATH_MODEL_YOLOV11_SEG)
        else:
            # model = YOLO(app_sys.PATH_MODEL_YOLOV12_SEG)
            model = YOLO(f'{model_selected}.pt')
        print(model.info())
        model.train(
            data=os.path.join(app_sys.PATH_ASSET_PREP_MSK_YAML),
            epochs=int(self.epochs.get()),
            imgsz=640,  # Image size
            batch=int(self.batch.get()),  # Adjust batch size based on GPU memory
            device=self.device_selected.get(),  # Use GPU ("cuda") or CPU ("cpu")
            augment=True,
            hsv_h=0.015,  # Hue shift
            hsv_s=0.7,    # Saturation shift
            hsv_v=0.4,     # Value (Brightness) shift
            mosaic=0.5,
        )


    """
    Mask non-wall area using the trained model.
    """
    def predict(self):
        thread_predict = threading.Thread(target=self.predict_thread)
        thread_predict.start()

    def predict_thread(self):
        model = YOLO(os.path.join(app_sys.PATH_TOOL, 'runs', 'segment', self.trained_model_selected.get(), 'weights', 'best.pt'))
        direc_exist_check(app_sys.PATH_ASSET_MSK)
        masked_video(self.video.vcap, model, saveas=os.path.join(app_sys.PATH_ASSET_MSK, f"{self.video.name.split('.')[0]}_masked.mp4"), class2keep=[0,1])
