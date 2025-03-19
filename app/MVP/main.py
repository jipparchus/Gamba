import os
import sys
import tkinter as tk
import tkinter.ttk as ttk

from PIL import ImageTk

path_current = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.split(path_current)[0])

from app_sys import AppSys
from tab1 import Tab1
from tab2 import Tab2
from tab3 import Tab3
from tab4 import Tab4
from tab5 import Tab5
from tab6 import Tab6
from tab7 import Tab7

app_sys = AppSys()

class Application(ttk.Notebook):
    def __init__(self, master=None):
        super().__init__(master)
        # self.root_width = 480
        self.root_width = 1200
        # self.root_height = 730
        self.root_height = 730
        self.master.title('Gamba!! MVP')
        self.master.geometry(f'{self.root_width}x{self.root_height}')
        # child frames can be expanded equally
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)

        # Fundamental devision of top and bottom of the master window
        self._master_top = tk.Frame(self.master, height=790)
        self._master_top.pack(side=tk.TOP, fill='both', expand=True)
        self._master_btm = tk.Frame(self.master, height=10)
        self._master_btm.pack(side=tk.TOP)

        # Canvas to make the master_top scrollable
        self.master_top_canvas = tk.Canvas(self._master_top)
        self.master_top_canvas.pack(side=tk.LEFT, fill='both', expand=True)

        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self._master_top, orient=tk.VERTICAL, command=self.master_top_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas
        self.master_top_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.master_top_canvas.bind('<Configure>', lambda e: self.master_top_canvas.configure(scrollregion=self.master_top_canvas.bbox('all')))

        self.master_top = tk.Frame(self.master_top_canvas)
        self.master_top_canvas.create_window((0,0), window=self.master_top, anchor='nw')

        self.master_btm = tk.Frame(self._master_btm, width=self.root_width, height=100)
        self.master_btm.pack(side=tk.TOP)

        """
        Left: tabs, Right : console
        """
        self.frame_left = tk.Frame(self.master_top, width=500)
        self.frame_left.pack(side=tk.LEFT, expand=True, fill='both')

        self.frame_right = tk.Frame(self.master_top, width=500)
        self.frame_right.pack(side=tk.LEFT,  expand=True, fill='both')

        self.console_text = tk.Text(self.frame_right, state='disabled', width=500)
        self.console_text.pack(expand=True, fill='both')

        self.btn_quit = tk.Button(self.master_btm, text='QUIT',width=60, justify='center' , command=root.destroy, foreground='black', background='yellow')
        self.btn_quit.pack(side=tk.BOTTOM)

        # Create main container frames
        """
        Tabs (ttk.Notebook)
        """
        # Notebook to hold tabs
        self.notebook = ttk.Notebook(self.frame_left, height=self.frame_left['height'], width=self.frame_left['width'])
        # self.notebook.pack(expand=True, fill='both')
        self.notebook.pack(side=tk.LEFT)
        # self.notebook.grid(row=0, column=0, sticky=tk.NSEW, padx=10, pady=10)
        # self.notebook.grid_rowconfigure(0, weight=1)
        # self.notebook.rowconfigure(0, weight=1)

        # Trim the head and tail of the video & depth estimation using video-depth-anything model
        tab1 = tk.Frame(self.notebook)
        self.notebook.add(tab1, text="Trimming")
        Tab1(master=tab1)

        # Sampling frames, add effects, annotate, and train a model for the video to support annotation
        tab2 = tk.Frame(self.notebook)
        self.notebook.add(tab2, text="Image Annotation")
        Tab2(master=tab2)

        # Keypoint annotation
        tab3 = tk.Frame(self.notebook)
        self.notebook.add(tab3, text="Key Points Annotation")
        Tab3(master=tab3)

        # Wall 3D coords reconstruction
        tab4 = tk.Frame(self.notebook)
        self.notebook.add(tab4, text="3D Wall")
        Tab4(master=tab4)

        # Solve RANSAC PnP -> get rotation & translation matrix -> 3D wall & human pose coordinates standardisation
        tab5 = tk.Frame(self.notebook)
        self.notebook.add(tab5, text="Solving PnP")
        Tab5(master=tab5)

        # Contacts on the wall
        tab6 = tk.Frame(self.notebook)
        self.notebook.add(tab6, text="Contact Detection")
        Tab6(master=tab6)

        # Wall & climber 3D model with contact on the wall
        tab7 = tk.Frame(self.notebook)
        self.notebook.add(tab7, text="Summary")
        Tab7(master=tab7)

        # On selection of Tab3, pop up a window to show an image of the key points selection.
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_selected)

        # redirect sys.stdout -> TextRedirector
        self.redirect_sysstd()

    def on_tab_selected(self, event):
        """ Detect when Tab 3 is selected and open a new window """
        selected_tab = event.widget.index('current')
        if selected_tab == 2:
            self.open_window_kp()

    def open_window_kp(self):
        """ Create a separate window """
        self.win_kp_skeleton = tk.Toplevel(self)
        self.win_kp_skeleton.title('Key Point Skeleton')
        self.win_kp_skeleton.geometry('600x900')

        # Canvas to show picture
        self.canvas_pic = tk.Canvas(self.win_kp_skeleton, width=600, height=900)
        self.canvas_pic.pack(side=tk.TOP)

        self.img = ImageTk.PhotoImage(file = os.path.join(app_sys.PATH_ASSET, 'kp_skeleton.png'))
        self.canvas_pic.create_image(0, 0, anchor=tk.NW, image=self.img)

    
    def redirect_sysstd(self):
        # We specify that sys.stdout point to TextRedirector
        sys.stdout = TextRedirector(self.console_text, "stdout")
        sys.stderr = TextRedirector(self.console_text, "stderr")


class TextRedirector(object):
    def __init__(self, widget, tag):
        self.widget = widget
        self.tag = tag

    def write(self, text):
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        self.widget.configure(state='normal') # Edit mode
        self.widget.insert(tk.END, text, (self.tag,)) # insert new text at the end of the widget
        self.widget.configure(state='disabled') # Static mode
        self.widget.see(tk.END) # Scroll down 
        self.widget.update_idletasks() # Update the console

    def flush(self):
        pass

if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(False, False)
    app = Application(master=root)
    app.mainloop()
