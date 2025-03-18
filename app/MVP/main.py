import os
import sys
import tkinter as tk
import tkinter.ttk as ttk

path_current = os.path.dirname(os.path.abspath('__file__'))
os.path.split(path_current)[0]
sys.path.append('/workspaces/MoonClimbers/app')

from tab1 import Tab1
from tab2 import Tab2
from tab3 import Tab3
from tab4 import Tab4
from tab5 import Tab5
from tab6 import Tab6


class Application(ttk.Notebook):
    def __init__(self, master=None):
        super().__init__(master)
        # self.root_width = 480
        self.root_width = 1200
        # self.root_height = 730
        self.root_height = 900
        self.master.title('Gamba!! MVP')
        self.master.geometry(f'{self.root_width}x{self.root_height}')

        """
        Left: tabs, Right : console
        """
        # Create main container frames
        self.frame_left = tk.Frame(self, width=800)
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_right = tk.Frame(self, relief=tk.SOLID, bd=5, width=700)
        self.frame_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.console_text = tk.Text(self.frame_right, state='disabled', height=800)
        self.console_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        """
        Tabs (ttk.Notebook)
        """
        # Notebook to hold tabs
        self.notebook = ttk.Notebook(self.frame_left)
        self.notebook.pack(expand=True, fill='both')

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
        Tab4(master=tab5)

        # Contacts on the wall
        tab6 = tk.Frame(self.notebook)
        self.notebook.add(tab6, text="Contact Detection")
        Tab5(master=tab6)

        # Wall & climber 3D model with contact on the wall
        tab7 = tk.Frame(self.notebook)
        self.notebook.add(tab7, text="Summary")
        Tab6(master=tab7)

        self._quit_app()
        self.pack()

        # redirect sys.stdout -> TextRedirector
        self.redirect_sysstd()

    def _quit_app(self):
        quit = tk.Button(self.master, text="QUIT", command=root.destroy)
        quit.pack(side=tk.BOTTOM)
    
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
    app = Application(master=root)
    app.mainloop()
