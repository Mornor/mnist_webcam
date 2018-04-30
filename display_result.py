"""
Python file used to display the classes the ML learning
recognize from the live video feed.
"""

import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Title
        result_title = tk.Text(self, width=60, height=0.5)
        result_title.tag_configure("center", justify='center')
        result_title.insert('1.0', 'Result detected by the Convolutional Network')
        result_title.tag_add("center", "1.0", "end")
        result_title.pack(side='top')

        # Result's frame
        result_frame = tk.Text(self, width=60, height=5)
        result_frame.tag_configure("center", justify='center')
        result_frame.insert('1.0', 'Result')
        result_frame.tag_add("center", "1.0", "end")
        result_frame.pack(side='top')

        


    def display_result(self):
        print('TODO')

root = tk.Tk()
app = Application(master=root)
app.mainloop()
