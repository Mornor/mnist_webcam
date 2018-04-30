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
        self.result = tk.Text(self, width=60)
        self.result.insert('1.0', 'Result detected by the Convolutional Network')
        self.result.pack(side='top')



    def display_result(self):
        print('TODO')

root = tk.Tk()
app = Application(master=root)
app.mainloop()
