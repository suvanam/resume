import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

def tk():
    filetypes = ('pdf files', '*.pdf'),('All files', '*.*')
    filename = fd.askopenfilename(title='Open a file',initialdir='/', filetypes=filetypes)
    #showinfo(title='Selected File', message=filename)

    return(filename)
