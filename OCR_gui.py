import tensorflow as tf
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import urllib.request
#from bs4 import BeautifulSoup

from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession





#main
window=Tk()
window.title("Handwritten word recognition")
window.geometry("800x300")
window.configure(background="black")

current_value = tk.DoubleVar()

def get_current_value():
    return '{: .2f}'.format(current_value.get())


def slider_changed(event):
    value_label.configure(text=get_current_value())





#upload command
def upload():
    #gets the file path to the image being called.
    chosen = fd.askopenfilenames(parent=window, title='Choose a File')
    print(window.splitlist(chosen))
    
    
    
    #image window after image is chosen
    image = Tk()
    image.title("image uploaded")
    image.geometry("500x500")

    image.configure(background="blue")
    slider_label = ttk.Label(
    image,
    text='Brightness:',
    background="blue",
    font="none 13 bold",
    foreground = "white"
        )

    slider_label.grid(
    column=6,
    row=5,
    sticky='w'
        )
    brightness=ttk.Scale(
        image,
        from_=0,
        to=100,
        orient='horizontal',
        variable=current_value        
        )
    brightness.grid(
        column=5,
        columnspan=3,
        row=6,
        sticky=W)
    current_value_label = ttk.Label(
    image,
    text='Current Value:',
    background="blue",
    font="none 13 bold",
    foreground = "white"
        )

    current_value_label.grid(
        row=8,
        column=6,
        columnspan=2,
        sticky='n',
        ipadx=10,
        ipady=10
        )
    #######
    value_label = ttk.Label(
    image,
    text=get_current_value()
        )
    value_label.configure(
    background="blue",
    font="none 13 bold",
    foreground = "white")
    value_label.grid(
        row=8,
        column=8,
        columnspan=2,
        sticky='n'
        )


    #picture = ImageTk.PhotoImage(Image.open(chosen))
    print (chosen)
    #print('selected', file)
    #image.print (file)

    Label (image, text="The uploaded image: ", bg="blue", fg="white", font ="none 13 bold") .grid(row=1, column=0, sticky=W)
    start=Button(image, text="Start", width=10, command=startLearning).grid(row=2, column=0, sticky=W)
    
def startLearning():
    print("Working")

def initTrain():
    global initLoc
    #initLoc = fd.askopenfilenames( title='choose initial training folder')
    initLoc=fd.askdirectory()
    
    #print (str(initLoc))
    #str(initLoc)
    #####NOTE FOR BELOW IMPORT: Import calls the module, which for some reason or another is opening a new gui from scratch, need to figure out 
    #import module1
    exec(open('module1.py').read())
    print("done:")





#img=PhotoImage(file="cat.png")
#lables
Label (window, text="please upload picture with single word:", bg="black", fg="white", font ="none 13 bold") .grid(row=1, column=0, sticky=W)
Label (window, text="if you have not trained the model yet, please select 'Initial Training' to get the model to work properly.", bg="black", fg="white", font="none 13 bold").grid(row=5, column=0, sticky=W)
#buttons
openFile=Button(window, text="UPLOAD", width=20, command=upload) .grid(row=2, column=0, sticky=W)
initialTrain=Button(window, text="Initial Training", width=25, command=initTrain).grid(row=3, column=0, sticky=W)
#ttk.Button(window, text="Select a File", command=upload).grid(row=3, columb=1, sticky=W)
#image for title screen
#path1="cat.png"
#img=ImageTk.PhotoImage(Image.open(path1))
#panel=ttk.Label(window, image=img). grid(row=3)


#run the main loop
window.mainloop()

			
