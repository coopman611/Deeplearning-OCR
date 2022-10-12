import tensorflow as tf
import tkinter as ttk
from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image

import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import urllib.request
#from bs4 import BeautifulSoup

from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


#main
window=Tk()
window.title("Handwritten word recognition")
window.geometry("8  00x300")
window.configure(background="black")


#upload command
def upload():
    #gets the file path to the image being called.
    chosen = fd.askopenfilenames(parent=window, title='Choose a File')
    print(window.splitlist(chosen))
    
    
    
    #image window after image is chosen
    image = Tk()
    image.title("image uploaded")
    image.geometry("300x300")
    image.configure(background="blue")
    #picture = ImageTk.PhotoImage(Image.open(chosen))
    print (chosen)
    #print('selected', file)
    #image.print (file)

    Label (image, text="The uploaded image: ", bg="blue", fg="white", font ="none 13 bold") .grid(row=1, column=0, sticky=W)
    start=Button(image, text="Start", width=10, command=startLearning).grid(row=2, column=0, sticky=W)
    
def startLearning():
    print("Working")

def initTrain():
    print("Training:")

    



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
