import tensorflow as tf
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image, ImageEnhance
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







#upload command
def upload():
    #gets the file path to the image being called.
    chosen = fd.askopenfilenames(parent=window, title='Choose a File')
    #print(window.splitlist(chosen))

    imageWin = Toplevel()
    imageWin.title("image uploaded")
    imageWin.geometry("740x500")

    imageWin.configure(background="blue")
    
    #image window after image is chosen
    
    slider_label = ttk.Label(
    imageWin,
    text='Brightness:',
    background="blue",
    font="none 13 bold",
    foreground = "white"
        ).grid(row=10, column=0, sticky=W)

    ########## Slider work
    current_value = tk.IntVar()

    def get_current_value():
        return '{: .2f}'.format(current_value.get())



    def slider_changed(event):
        
        value_label.configure(text=get_current_value())



    def brightnessUpdate(e):
        #img_for_brightness=Image.open(path1)
        #img_brightness_obj=ImageEnhance.Brightness(img_for_brightness)
        #factor=current_value.get()
    
        #enhanced_img=img_brightness_obj.enhance(factor)
        #enhanced_img.save(path1)
        path2="C:\\Users\\somer\\source\\repos\\OCR_gui\\tumblr_oz1pbalxNn1us9frso1_1280.jpg"
        img2=ImageTk.PhotoImage(Image.open(path2))
        panel.configure(image=img2)
        panel.image=img2
        #image_with_brightness=ttk.Label(imageWin, image=img2).grid(row=20, column=4, sticky=S)  
        #enhanced_img.save(path1)



    brightness=tk.Scale(
        imageWin,
        from_=1,
        to=10,
        orient='horizontal',
        variable=current_value,  
        command = slider_changed
        )
    brightness.grid(
        column=0,
        columnspan=2,
        row=11,
        ipadx=50,
        pady=10,
        sticky=W)

    #######
    value_label = tk.Label(imageWin, text=get_current_value())

    


    #picture = ImageTk.PhotoImage(Image.open(chosen))
    print (chosen)
    #print('selected', file)
    #image.print (file)

    Label (imageWin, text="The uploaded image: ", bg="blue", fg="white", font ="none 13 bold") .grid(row=18, column=0, sticky=N)
    Label(imageWin, text="Please adjust the brightness to where the writing is still visible, but any other infractions are not", bg="blue", fg="white", font="none 13 bold").grid(row=1, column=0,sticky=W)
    Label(imageWin, text="Hitting start will start the train process", bg="blue", fg="white", font="none 10").grid(row=2, column=0, sticky=W)
    start=Button(imageWin, text="Start", width=10, command=startLearning).grid(row=3, column=0, sticky=W)
    updateBrightness=Button(imageWin, text="Update image", width=15, command=brightnessUpdate(panel)).grid(row=3, column=2, sticky=E)
    #############################displaying image chosen to be able to adjust brightness
    st=''
    #make chosen tuple a string instead 
    for item in chosen:
        st=st+item
    global path1
    path1=st
    global img
    img=ImageTk.PhotoImage(Image.open(path1))
    panel=ttk.Label(imageWin, image=img).grid(row=20, column=0, sticky=S)
    

    #img_for_brightness=Image.open(path1)
    #img_brightness_obj=ImageEnhance.Brightness(img_for_brightness)
    #factor=current_value.get()
    
    #enhanced_img=img_brightness_obj.enhance(factor)
    #image_with_brightness=ttk.Label(imageWin, image=img).grid(row=20, column=0, sticky=S)
    #enhanced_img.save(path1)
    print(img)
    imageWin.mainloop()

    
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

			
