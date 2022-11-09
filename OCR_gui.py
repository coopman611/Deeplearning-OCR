import shutil
import tensorflow as tf
import tkinter as tk
import handwriting_recognition as hr
from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from tkinter import ttk

import os
import sys
import urllib.request
#from bs4 import BeautifulSoup

from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession





# Main
window=Tk()

# Getting screen width and height of display
width= window.winfo_screenwidth()
height= window.winfo_screenheight()

# Setting tkinter window size
window.geometry("%dx%d" % (width, height))
window.title("Handwritten Word Recognition")
window.configure(background="black")






# upload command
def upload():
    # gets the file path to the image being called.
    chosen = fd.askopenfilenames(parent=window, title='Choose a File')
    # print(window.splitlist(chosen))

    imageWin = Toplevel()
    imageWin.title("image uploaded")
    imageWin.geometry("840x500")

    imageWin.configure(background="blue")
    
    # image window after image is chosen
    
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



    def brightnessUpdate():
        #print("brightness is running")
        
        img_for_brightness=Image.open(path1)
        img_brightness_obj=ImageEnhance.Brightness(img_for_brightness)
        factor=current_value.get()
        #factor=.75
        enhanced_img=img_brightness_obj.enhance(factor)
        enhanced_img.save(path1)
        global img2
        img2=ImageTk.PhotoImage(Image.open(path1))
        panel.configure(image=img2)
        panel.image=img2




    brightness=tk.Scale(
        imageWin,
        from_=1,
        to=5,
        orient='horizontal',
        variable=current_value,  
        command = slider_changed
        )
    brightness.grid(
        column=0,
        columnspan=2,
        row=15,
        ipadx=50,
        pady=10,
        sticky=W)

    #######
    value_label = tk.Label(imageWin, text=get_current_value())


    print (chosen)

    Label(imageWin, text="The uploaded image: ", bg="blue", fg="white", font ="none 13 bold") .grid(row=18, column=0, sticky=N)
    Label(imageWin, text="Please adjust the brightness to where the writing is still visible, but any other infractions are not", bg="blue", fg="white", font="none 13 bold").grid(row=1, column=0,sticky=W)


    def uploadImage():
        current_dir = os.getcwd()
        dst_path = os.path.join(current_dir, "data\\inputs")
        lbl_path = os.path.join(current_dir, "data\\inputs.txt")

        _, _, files = next(os.walk(dst_path))
        file_count = len(files) + 1

        imageName = os.path.basename(path1)
        imageName = os.path.splitext(imageName)
        
        shutil.copy(path1, dst_path)

        img_path = os.path.join(dst_path, imageName[0] + imageName[1])
        global newName
        newName = os.path.join(dst_path, "img" + str(file_count) + ".png")
        os.rename(img_path, newName)

        imgList = os.listdir(dst_path)

        with open(lbl_path, 'r') as file:
            # read a list of lines into data
            data = file.readlines()

        index = 4   

        # iterate each line
        for line in (imgList):
            
            line = os.path.basename(line)
            line = os.path.splitext(line)
            try:
                data[index] = line[0] + '\n'
                index = index + 1
            except:
                data.append(line[0] + '\n')

        with open(lbl_path, 'w') as file:
            # write lines from data into file
            file.writelines(data)

        imageWin.destroy()

        predFile=Button(window, text="Make Prediction", width=20, command=predWord).grid(row=3, column=0, sticky=W)




    upload =Button(imageWin, text="Upload Image", width=10, command=uploadImage).grid(row=3, column=0, sticky=W)
    updateBrightness =Button(imageWin, text="Update Image", width=15, command=brightnessUpdate).grid(row=6, column=0, sticky=W, pady=5)

    #############################displaying image chosen to be able to adjust brightness
    st=''
    #make chosen tuple a string instead 
    for item in chosen:
        st=st+item
    global path1
    path1=st

    img=Image.open(path1)

    #Resize the Image using resize method
    resized_image= img.resize((300,205), Image.ANTIALIAS)
    new_image= ImageTk.PhotoImage(resized_image)

    global panel
    panel=ttk.Label(imageWin, image=new_image)
    panel.grid(row=30, column=0, sticky=S)    

    print(new_image)
        

    imageWin.mainloop()




def predWord():
    word = hr.pred_input(0)
    predLabel = word
    prediction_label = Label (window, text="Prediction: " + predLabel, bg="black", fg="white", font="none 13 bold")
    prediction_label.grid(row=7, column=0, sticky=W)

    confirm_check = Label (window, text="Is the prediction correct?", bg="black", fg="white", font="none 13 bold")
    confirm_check.grid(row=8, column=0, sticky=W)


    def correctLabel(cLabel):
        current_dir = os.getcwd()
        dst_path = os.path.join(current_dir, "data\\inputs")
        lbl_path = os.path.join(current_dir, "data\\inputs.txt")

        correctLabel = os.path.join(dst_path, cLabel + ".png")
        os.rename(newName, correctLabel)

        imgList = os.listdir(dst_path)

        with open(lbl_path, 'r') as file:
            # read a list of lines into data
            data = file.readlines()

        index = 4   

        # iterate each line
        for line in (imgList):
            
            line = os.path.basename(line)
            line = os.path.splitext(line)
            try:
                data[index] = line[0] + '\n'
                index = index + 1
            except:
                data.append(line[0] + '\n')

        with open(lbl_path, 'w') as file:
            # write lines from data into file
            file.writelines(data)

        confirm_check.destroy()
        confirmLabel.destroy()
        fixLabel.destroy()
        img_prompt.destroy()
        openFile.destroy()

        
        trainLabel = Label(window, text="Hit the train button to train the model with your image.", fg="white", bg="black", font="none 13 bold")
        trainLabel.grid(row=0, column=0, sticky=W)

        train = Button(window, text="Train", command=trainInput)
        train.grid(row=1, column=0, sticky=W)


    confirmLabel = Button(window, text="Yes", width=20, command=lambda: correctLabel(predLabel))
    confirmLabel.grid(row=9, column=0, sticky=W)


    def wrongLabel():
        cLabelPrompt = Label(window, text="Enter the correct label: ", fg="white", bg="black", font="none 13 bold")
        cLabelPrompt.grid(row=8, column=0, sticky=W)
        global inputtxt
        inputtxt = tk.Text(window, height = 1, width = 20)
        inputtxt.grid(row=8, column=1, sticky=W)

        def getLabel():
            newLabel = inputtxt.get("1.0", "end-1c")
            correctLabel(newLabel)  
            cLabelPrompt.destroy()
            inputtxt.destroy()
            enterLabel.destroy()

        enterLabel = Button(window, text="Enter", width=20, command=getLabel)
        enterLabel.grid(row=9, column=0, sticky=W)

        confirm_check.destroy()
        confirmLabel.destroy()
        fixLabel.destroy()
        img_prompt.destroy()
        openFile.destroy()

        trainLabel = Label(window, text="Hit the train button to train the model with your image.", fg="white", bg="black", font="none 13 bold")
        trainLabel.grid(row=0, column=0, sticky=W)

        train = Button(window, text="Train", command=trainInput)
        train.grid(row=1, column=0, sticky=W)


    fixLabel = Button(window, text="No", width=20, command=wrongLabel)
    fixLabel.grid(row=9, column=1, sticky=W)



def trainInput():
    hr.train_input()

    trainCompleteLabel = Label(window, text="Training Complete.", fg="white", bg="black", font="none 13 bold")
    trainCompleteLabel.grid(row=0, column=0, sticky=W)  



# labels
img_prompt = Label(window, text="Please upload picture of a single word:", bg="black", fg="white", font="none 13 bold")
img_prompt.grid(row=1, column=0, sticky=W)

#buttons
openFile=Button(window, text="UPLOAD", width=20, command=upload)
openFile.grid(row=2, column=0, sticky=W)

#ttk.Button(window, text="Select a File", command=upload).grid(row=3, columb=1, sticky=W)
#image for title screen
#path1="cat.png"
#img=ImageTk.PhotoImage(Image.open(path1))
#panel=ttk.Label(window, image=img). grid(row=3)


#run the main loop
window.mainloop()
