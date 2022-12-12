import shutil
import tkinter as tk
import handwriting_recognition as hr
from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image, ImageEnhance
from tkinter import ttk

import os
import sys
import urllib.request
#from bs4 import BeautifulSoup



# Main
window=Tk()

# Getting screen width and height of display
width= window.winfo_screenwidth()
height= window.winfo_screenheight()

# Setting tkinter window size
window.geometry("%dx%d" % (width, height))
window.title("Handwritten Word Recognition")
window.configure(background="white")



title_path="Title_img.jpg"
originalImage=(Image.open(title_path))
titleImage=originalImage.resize((800,400), Image.ANTIALIAS)
titleImage=ImageTk.PhotoImage(titleImage)



# Method for selecting input image and displaying it on the imageWin window
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

    # Method to get value of current slider position
    def get_current_value():
        return '{: .2f}'.format(current_value.get())


    # Method to get sliders current position
    def slider_changed(event):
        
        value_label.configure(text=get_current_value())
        


    # Method for adjusting input image's brightness
    def brightnessUpdate():
        #print("brightness is running")
        
        img_for_brightness=Image.open(path1)
        img_brightness_obj=ImageEnhance.Brightness(img_for_brightness)
        factor=current_value.get()
        #factor=.75
        enhanced_img=img_brightness_obj.enhance(factor)
        enhanced_img.save("tempImg.png")

        global img2
        img3=Image.open("tempImg.png")
        resizedImg=img3.resize((300,205), Image.ANTIALIAS)
        img2=ImageTk.PhotoImage(resizedImg)
        #updates the image from the original image
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

    Label(imageWin, text="The uploaded image: ", bg="blue", fg="white", font ="none 13 bold").grid(row=18, column=0, sticky=N)
    Label(imageWin, text="Please adjust the brightness to where the writing is still visible, but any other infractions are not", bg="blue", fg="white", font="none 13 bold").grid(row=1, column=0,sticky=W)


    # Method to upload image to the inputs folder and add a generic label to the inputs.txt file
    def uploadImage():
        trainCompleteLabel.pack_forget()

        current_dir = os.getcwd()
        dst_path = os.path.join(current_dir, "data\\inputs")
        lbl_path = os.path.join(current_dir, "data\\inputs.txt")

        
        for filename in os.listdir(dst_path):
            file_path = os.path.join(dst_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except:
                pass

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


        if predFile.winfo_ismapped():
            pass
        else:
            predFile.pack()


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



# Method for predicting word based on image of handwritten word
def predWord():
    predFile.pack_forget()

    word = hr.pred_input(0)
    predLabel = word
    prediction_label = Label (window, text="Prediction: " + predLabel, bg="white", fg="black", font="none 13 bold")
    prediction_label.pack()

    confirm_check = Label (window, text="Is the prediction correct?", bg="white", fg="black", font="none 13 bold")
    confirm_check.pack()


    # Method to change image name to the correct label
    def correctLabel(cLabel):
        prediction_label.pack_forget()

        current_dir = os.getcwd()
        dst_path = os.path.join(current_dir, "data\\inputs")
        lbl_path = os.path.join(current_dir, "data\\inputs.txt")

        global correctLabel 
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

        confirm_check.pack_forget()
        confirmLabel.pack_forget()
        fixLabel.pack_forget()
        img_prompt.pack_forget()
        openFile.pack_forget()

        trainLabel.pack()
        train.pack()
        skipTrain.pack()


    confirmLabel = Button(window, text="Yes", width=20, command=lambda: correctLabel(predLabel))
    confirmLabel.pack()


    # Method to correct the label if the predicted label is wrong
    def wrongLabel():
        prediction_label.pack_forget()

        cLabelPrompt = Label(window, text="Enter the correct label: ", fg="black", bg="white", font="none 13 bold")
        cLabelPrompt.pack()

        global inputtxt
        inputtxt = tk.Text(window, height = 1, width = 20)
        inputtxt.pack()

        # Method for getting correct label from text box
        def getLabel():
            newLabel = inputtxt.get("1.0", "end-1c")

            cLabelPrompt.pack_forget()
            inputtxt.pack_forget()
            enterLabel.pack_forget()

            correctLabel(newLabel)


        enterLabel = Button(window, text="Enter", width=20, command=getLabel)
        enterLabel.pack()

        confirm_check.pack_forget()
        confirmLabel.pack_forget()
        fixLabel.pack_forget()
        img_prompt.pack_forget()
        openFile.pack_forget()


    fixLabel = Button(window, text="No", width=20, command=wrongLabel)
    fixLabel.pack()


# Method for training the model with uploaded image and correct label
def trainInput():

    skipTrain.pack_forget()

    hr.train_input()

    train.pack_forget()
    trainLabel.pack_forget()
    title_img.pack_forget()

    trainCompleteLabel.pack()
    saveWordFile.pack()
    dontSave.pack()
    title_img.pack()


# Method to skip training after prediction
def noTrain():
    train.pack_forget()
    skipTrain.pack_forget()
    trainLabel.pack_forget()
    title_img.pack_forget()

    saveWordFile.pack()
    dontSave.pack()
    title_img.pack()


# Method to save the word to a file
def saveWord(cLabel):
    saveWord = fd.asksaveasfile(parent=window, title='Save file')
    line = os.path.basename(cLabel)
    line = os.path.splitext(line)
    saveWord.write(line[0])

    saveWordFile.pack_forget()
    dontSave.pack_forget()
    title_img.pack_forget()

    img_prompt.pack()
    openFile.pack()
    title_img.pack()


# Method to skip saving the word to a file
def noSave():
    saveWordFile.pack_forget()
    dontSave.pack_forget()
    title_img.pack_forget()

    img_prompt.pack()
    openFile.pack()
    title_img.pack()

# labels

img_prompt = Label(window, text="Click to upload an image:", bg="white", fg="black", font="none 13 bold")
img_prompt.pack()


# Load an image from a file using PIL
image = Image.open("upload_button.png")

# Set the image's transparent color to white
image.putalpha(255)

resizedImage=image.resize((150,40), Image.ANTIALIAS)

# Create a PhotoImage object from the image
buttonImage = ImageTk.PhotoImage(resizedImage)


#buttons
openFile=Button(window, text="UPLOAD", image=buttonImage, borderwidth=0, command=upload)
openFile.pack()


title_img=ttk.Label(window, image=titleImage)
title_img.pack()

predFile=Button(window, text="Make Prediction", width=20, command=predWord)

trainLabel = Label(window, text="Hit the train button to train the model with your image.", fg="black", bg="white", font="none 13 bold")
train = Button(window, text="Train", command=trainInput)
skipTrain = Button(window, text="Don't train", command=noTrain)
saveWordFile = Button(window, text="Save to file", command=lambda: saveWord(correctLabel))
dontSave = Button(window, text="Don't save", command=noSave)

trainCompleteLabel = Label(window, text="Training Complete.", fg="black", bg="white", font="none 13 bold")


#run the main loop
window.mainloop()