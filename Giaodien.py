#Import thư viện
import sys
import cv2
import pathlib
import PIL.Image
import PIL.ImageTk
from tkinter import *
from tkinter.ttk import *
import tkinter
from tkinter import ttk
from tkinter.filedialog import Open, SaveAs
from tkinter import messagebox
from matplotlib.ft2font import BOLD
from matplotlib.pyplot import bar_label, text
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.models import load_model
import tensorflow as tf
from yaml import load
import glob
import numpy as np
import pandas as pd
import os
import wave
import pylab
import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style
import pygame
from tkinter import filedialog
from playsound import playsound

#Classify by audio files
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram

audio_file_name = ''

pygame.mixer.init() # initializing the mixer
#Load model
model = load_model('Classification.h5')

class Example(Frame):
    def __init__(self, parent):
        tkinter.Frame.__init__(self, parent, background="white")
    
        self.parent = parent
        self.initUI()
    
    def initUI(self):
        self.parent.title("Classification_CNN_LE QUANG CHIEN_19146310")
        self.style = Style()
        self.style.theme_use("default")
        self.pack(fill=BOTH, expand=1)
        #Tạo Label
        lbl = tkinter.Label(self, text = "PROJECT AI", fg="dark red", bg="white", font=("Arial",24,"bold"))
        lbl.pack(side = TOP, padx=15,pady=15)

        lbl1 = tkinter.Label(self, text = "CLASIFICATION OF 10 ANIMALS BY SOUND \nUSING CONVOLUTIONAL NEURAL NETWORK", fg="dark blue", bg="white", font=("Arial",20,"bold"))
        lbl1.pack(side = TOP,padx=0,pady=0)

        lbl2 = tkinter.Label(self, text="Mode:",fg='#442265', bg="white", font=("Cambria",14))
        lbl2.place(x=55,y=150, width=150,height=50)

        lbl7 = tkinter.Label(self, text="Le Quang Chien - Student Code: 19146310",fg='#442265', bg="white", font=("Cambria",14))
        lbl7.place(x=350,y=550, width=350,height=25)

        #Tạo Nút Nhấn
        AudioButton = tkinter.Button(self, text="Load Audio", bg="#FFFF99",activebackground="white",font=("Cambria",16), command=self.onAudio)
        AudioButton.place(x=60, y=210, width=150,height=50)

        ImageButton = tkinter.Button(self, text="Load Images", bg="Light Blue",activebackground="white",font=("Cambria",16), command=self.onImage)
        ImageButton.place(x=60, y=300, width=150,height=50)

        RegButton = tkinter.Button(self, text="Classify", bg="Light green", activebackground="white",font=("Cambria",16), command=self.onReg)
        RegButton.place(x=60, y=390, width=150,height=50)

        exitButton = tkinter.Button(self, text="Exit",bg="pink", activebackground="white",font=("Cambria",16), command=self.onExit)
        exitButton.place(x=60, y=480, width=150,height=50)

    #Load ảnh 
    def onImage(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png'),('All', '*.all')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
 
        if fl != '':
            global imgin
            imgin = cv2.imread(fl,cv2.IMREAD_COLOR)
            imgin = cv2.resize(imgin, (150,150))
            
            lbl3 = tkinter.Label(self, text="Predicted Object:",fg='#442265', bg="white", font=("Cambria",14))
            lbl3.place(x=250,y=400, width=350,height=25)
            lbl4 = tkinter.Label(self, text="",fg='#442265', bg="white", font=("Cambria",14))
            lbl4.place(x=550,y=400, width=100,height=25)
            lbl5 = tkinter.Label(self, text="Spectrogram:",fg='#442265', bg="white", font=("Cambria",14))
            lbl5.place(x=300,y=170, width=350,height=25)
            #cv2.namedWindow("Image_Test", cv2.WINDOW_AUTOSIZE)
            cv2.imwrite("Image.png", imgin)

            img = PIL.Image.open("Image.png")
            img = PIL.ImageTk.PhotoImage(img)
            img1 = Label(image=img)
            img1.image = img
            img1.place(x=400, y=210)

    # Xử lý audio
    def onAudio(self):
        global imgin
        global audio_file_name
        audio_file_name = filedialog.askopenfilename(filetypes=(("Audio Files", ".wav .ogg"),   ("All Files", "*.*")))


        audio = Audio.from_file(audio_file_name)
        image_shape = (150, 150)
        spectrogram = Spectrogram.from_audio(audio)

        noise = pygame.mixer.Sound(audio_file_name)
        noise.play(0, 5000)

        # Convert Spectrogram object to Python Imaging Library (PIL) Image
        imgin = np.array(spectrogram.to_image(shape=image_shape,invert=True))
        imgin = cv2.resize(imgin, (150,150))
        cv2.imwrite("Image.png", imgin)
        imgin = cv2.imread("Image.png",cv2.IMREAD_COLOR)
        imgin = cv2.resize(imgin, (150,150))
        lbl3 = tkinter.Label(self, text="Predicted Object:",fg='#442265', bg="white", font=("Cambria",14))
        lbl3.place(x=250,y=400, width=350,height=25)
        lbl4 = tkinter.Label(self, text="",fg='#442265', bg="white", font=("Cambria",14))
        lbl4.place(x=550,y=400, width=100,height=25)
        lbl5 = tkinter.Label(self, text="Spectrogram:",fg='#442265', bg="white", font=("Cambria",14))
        lbl5.place(x=300,y=170, width=350,height=25)
        cv2.imwrite("Image.png", imgin)

        img = PIL.Image.open("Image.png")
        img = PIL.ImageTk.PhotoImage(img)
        img1 = Label(image=img)
        img1.image = img
        img1.place(x=400, y=210)
        
    def onExit(self):
        msg = messagebox.showinfo( "Message","Do you exit?")
        self.quit()

    # Xử lý ảnh
    def onReg(self):
        img = cv2.resize(imgin, (150,150))
        img = np.array(imgin)
        img = img.reshape(1,150,150,3)
        img = img.astype('float32')
        img /=255
        pred = np.argmax(model.predict(img),axis = -1)
        label = ['CAT', 'DOG','ROOSTER','CRICKET','DUCK','FROG','MONKEY','GOAT','ELEPHANT','COW']
        label[pred[-1]]
        lbl4 = tkinter.Label(self, text=""+ label[np.argmax(model.predict(img.reshape(1,150,150,3)))],fg='#442265', bg="white", font=("Cambria",14))
        lbl4.place(x=550,y=400, width=100,height=25)

        lbl6 = tkinter.Label(self, text="Accuracy: 0.900 \nPrecision: 0.909\nRecall: 0.900   ",fg='#442265', bg="white", font=("Cambria",14))
        lbl6.place(x=400,y=450, width=200,height=100)

window = Tk()
window.title("Giao Dien Project")
#Gọi class chứa các hàm def
app = Example(window)
window.geometry("800x600+400+100")
window.mainloop()
