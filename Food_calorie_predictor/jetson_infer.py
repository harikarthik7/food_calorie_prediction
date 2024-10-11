from tkinter import *
from PIL import Image, ImageTk
import cv2
import os
import warnings
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras.backend as K
root = Tk()
root.title('Camera App')
# root.geometry('640x520')
root.minsize(646,530)
root.maxsize(646,530)
root.configure(bg='#58F')
cap= cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print("Unable to read camera feed")
path_to_model="model_v1_inceptionV3_new.h5"
print("Loading the model..")
model = load_model(path_to_model)
print("Done!")
category={
0:['Bajji 150'],1:['Biriyani (veg=200 ,chicken=500 ,mutton=640)'],2:['Burger 295'], 3: ['Butter Naan 380'], 4: ['Chai 120'],
5: ['Chapati 70'], 6: ['Chole Puri 427'], 7:['Curd rice 160'],8: ['Dal Makhani 280'],
9: ['Fried Rice 165'], 10: ['Idli 60'], 11: ['Jalebi 150'],
12: ['Kaathi Rolls 200'], 13: ['Kadai Paneer 325'], 14: ['Kulfi 160'],
15: ['Masala Dosa 168'],16:['Methu vadai 75'] ,17:['Milkshake 110'],18: ['Momos 40'], 19:['Noodles 140'],20: ['Paani Puri 35/pcs'],
21: ['Pakode 315'], 22: ['Pav Bhaji 400'], 23: ['Pizza 280'], 24: ['Samosa 250']}

def predict_image(filename,model):
    img_ = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    name=category[index][0]+" cal"
    open_popup(name)
def captureImage():
    image=Image.fromarray(img1)
    time='saved_img.jpg'
    image.save(time)
    predict_image('saved_img.jpg',model)
def exitWindow():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()
    root.quit()
def open_popup(a):
    top= Toplevel(root)
    top.geometry("750x250")
    top.title("PREDICTION")
    Label(top, text= a, font=('Helvetica 18 bold')).place(x=150,y=80)
    top.after(5000, top.destroy)
f1=LabelFrame(root,bg='red')
f1.pack()
l1=Label(f1,bg='red')
l1.pack()
b1=Button(root,bg='green',fg='white',activebackground='white',activeforeground='green',text='PredictCalorie📷',relief=RIDGE,height=200,width=30,command=captureImage)
b1.pack(side=LEFT,padx=60,pady=5)
b2=Button(root,fg='white',bg='red',activebackground='white',activeforeground='red',text='EXIT❌',relief=RIDGE,height=200,width=20,command=exitWindow)
b2.pack(side=LEFT,padx=40,pady=5)
while True:
    img=cap.read()[1]
    img=cv2.flip(img,1)
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=ImageTk.PhotoImage(Image.fromarray(img1))
    l1['image']=img
    root.update()
    cap.release()
