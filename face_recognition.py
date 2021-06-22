from tkinter import *
import numpy as np
import cv2 as cv
from PIL import ImageTk, Image
from tkinter import filedialog


def openfilename():
    filename = filedialog.askopenfilename(initialdir="/",title="Select a File")
    return filename
    # Change label contents

def recognize():
    haar_cascade = cv.CascadeClassifier('haar_default.xml')

    people = ['Akshay kumar', 'MS Dhoni', 'Narendra modi', 'Pankaj tripathi', 'Shahrukh khan']

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    face_recognizer.read('face_trained.yml')

    img = cv.imread(r'{}'.format(openfilename()))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(img, str(people[label]), (20, 20),cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

        cv.imshow('Detected Face', img)
        lbl.configure(text="Prediction is : "+people[label],font=(16))
        cv.waitKey(0)
        

window = Tk()
window.title('Image (Human Face) Recognition system')
window.geometry("500x500")
window.config(background="white")

label_file_explorer = Label(window,text="Image Recognition System using OpenCV",width=100, height=4,fg="blue")
label_file_explorer.config(font=("Courier", 16))
label_file_explorer.pack()

recog_button = Button(window,text="Recognize Image",command=recognize)
recog_button.pack()

lbl = Label(window, text="Your Prediction is : ")
lbl.pack()

lbl.place(relx = 0.5,rely = 0.5,anchor = 'center')

window.mainloop()
