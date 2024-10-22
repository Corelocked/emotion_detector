import cv2
import numpy as np
import tkinter as tk
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

model = load_model('emotion_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

root = tk.Tk()
root.title("Emotion Detector")

emotion_label = tk.Label(root, text="Emotion: ", font=("Helvetica", 24))
emotion_label.pack()

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

cap = cv2.VideoCapture(0)  

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  
    face = cv2.resize(face, (48, 48))              
    face = face.astype('float32') / 255            
    face = np.reshape(face, (1, 48, 48, 1))        
    return face


def update_frame():

    ret, frame = cap.read()
    
    if not ret:
        return
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]
        
        preprocessed_face = preprocess_face(face)
        
        prediction = model.predict(preprocessed_face)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        emotion_label.config(text=f"Emotion: {emotion}")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.imgtk = imgtk 
    canvas.after(10, update_frame) 

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
