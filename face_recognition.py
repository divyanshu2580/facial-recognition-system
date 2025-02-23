import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = "dataset"
user_details_file = "users.txt"

user_dict = {}
if os.path.exists(user_details_file):
    with open(user_details_file, "r") as f:
        for line in f:
            user_id, name = line.strip().split(",", 1)
            user_dict[int(user_id)] = name

def capture_faces():
    user_id = user_id_entry.get().strip()
    user_name = user_name_entry.get().strip()

    if not user_id or not user_name:
        messagebox.showerror("Error", "Please enter both User ID and Name!")
        return

    if not user_id.isdigit():
        messagebox.showerror("Error", "User ID must be a number!")
        return

    user_id = int(user_id)

    if user_id not in user_dict:
        with open(user_details_file, "a") as f:
            f.write(f"{user_id},{user_name}\n")
        user_dict[user_id] = user_name

    cap = cv2.VideoCapture(0)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    messagebox.showinfo("Info", "Capturing faces... Look at the camera.")

    start_time = time.time()
    count = 0

    while time.time() - start_time < 3.5:  
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            filename = f"{dataset_path}/User.{user_id}.{count}.jpg"
            cv2.imwrite(filename, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face Capture', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Captured {count} images for User {user_id} - {user_name}")

def train_model():
    images, labels = [], []

    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        messagebox.showerror("Error", "No face data found! Capture faces first.")
        return

    messagebox.showinfo("Info", "Training model... Please wait.")

    for file in os.listdir(dataset_path):
        if file.endswith(".jpg"):
            path = os.path.join(dataset_path, file)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = int(file.split(".")[1])
            images.append(image)
            labels.append(label)

    recognizer.train(images, np.array(labels))
    recognizer.save("face_model.yml")
    messagebox.showinfo("Success", "Model training completed.")

def recognize_faces():
    if not os.path.exists("face_model.yml"):
        messagebox.showerror("Error", "Model not trained! Train the model first.")
        return

    recognizer.read("face_model.yml")
    cap = cv2.VideoCapture(0)

    messagebox.showinfo("Process Completed!! ", "Recognizing faces... Look at the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < 100 and label in user_dict:
                text = f"{user_dict[label]} (ID: {label})"
                color = (0, 255, 0)  
            else:
                text = "Unknown"
                color = (0, 0, 255)  

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        label_text = "Press 'Q' to exit"
        label_bg = (50, 50, 50)  
        label_text_color = (255, 255, 255)
        height, width, _ = frame.shape
        cv2.rectangle(frame, (0, height - 40), (width, height), label_bg, -1) 
        cv2.putText(frame, label_text, (width // 2 - 100, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_text_color, 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def reset():
    user_id_entry.delete(0, tk.END)  
    user_name_entry.delete(0, tk.END)  
    messagebox.showinfo("Reset Completed!!", "User details have been cleared.")  

root = tk.Tk()
root.title("Face Recognition System")
root.geometry("600x600+450+70")
root.configure(bg="lavender")

tk.Label(root, text="__________________________________________", font=("times new roman",20, "bold"), bg="lavender",fg="navy").place(x=300,y=46,anchor="center")

tk.Label(root, text="FACE RECOGNITION SYSTEM", font=("times new roman",29, "bold"), bg="lavender",fg="navy").place(x=300,y=35,anchor="center")

tk.Label(root, text="__________________________________________", font=("times new roman",20, "bold"), bg="lavender",fg="navy").place(x=300,y=155,anchor="center")

tk.Label(root, text=" ENTER USER DETAILS ", font=("times new roman",24, "bold"), bg="lavender",fg="navy").place(x=300,y=170,anchor="center")

user_id_label = tk.Label(root, text="Enter User ID:", font=("times new roman", 20,"bold"), bg="lavender",fg="navy")
user_id_label.place(x=20,y=230, anchor="w")

user_id_entry = tk.Entry(root, font=("times new roman",20),border=6,width=25,fg="navy")
user_id_entry.place(x=570,y=230, anchor="e")

user_name_label =  tk.Label(root, text="Enter Name:", font=("times new roman", 20,"bold"), bg="lavender",fg="navy")
user_name_label.place(x=30,y=290, anchor="w")

user_name_entry = tk.Entry(root, font=("times new roman",20),border=6,width=25,fg="navy")
user_name_entry.place(x=570,y=290, anchor="e")

tk.Button(root, text="RESET", font=("times new roman",20,"bold"), width=12, command=reset, bg="mediumpurple3",fg="white",border=5).place(x=20,y=105,anchor="w")

tk.Button(root, text="EXIT", font=("times new roman",20,"bold"), width=12, command= lambda : root.destroy() , bg="red2", fg="white",border=5).place(x=570,y=105,anchor="e")

tk.Button(root, text="CAPTURE FACES", font=("times new roman",20,"bold"), width= 30, command=capture_faces, bg="limegreen", fg="white",border=5).place(x=300,y=370,anchor="center")

tk.Button(root, text="TRAIN MODEL", font=("times new roman",20,"bold"), width= 30, command=train_model, bg="darkgoldenrod", fg="white",border=5).place(x=300,y=450,anchor="center")

tk.Button(root, text="RECOGNIZE FACES", font=("times new roman",20,"bold"), width= 30,command=recognize_faces, bg="dodgerblue", fg="white",border=5).place(x=300,y=530,anchor="center")

tk.Label(root, text="© Divyanshu Sharma ",font=("times new roman",13,"bold"),bg="lavender",fg="navy").place(x=597,y=587,anchor="e")

root.mainloop()