from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk

vid = cv.VideoCapture(0)

app = Tk()
app.attributes('-fullscreen',True)
app.bind('<Escape>', lambda e: app.quit())

label_widget = Label(app)
label_widget.pack()


def open_camera():
    _, frame = vid.read()

    opencv_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    captured_image = Image.fromarray(opencv_image)

    photo_image = ImageTk.PhotoImage(image=captured_image)

    label_widget.photo_image = photo_image

    label_widget.configure(image = photo_image)

    label_widget.after(10, open_camera)

button1 = Button(app, text="Open Camera", command=open_camera)
button1.pack()

app.mainloop()