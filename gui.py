import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import time

class WebcamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Webcam App")

        self.vid = cv.VideoCapture(0)
        self.canvas = tk.Canvas(window, width = 640, height=480)
        self.canvas.pack()
        self.update_webcam()

    def update_webcam(self):
        ret, frame = self.vid.read()

        if ret:
            self.current_image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.current_image)

            self.canvas.create_image(0,0,image=self.photo, anchor=tk.NW)


            self.window.after(10, self.update_webcam)
         

root = tk.Tk()
app = WebcamApp(root)
root.mainloop()