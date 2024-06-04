import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import time

class WebcamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Webcam App")

        self.current_image = None
        self.vid = cv.VideoCapture(0)
        self.canvas = tk.Canvas(window, width = 640, height=480)
        self.canvas.pack()

        self.capture_button = tk.Button(window, text = "Capture", command = self.capture_image)
        self.capture_button.pack()

        self.update_webcam()

    def update_webcam(self):
        ret, self.frame = self.vid.read()

        if ret:
            self.current_image = Image.fromarray(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.current_image)

            self.canvas.create_image(0,0,image=self.photo, anchor=tk.NW)


            self.window.after(10, self.update_webcam)

    def capture_image(self):
        print(self.current_image)
        print("Trying to take picture")
        if self.current_image is not None:
            print("Picture taken!")
            cv.imwrite('./images/capture.png', self.frame)
         

root = tk.Tk()
app = WebcamApp(root)
root.mainloop()