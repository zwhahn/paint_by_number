import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import time

class WebcamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Webcam App")

        self.current_image = None
        self.remaining_time = None
        self.start_time = None
        self.vid = cv.VideoCapture(0)
        self.canvas = tk.Canvas(window, width = 640, height=480)
        self.canvas.pack()

        self.capture_button = tk.Button(window, text = "Capture", command = self.capture_image)
        self.capture_button.pack()

        self.update_webcam()

    def update_webcam(self):
        ret, self.frame = self.vid.read()

        if self.start_time is not None:  # Dont add text to image until timer has started
            self.remaining_time = int(3 - ((time.time() - self.start_time) //1))
            if self.remaining_time > 0:  # If there is no remaining time than we should not alter the frame, otherwise add countdown
                cv.putText(self.frame, str(self.remaining_time), (200, 300), cv.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 20, cv.FILLED) 
      
        # Display video feed
        self.current_image = Image.fromarray(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image=self.current_image)

        self.canvas.create_image(0,0,image=self.photo, anchor=tk.NW)


        self.window.after(10, self.update_webcam)

        # Once the countdown is over, the image is captured and saved to the images folder
        if self.remaining_time == 0:
            print("Image Captured!")
            cv.imwrite('./images/capture.png', self.frame)  # overwrites the last captured image
    
    
    def capture_image(self):
        self.start_time = time.time()

         

root = tk.Tk()
app = WebcamApp(root)
root.mainloop()