import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import time
from PictureProcessing import PaintByNumber

class WebcamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Webcam App")

        # Initial variables
        self.current_image = None
        self.remaining_time = None
        self.start_time = None
        self.picture_taken = False
        self.countdown = 3
        self.vid = cv.VideoCapture(0)

        # Calculate center of frame for countdown position
        self.x_center = int(self.vid.get(cv.CAP_PROP_FRAME_WIDTH)/2)
        self.y_center = int(self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)/2)

        self.capture_button = tk.Button(window, text = "Capture", command = self.capture_image)
        self.capture_button.grid(row=1, column=0)

        self.canvas_created = False
        self.update_webcam()

    def update_webcam(self):
        if self.canvas_created == False:
            self.canvas = tk.Canvas(self.window, width = 640, height=480)
            self.canvas.grid(row=0, column=0)
        
        if self.picture_taken == False:
            _, self.frame = self.vid.read()

            if self.start_time is not None:  # Dont add text to image until timer has started
                self.remaining_time = int(self.countdown - ((time.time() - self.start_time) //1))
                if self.remaining_time > 0:  # If there is no remaining time than we should not alter the frame, otherwise add countdown
                    cv.putText(self.frame, str(self.remaining_time), (self.y_center, self.x_center), cv.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 20, cv.FILLED) 
        
            # Display video feed
            self.current_image = Image.fromarray(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.current_image)

            self.canvas.create_image(0,0,image=self.photo, anchor=tk.NW)

            self.window.after(10, self.update_webcam)

            # Once the countdown is over, the image is captured and saved to the images folder
            if self.remaining_time == 0:
                print("Image Captured!")
                self.picture_taken = True
                cv.imwrite('./images/capture.png', self.frame)  # overwrites the last captured image
                self.canvas.destroy()
                self.picture_taken = True
                self.picture_processing()
    
    
    def capture_image(self):
        self.start_time = time.time()

    def picture_processing(self):
        PaintByNumber()
        print("end")
        self.final_img_file = './images/final_image.jpg'
        self.final_img = ImageTk.PhotoImage(Image.open(self.final_img_file))
        self.canvas = tk.Canvas(self.window, width = 640, height=480)
        self.canvas.grid(row=0, column=0)
        self.canvas.create_image(0,0,image=self.final_img, anchor = tk.NW)
         

root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())  # Kill loop with escape button
root.protocol('WM_DELETE_WINDOW', lambda : root.quit())    # Kill loop with [X] button (top left)
app = WebcamApp(root)
root.mainloop()
print("Loop done")