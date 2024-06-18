import tkinter as tk
from tkinter import font as tkFont
import cv2 as cv
from PIL import Image, ImageTk
import time
from PictureProcessing import PaintByNumber, PrintPDF

class WebcamApp:
    def __init__(self, window):
        self.button_font = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.button_font_sm = tkFont.Font(family='Helvetica', size=10, weight='bold')
        self.title_font = tkFont.Font(family='Helvetica', size=40, weight='bold')
        self.window = window
        self.window.attributes('-fullscreen', True)
        self.window.title("Webcam App")
        self.window.configure(background="white")
        self.title = tk.Label(self.window, text = "PAint By Number Generator", font = self.title_font, bg="white")
        self.title.grid(row=0, column=0, columnspan=2, sticky='N', pady = 15)

        # Initial variables
        self.current_image = None
        self.remaining_time = None
        self.start_time = None
        self.picture_taken = False
        self.countdown = 3
        self.vid = cv.VideoCapture(0)
        self.vid_size_w = 700
        self.vid_size_h = int(self.vid_size_w*0.75)
        self.print_text = "Print and Color!"

        # Calculate center of frame for countdown position
        self.x_center = int(self.vid.get(cv.CAP_PROP_FRAME_WIDTH)/2)
        self.y_center = int(self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)/2)

        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()

        self.capture_button = tk.Button(window, text = "Capture", command = self.capture_image)
        self.capture_button.grid(row=2, column=0, columnspan=2, sticky='N', pady = 15)
        self.capture_button.config(height= 1, font=self.button_font, bg="light salmon", relief="groove", cursor="circle")

        self.canvas_created = False
        self.update_webcam()

    def update_webcam(self):
        if self.canvas_created == False:
            self.canvas = tk.Canvas(self.window, width = self.screen_width, height=self.vid_size_h,
                                     background="white", highlightthickness=0)
            self.canvas.grid(row=1, column=0, columnspan=2)
            self.canvas_created = True  # Stop multiple canvas' from being created
        
        if self.picture_taken == False:
            _, self.frame = self.vid.read()

            if self.start_time is not None:  # Dont add text to image until timer has started
                self.remaining_time = int(self.countdown - ((time.time() - self.start_time) //1))
                if self.remaining_time > 0:  # If there is no remaining time than we should not alter the frame, otherwise add countdown
                    cv.putText(self.frame, str(self.remaining_time), (self.y_center, self.x_center), cv.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 20, cv.FILLED) 
        
            # Display video feed
            self.current_image = Image.fromarray(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB))
            self.resized_image= self.current_image.resize((self.vid_size_w, self.vid_size_h))
            self.photo = ImageTk.PhotoImage(image=self.resized_image)

            self.canvas.create_image(self.screen_width/2,0,image=self.photo, anchor=tk.N)

            self.window.after(10, self.update_webcam)

            # Once the countdown is over, the image is captured and saved to the images folder
            if self.remaining_time == 0:
                self.picture_taken = True
                print("Image Captured!")
                cv.imwrite('./images/capture.png', self.frame)  # overwrites the last captured image
                self.canvas.destroy()
                self.picture_processing()
    
    
    def capture_image(self):
        self.start_time = time.time()


    def picture_processing(self):
        PaintByNumber()  # Run paint-by-number function
        self.capture_button.destroy()  # Remove 'capture' button

        # Upload final image and display
        self.final_img_file = './images/final_image.jpg'
        self.final_img_open = Image.open(self.final_img_file)
        self.resized_final_img= self.final_img_open.resize((self.vid_size_w, self.vid_size_h))
        self.final_img = ImageTk.PhotoImage(self.resized_final_img)
        self.canvas = tk.Canvas(self.window, width = self.screen_width, height=self.vid_size_h,
                                     background="white", highlightthickness=0)
        self.canvas.grid(row=1, column=0, columnspan=2)
        self.canvas.create_image(self.screen_width/2,0,image=self.final_img, anchor = tk.N)

        # Create 'Restart' and 'Print Button'
        self.restart_button = tk.Button(self.window, text = "Try Again", background="gainsboro",
                                        command = self.restart, font=self.button_font_sm, cursor="exchange", relief="flat")
        self.restart_button.grid(row=3, column=0, sticky="N", columnspan=2, pady=5)
        self.print_button = tk.Button(self.window, text = self.print_text, width=20, background="pale green",
                                        command = self.print_pdf, font=self.button_font, cursor="heart")
        self.print_button.grid(row=2, column=0, sticky="N", columnspan=2, pady = 15)

    def restart(self):
        # Remove 'restart' and 'print' button and final image
        self.restart_button.destroy()  # Remove 'capture' button
        self.print_button.destroy()
        self.canvas.destroy()

        # Reset variables to run camera feed
        self.remaining_time = None
        self.start_time = None
        self.picture_taken = False
        self.canvas_created = False

        # Add back the capture button
        self.capture_button = tk.Button(self.window, text = "Capture", command = self.capture_image)
        self.capture_button.grid(row=2, column=0, columnspan=2, sticky='N', pady = 15)
        self.capture_button.config(height= 1, font=self.button_font, bg="light salmon", relief="groove", cursor="circle")
        
        self.update_webcam()

    def update_print_button(self):
        self.print_button.config(text="Printing... Enjoy!", foreground="gray20", relief="sunken", background="gainsboro")

    def print_pdf(self):
        PrintPDF()
        self.update_print_button()
        self.window.after(2500, self.restart)

root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())  # Kill loop with escape button
root.protocol('WM_DELETE_WINDOW', lambda : root.quit())    # Kill loop with [X] button (top left)
app = WebcamApp(root)
root.mainloop()
print("Loop done")