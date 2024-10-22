import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
import imutils
import time
from PrinterFormat import CreatePDF
from win32printing import Printer
import win32api

# AI Imports
import io
import os
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


'''TAKE PICTURE'''
TAKING_PICTURE = False
def take_picture():
    if TAKING_PICTURE:
        print("Starting Camera...")

        # Start video object, 0 uses first camera available
        vid = cv.VideoCapture(0)
        print("Camera On! Press 'y' to capture an image!")

        # Create fullscreen window for video feed
        cv.namedWindow("video_feed", cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty("video_feed", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        # Calculate center of frame for countdown position
        x_center = int(vid.get(cv.CAP_PROP_FRAME_WIDTH)/2)
        y_center = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)/2)

        # Initial variable values
        font = cv.FONT_HERSHEY_SIMPLEX
        countdown = 3
        start_time = None
        remaining_time = 1

        while (True):
            ret, frame = vid.read()

            if start_time is not None:  # Dont add text to image until timer has started
                remaining_time = int(countdown - ((time.time() - start_time) //1))
                if remaining_time > 0:  # If there is no remaining time than we should not alter the frame, otherwise add countdown
                    cv.putText(frame, str(remaining_time), (y_center, x_center), font, 7, (0, 0, 0), 20, cv.FILLED) 

            # Press 'y' to start countdown
            if cv.waitKey(1) & 0x0FF == ord('y'):
                print("Countdown Started")
                start_time = time.time()

            if cv.waitKey(1) & 0x0FF == ord('q'):
                break
            
            cv.imshow('video_feed', frame)  # Display video feed

            # Once the countdown is over, the image is captured and saved to the images folder
            if remaining_time == 0:
                print("Image Captured!")
                cv.imwrite('./images/capture.png', frame)  # overwrites the last captured image
                break
                
        # Shut down video object
        vid.release()
        cv.destroyAllWindows()

    if not TAKING_PICTURE:
        print("Not taking new picture, using previously loaded one. If this incorrect, check 'TAKING_PICTURE' variable.")

def PaintByNumber():
    '''LOAD IMAGE'''
    # Timer Start
    print("Generating paint-by-number...")
    start_time = time.time()

    # Load original image
    # img = cv.imread("./images/pa_logo.png")
    # img = cv.imread("./images/golden_gate_bridge.jpg")
    # img = cv.imread("./images/clifford.jpg")
    # img = cv.imread("./images/color_circles.jpg")
    # img = cv.imread("./images/brad_pitt.jpg")
    # img = cv.imread("./images/mona_lisa.jpg")
    img = cv.imread("./images/capture.png")  # The video captured image


    '''AI IMAGE-TO-IMAGE GENERATION'''
    # Set to False if you don't want AI generated image
    USING_AI = False
    if not USING_AI:
        print("AI generation skipped. If this is incorrect, check 'USING_AI' variable in the script.")

    if USING_AI:
        print("AI generation beginning...")
        # Following example from Stability AI: https://platform.stability.ai/docs/features/image-to-image#Python

        # Stability API requires a PIL image, so we convert
        img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_RGB)

        # Set up environemnet 
        api_token_file = open(r"C:\Users\SFRZH\Documents\StabilityAIToken.txt")  # Replace with your correct file location
        api_token = api_token_file.read()

        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        os.environ['STABILITY_KEY'] = api_token

        # Establish connection to Stability API
        try: 
            stability_api = client.StabilityInference(
                key = os.environ['STABILITY_KEY'],
                verbose = True,  # Print debug messages
                engine = "stable-diffusion-xl-1024-v1-0"  # List of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
            )
        except:
            print("Error: Connection to Stability API failed")

        # Generation timer start
        print("Image-to-image generation started...")
        stability_start_time = time.time()

        # Generation Parameters
        answers = stability_api.generate(
            prompt="Portrait in the style of vincent van gogh's Sunflowers, beautiful paint strokes, oil painting, van gogh's colors, portrait, paint strokes visible", 
            init_image=pil_img,  # Initial image for transformation
            start_schedule=0.55,  # Strength of prompt in relation to original image
            steps=30,  # Number of intereference steps. Default is 30
            cfg_scale=7.0,  # Influences how strongly generation is guided to match prompt- higher values increase strength in which it tries to match prompt. Default 7.0
            width=512,
            height=512,
            sampler=generation.SAMPLER_K_DPMPP_2M  # Sampler to denoise generation with. Default is k_dpmpp_2m
        )

        # Trigger warning if adult content classifier is tripped
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed."
                        "Please modify the prompt and try again."
                    )
                if artifact.type == generation.ARTIFACT_IMAGE:
                    global img_generated_pil
                    img_generated_pil = Image.open(io.BytesIO(artifact.binary))

            # Generation timer end
            stability_end_time = time.time()
            stability_total_time = (stability_end_time-stability_start_time)
            print(f'Image-to-image generation succesful. Generation Time: {stability_total_time:.4f} seconds')
            pil_img.show()
            img_generated_pil.show()

            # Convert PIL image to numpy array for OpenCV processing
            img_generated = np.array(img_generated_pil)
            img = cv.cvtColor(img_generated, cv.COLOR_RGB2BGR)


    '''IMAGE PREPROCESSING'''
    # Convert image to CIELAB color space for processing
    img_LAB = cv.cvtColor(img, cv.COLOR_BGR2Lab)

    # It was found that the increasing lightness resulted in an image with more small sections which is undesirable
    # This section is being left in for possible future improvement
    # # Adjust L, a, and b channel values using Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # # The L channel represent lightness, a channel represents color spectrum from green to red, 
    # # and b channel represent color spectrum from blue to yellow
    # l, a, b = cv.split(img_LAB)
    # clahe_l = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    # clahe_a = cv.createCLAHE(clipLimit=1.0, tileGridSize=(3,3))
    # clahe_b = cv.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    # l = clahe_l.apply(l)
    # a = clahe_a.apply(a)
    # b = clahe_b.apply(b)
    # img_LAB = cv.merge((l,a,b))

    # Blur image to reduce noise for improved edge detection
    img_blur = cv.GaussianBlur(img_LAB,(7,7), sigmaX=30, sigmaY=30)

    # Reshape the image to be a 2D array with 3 channels. 
        # The value -1 means the number of rows needed is calculated automatically based on the colomns. By reshaping to a 2D array, 
        # each pixel is a row and each column represents a color (L, A, B).
        # This allows the k-means cluster algorithm to cluster similar colors together.  
    img_reshape = img_blur.reshape((-1, 3))

    # Convert to float32 for floating-point calculations
    img_reshape = np.float32(img_reshape)


    '''COLOR QUANTIZATION'''
    # Define criteria, number of clusters(K), and apply kmeans()
        # cv.TERM_CRITERIA_EPS indicates that the algorithm should stop when the specified accuracy (epsilon) is reached.
        # cv.TERM_CRITERIA_MAX_ITER indicates that the algorithm should stop after the specified number of iterations (max_iter) 1.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)  # stop criteria, epsilon, max iterations
    color_quantity = 9 # number of clusters (or colors)
    ret, label, base_colors = cv.kmeans(img_reshape, color_quantity, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)


    '''FIND UNIQUE BASE COLORS'''
    # List of LAB values from croyola 12 pack
        # L [0, 100], A [-128, 127], B [-128, 127]
    color_list_LAB = [(26.691484008200362, 46.9566933495468, 35.213282818068606),
                    (37.32462776246022, 61.73353636015208, 49.70761097914498),
                    (48.56034325016407, 60.801423247291595, 260.419969209528926),
                    (71.25097323326531, -4.660282552567285, 73.53468745541643),
                    (43.44990229796805, -48.926808323815216, 46.76834532999551),
                    (58.325844779003816, -9.83338602909467, -41.81839089809676),
                    (8.838401266601803, 25.494537383586938, -42.59741211318631),
                    (11.087927325489154, 28.22859583509954, 16.116404226973568),
                    (1.7619641595337825, -0.9567793028630311, 11.8000166591868982),
                    (84.19846444703293, 0.004543913948662492, -0.008990380233764306)]

    # List of BGR values from croyola 12 pack
    color_list_BGR = [(10, 10, 130),
                    (5, 0, 180),
                    (1, 54, 216),         
                    (1, 174, 200),
                    (4, 120, 3),
                    (130, 10, 10),
                    (180, 0, 5),
                    (216, 54, 1),
                    (200, 174, 1),
                    (3, 120, 4),
                    (1, 149, 213),
                    (0, 15, 84),
                    (66, 5, 3),
                    (6, 7, 2),
                    (210, 210, 210)]

    # Convert from the typical LAB color range to the [0,255] range OpenCV uses
    def convert_LAB_to_opencv(color_list_LAB):
        color_list_LAB_opencv = []
        for color in color_list_LAB:
            L, a, b = color
            # Scale the L channel from [0, 100] to [0, 255]
            L = L * 255 / 100
            # Scale the a and b channels from [-128, 127] to [0, 255]
            a = (a + 128)
            b = (b + 128)
            color_list_LAB_opencv.append((L, a, b))
        return color_list_LAB_opencv
    color_list_LAB_opencv = convert_LAB_to_opencv(color_list_LAB)

    def find_unique_base_colors(base_colors, color_list):
        unique_base_colors = []
        for base_color in base_colors:
            # Reset min_distance for each base_color
            min_distance = float('inf')
            for unique_color in color_list:
                # Convert the color to a numpy array
                base_color = np.array(base_color)
                unique_color = np.array(unique_color)

                # Calculate the Euclidean distance between the target color and the color
                distance = np.sqrt(np.sum((unique_color - base_color) ** 2))

                # If the distance is smaller than the current minimum distance, update the minimum distance and the most similar color
                if distance < min_distance:
                    min_distance = distance
                    most_similar_color = unique_color
                    # print("most similar", most_similar_color)
            unique_base_colors.append(most_similar_color)
            # Remove the color that was just chosen as the most_similar_color
            color_list = [color for color in color_list if color not in most_similar_color] 
        
        return unique_base_colors

    unique_base_colors = [find_unique_base_colors(base_colors, color_list_LAB_opencv)]
    unique_base_colors = np.array([np.array(base_color) for base_color in unique_base_colors])[0]  # convert to numpy array
    unique_base_colors = np.uint8(unique_base_colors) 
    img_simplified = unique_base_colors[label.flatten()]  # Replace each pixel with its corresponding base color
    img_simplified = img_simplified.reshape((img.shape))
    img_simplified = cv.cvtColor(img_simplified, cv.COLOR_LAB2BGR) 

    def LAB_to_BGR(LAB_color):
        # Convert the LAB color to a 2D array
        LAB_color_2d = np.uint8([[LAB_color]])

        # Convert the LAB color to BGR
        bgr_color = cv.cvtColor(LAB_color_2d, cv.COLOR_LAB2BGR)

        return bgr_color[0][0]
    unique_base_colors = [LAB_to_BGR(LAB_color) for LAB_color in unique_base_colors]  # convert unique_base_colors to BGR for color masking operations

    unique_base_colors = np.array([np.array(base_color) for base_color in unique_base_colors])


    '''COLOR MASKING'''
    # For each base_color, calculate max and min values to use as mask 
    tol = 0  # tolerance 
    bgr_color_limit_dict = {}
    for i, bgr_color in enumerate(unique_base_colors):
        b_val = unique_base_colors[i][0]
        g_val = unique_base_colors[i][1]
        r_val = unique_base_colors[i][2]
        bgr_color_limit_dict[i] = np.array([b_val - tol, g_val - tol, r_val - tol]), np.array([b_val + tol, g_val + tol, r_val + tol])

    # Create masks
    mask_dict = {}
    for i, color_limit in bgr_color_limit_dict.items():
        # Each pixel that falls in the color range is set to white (255), the rest are set to black (0)
        mask_dict[i] = cv.inRange(img_simplified, bgr_color_limit_dict[i][0], bgr_color_limit_dict[i][1]) 

    # Apply masks
    img_mask_dict = {}
    for i, mask in mask_dict.items():
        # Keeps the pixel values from img_simplified where the mask is white
        img_mask_dict[i] = cv.bitwise_and(img_simplified, img_simplified, mask = mask_dict[i])


    '''FIND AND DRAW CONTOURS AND LABELS'''
    img_size = img.shape[:2]  # Columns and rows
    area_limit = 500  # Don't label feature that is too small
    width_limit = 10  # Don't label feature that is too thin
    border_size = 1

    def find_contours(img_mask, img_thresh):
        # Following method from pyimagesearch.com (https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/)
        contours, hierarchy = cv.findContours(img_thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours([contours, hierarchy])  # Extract contours and returns them as a list. Output of cv.findContours can be different depending on version being used
        return hierarchy, contours

    # hierarchy_dict = {}  # Contour hierarchy information
    cntr_dict = {}  # Values are lists of numpy arrays, each array represents a contour
    for i, img_mask in img_mask_dict.items():
        _, contours = find_contours(img_mask, mask_dict[i])
        cntr_dict[i] = contours
        # hierarchy_dict[i] = hierarchy

    # Check if blob is on the edge of the image
    def blob_is_on_image_edge(x_pos, y_pos, width, height, img_size = img_size):
        return x_pos == 0 or y_pos == 0 or (x_pos + width) == img_size[1] or (y_pos + height) == img_size[0]

    # Add border to ensure distanceTransform recognizes edge of photo
    def add_border(img, border_size=border_size):
        img_border = cv.copyMakeBorder(img, 
                                    top= border_size,
                                    bottom= border_size,
                                    left= border_size,
                                    right= border_size,
                                    borderType= cv.BORDER_CONSTANT,
                                    value= [0, 0, 0])
        return img_border

    def find_label_location(blob):
        # Calculate shortest distance from each white pixel to the closest black pixel
        dist_transform = cv.distanceTransform(blob, cv.DIST_L2, 3)

        # Return the largest distance and location of the pixel (most space for a clear number label)
        _,max_val,_, max_loc = cv.minMaxLoc(dist_transform)
        return max_val, max_loc

    def draw_empty_contours_and_labels(mask):
        label_location_list = []  # Initialize list to store label locations

        # Find all 'blobs' in threshold image
        (total_labels, label_ids, stats, _) = cv.connectedComponentsWithStats(mask, 4, cv.CV_32S)

        # Initialize empty mask
        empty_contour = np.zeros(mask.shape, dtype="uint8")

        for blob_id in range(1,total_labels):
            x_pos, y_pos, width, height, area = stats[blob_id]
            blob = (label_ids == blob_id).astype("uint8") * 255  # draw blob in white

            # If the blob is on the edge of the image, add a border to distanceTransform will recognize edge
            if blob_is_on_image_edge(x_pos, y_pos, width, height):
                blob_with_border = add_border(blob)
                max_val, max_loc = find_label_location(blob_with_border)
            
            else:
                max_val, max_loc = find_label_location(blob)

            # If the blob area and the distance from any wall is large enough
            if area > area_limit and max_val > width_limit:
                empty_contour = cv.bitwise_or(empty_contour, blob)
                if max_val > width_limit:
                    label_location_list.append(max_loc)
        return empty_contour, label_location_list

    label_locations_dict = {}
    empty_contours_dict = {}
    for i, mask in mask_dict.items():
        empty_contour, label_location_list = draw_empty_contours_and_labels(mask)
        empty_contours_dict[i] = empty_contour
        label_locations_dict[i] = label_location_list


    '''COMBINING IMAGES'''
    def blend_mask_and_contours(mask, contour_image):
        three_channel_thresh_image = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        blended = cv.addWeighted(three_channel_thresh_image, 1, contour_image, 1, 0)
        return blended

    img_blended_dict = {}
    for i, empty_contours in empty_contours_dict.items():
        img_blended_dict[i] = blend_mask_and_contours(empty_contours, img_mask_dict[i])


    def combine_all(previous_image, current_image):
        final_image = cv.addWeighted(previous_image, 1, current_image, 1, 0)
        return final_image

    for i, blended_image in img_blended_dict.items():
        if i == 0:
            final_image = blended_image
        else:
            previous_image = final_image
            current_image = blended_image
            final_image = combine_all(previous_image, current_image)

    # Draw all contour outlines (for coloring in)
    for i, contour_list in cntr_dict.items():
        for j, contour in enumerate(contour_list):
            final_image = cv.drawContours(final_image, [contour], -1, (0,0,0), 1)

    # Label with corresponding numbers
    font = cv.FONT_HERSHEY_COMPLEX
    font_scale = 0.6
    font_thickness = 1
    font_color = (0,0,0)
    for color_number, label_location_list in label_locations_dict.items():
        for label_location in label_location_list:
            # If area is filled with color, don't label
            y_pos = int(label_location[1])
            x_pos = int(label_location[0])
            b_color = img_mask_dict[color_number][y_pos, x_pos, 0]
            g_color = img_mask_dict[color_number][y_pos, x_pos, 1]
            r_color = img_mask_dict[color_number][y_pos, x_pos, 2]
            if b_color != 0 or g_color != 0 or r_color != 0:
                label_location_circ = (x_pos, y_pos)
                # Center text using method by xcsrz (https://gist.github.com/xcsrz/8938a5d4a47976c745407fe2788c813a)
                text_size = cv.getTextSize(str(color_number + 1), font, font_scale, font_thickness)[0]
                text_width = text_size[0]
                text_height = text_size[1]
                # Center horizontally (move position left by half the width), center vertically (move position down by half the height)
                label_location = (int(x_pos - (text_width/2)), int(y_pos + (text_height/2)))
                cv.putText(final_image, str(color_number + 1), label_location, font, font_scale, font_color, font_thickness)
                # cv.circle(final_image, label_location_circ, 3, (0,0,255), -1)  # Highlight label location (uncomment to check placement)

    # Timer End
    end_time = time.time()
    total_time = (end_time-start_time)
    print(f'Script Complete. Total Run Time: {total_time:.4f} seconds')


    '''MATPLOTLIB DISPLAY'''
    # Used method from geeksforgeeks.org (https://www.geeksforgeeks.org/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/)
    fig = plt.figure(figsize=(10,7))

    rows = 2
    columns = 2

    # Add subplot in first position
    fig.add_subplot(rows, columns, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Original Image")

    # Add subplot in second position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(cv.cvtColor(img_LAB, cv.COLOR_LAB2RGB))
    plt.axis('off')
    plt.title(f"CIELAB Image")

    # Add subplot in third position
    fig.add_subplot(rows, columns, 3)
    plt.imshow(cv.cvtColor(img_simplified, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Image with Grouped Colors, Color Quantity: {color_quantity}")


    # Add subplot in fourth position
    fig.add_subplot(rows, columns, 4)
    plt.imshow(cv.cvtColor(final_image, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Final Paint-by-Number")

    # plt.show()  # display matplotlib figures 


    '''IMSHOW'''
    # cv.imshow("Original Image", img)
    # cv.imshow("Blurred Image", img_blur)
    # cv.imshow("Simplified Image", img_simplified)
    # cv.imshow("Simplified Image Edges", edges)
    # cv.imshow("Mask Image 1", img_mask_dict[0])
    # cv.imshow("Mask Image 1 Gray Scale", img_gray)
    # cv.imshow("Mask Image 1 Threshold", img_thresh)
    # cv.imshow("Mask Image 2", img_mask_dict[1])
    # cv.imshow("Mask Image 3", img_mask_dict[2])
    # cv.imshow("Final Image", final_image)

    # cv.waitKey(0)  # keep images open until any key is pressed


    '''GENERATE PDF'''
    # Convert numpy array to .jpg format
    final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)
    final_image = Image.fromarray(final_image)
    final_image.save("./images/final_image.jpg")
    CreatePDF("./images/final_image.jpg", unique_base_colors)

def PrintPDF():
    try: 
        printer_name = Printer.get_default_printer_name()
        print("Printer:", printer_name)
        print("Printing...")
        # Requires PDF reader installed on computer to work properly
        win32api.ShellExecute(0, "print", "PAintByNumber.pdf", '/d:"%s"' % printer_name, ".", 0)
    except:
        print("Unable to Print")
