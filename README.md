# Paint-By-Number

## Overview
This project is a Python script that transforms any image into a paint-by-number style image primarily using the OenCV library. Additionally, the Stability AI platform is integrated to allow the user to perform Image-to-Image generation based on the original input image. Using this AI generation is optional. 

## How it Works
The script has 7 main sections:
1. Image Input
2. Image-to-Image Generation (optional)
3. Color Quantization
4. Color Masking
5. Finding and Drawing Contours/Labels
6. Combining Final Images
7. Image Display

### Image Input
Images are loaded using cv.imread("/file_path"). Example images have been stored in the 'images' folder and can be easily swapped by uncommenting the corresponding image line.

### Image-to-Image Generation (optional)
The method of calling the Stability AI comes from the [Stability AI's python tutorial](https://platform.stability.ai/docs/features/image-to-image#Python). A PIL image must be used, so the OpenCV input image is converted. The output of the API call is also a PIL image, so this image is converted back into a numpy array for usage in the rest of the script. In the Generation Parameters section, the user can edit the prompt and the strength of the prompt.

The user must create their own Stability AI account in order to get a unique API token. With the new account comes 25 free credits, additional credit may be purchased. Save the unique API token locally in a .txt file (not in the repository!) and update the api_token_file with the path to where the API token is saved.

There is a filter that throws a warning if the adult classifier is tripped to ensure no inappropriate images are shown.

Image-to-image generation is not needed for the script to run succesfully. If the user prefers paint-by-number of the original image than this section can be removed.

### Color Quantization
The script performs color quantization using k-means clustering to reduce the number of colors in the output image to a user-set number (variable name is *color_quantity*), making it possible to paint by hand.

This is accomplished by first applying a Gaussian blur is applied to the image, which is a way of smoothing out the pixels and making them less sharp. This helps to reduce the noise and make the edges more clear.

Next, the k-means clustering algorithm is run, which is a way of grouping similar data points together. The code uses the cv.kmeans function, which takes six arguments: the input data, **the number of clusters (*color_quantity*)**, the initial labels, the criteria for stopping the algorithm, the number of attempts, and the method of choosing the initial cluster centers. The output of the cv.kmeans function are three values: the final error, the final labels, and the final cluster centers. The final labels are the final colors used for painting, these values are saved in the variable *base_colors*.

### Color Masking
 A mask is a way of selecting or hiding certain parts of an image based on some criteria. The is done in three steps:

1. It calculates the maximum and minimum values for each base color, using a tolerance of 5. This means that any pixel that is within 5 units of the base color will be considered part of the same cluster. It stores these values in a dictionary, where the key is the cluster index and the value is a pair of arrays representing the lower and upper limits of the color range.
2. It creates masks for each base color, using the cv.inRange function. This function takes the simplified image, the lower and upper limits of the color range, and returns a binary image where each pixel that falls within the range is set to white (255) and the rest are set to black (0). It stores these masks in another dictionary, where the key is the cluster index and the value is the mask image.
3. It applies the masks to the simplified image, using the cv.bitwise_and function. This function takes the simplified image, a copy of the simplified image, and a mask image, and returns an image where only the pixels that are white in the mask are kept from the simplified image. It stores these masked images in a third dictionary, where the key is the cluster index and the value is the masked image.

### Finding and Drawing Contours/Labels
Using the openCV function connectComponents, the script is now able to break masked region into independent areas or 'blobs'. 