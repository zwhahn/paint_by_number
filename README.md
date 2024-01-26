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

#### Image Input
Images are loaded using cv.imread("/file_path"). Example images have been stored in the 'images' folder and can be easily swapped by uncommenting the corresponding image line.

#### Image-to-Image Generation (optional)
The method of calling the Stability AI comes from the [Stability AI's python tutorial](https://platform.stability.ai/docs/features/image-to-image#Python). A PIL image must be used, so the OpenCV input image is converted. The output of the API call is also a PIL image, so this image is converted back into a numpy array for usage in the rest of the script. In the Generation Parameters section, the user can edit the prompt and the strength of the prompt.

The user must create their own Stability AI account in order to get a unique API token. With the new account comes 25 free credits, additional credit may be purchased. Save the unique API token locally in a .txt file (not in the repository!) and update the api_token_file with the path to where the API token is saved.

There is filter that throws a warning if the adult classifier is tripped to ensure no inappropriate images are shown.

#### Color Quantization
The script performs color quantization to reduce the number of colors in the output image to a user-set number, making it possible to paint by hand.