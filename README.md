# Paint-By-Number

## Overview
This project is a Python script that transforms any image into a paint-by-number style image primarily using the OenCV library. Additionally, the Stability AI platform is integrated to allow the user to perform Image-to-Image generation based on the original input image. Using this AI generation is optional. 

## How it Works
The script has 5 main sections:
1. Image Input
2. Image-to-Image Generation (optional)
3. Color Quantization
4. Color Masking
5. Finding and Drawing Contours/Labels
6. Combining Final Images
7. Image Display

#### Image Input
Images are loaded using cv.imread("/file_path"). Example images have been stored in the 'images' folder and can be easily swapped by uncommenting the corresponding image line.


The script performs color quantization to reduce the number of colors in the output image to a user-set number, making it possible to paint by hand. The script also labels the regions of the output image with numbers corresponding to the colors, and generates a color palette with the matching numbers and color codes.