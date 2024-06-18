from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.graphics import shapes

pa_logo_path = "./images/pa_logo.png"
sfdw_logo_path = "./images/SFDW_Logo_2024.png"
img_width = None
img_height = 0.4*inch
color_square_size = 0.7*inch # [inch]

def CreatePDF(img, base_colors):
    # Create blank PDF 
    canvas = Canvas("PAintByNumber.pdf", pagesize=LETTER)

    # Add the paint-by-number image
    canvas.drawImage(img, 30, 310, width=550, preserveAspectRatio=True)

    # Add PA logo 
    canvas.drawImage(pa_logo_path, 7.4*inch, 0.3*inch, height = img_height, preserveAspectRatio=True)

    # Add SFDW logo
    canvas.drawImage(sfdw_logo_path, 4.2*inch, 0.3*inch, height = img_height, preserveAspectRatio=True)

    # Add color/number legend
    for num in range(len(base_colors)):
        color = base_colors[num]

        # Create 3x3 grid of color squares
        if 0 <= num <= 2:
            center_y = 3.7*inch
            if num == 0:
                center_x = (8.5/3)*inch
            if num == 1:   
                center_x = (8.5/2)*inch
            if num == 2:
                center_x = ((2*8.5)/3)*inch

        if 3 <= num <= 5:
            center_y = 2.4*inch
            if num == 3:
                center_x = (8.5/3)*inch
            if num == 4:   
                center_x = (8.5/2)*inch
            if num == 5:
                center_x = ((2*8.5)/3)*inch

        if 6 <= num <= 8:
            center_y = 1*inch
            if num == 6:
                center_x = (8.5/3)*inch
            if num == 7:   
                center_x = (8.5/2)*inch
            if num == 8:
                center_x = ((2*8.5)/3)*inch

        # Rectangle position is placed by bottom left corner, so that position is calculated
        pos_x = center_x - (color_square_size/2)
        pos_y = center_y - (color_square_size/2)

        # Map 0-255 value to 0-1 range used by reportlab
        # The base colors are in BGR format so the color orders are reversed
        R_map = num_to_range(color[2])
        G_map = num_to_range(color[1])
        B_map = num_to_range(color[0])
        canvas.setFillColorRGB(R_map, G_map, B_map)

        # Draw the filled recangle
        canvas.rect(pos_x, pos_y, color_square_size, color_square_size, fill=True)
        # Write number next to it
        canvas.setFont("Helvetica", 16)
        canvas.setFillColorRGB(0, 0, 0)  # Reset font color to black
        canvas.drawString(pos_x-(0.24*inch), pos_y+(0.05*inch), str(num + 1) + ".")
    
    # Save pdf to parent directory
    canvas.save()

# Source: https://www.30secondsofcode.org/python/s/num-to-range/
def num_to_range(num, inMin=0, inMax=255, outMin=0, outMax=1):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin))