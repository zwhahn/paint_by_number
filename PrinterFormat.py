from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.graphics import shapes

pa_logo_path = "./images/pa_logo.png"
color_square_size = 0.5*inch # [inch]

def CreatePDF(img, base_colors):
    # Create blank PDF 
    canvas = Canvas("PAintByNumber.pdf", pagesize=LETTER)

    # Add the paint-by-number image
    canvas.drawImage(img, 30, 310, width=550, preserveAspectRatio=True)

    # Add PA logo 
    canvas.drawImage(pa_logo_path, 7.1*inch, 0.3*inch, preserveAspectRatio=True)

    # Add color/number legend
    for num in range(len(base_colors)):
        color = base_colors[num]

        # Desired center position
        if 0 <= num <= 2:
            center_y = 3*inch
            if num == 0:
                center_x = (8.5/3)*inch
            if num == 1:   
                center_x = (8.5/2)*inch
            if num == 2:
                center_x = ((2*8.5)/3)*inch

        if 3 <= num <= 5:
            center_y = 2*inch
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

        print("center_x:", center_x)
        print("center_y:", center_y)

        # Rectangle position is placed by bottom left corner, so that position is calculated
        pos_x = center_x - (color_square_size/2)
        pos_y = center_y - (color_square_size/2)

        print("pos_x:", pos_x)
        print("pos_y:", pos_y)

        # Map 0-255 value to 0-1 range used by reportlab
        # The base colors are in BGR format so the color orders are reversed
        R_map = num_to_range(color[2])
        G_map = num_to_range(color[1])
        B_map = num_to_range(color[0])
        canvas.setFillColorRGB(R_map, G_map, B_map)

        # Draw the filled recangle
        canvas.rect(pos_x, pos_y, color_square_size, color_square_size, fill=True)
    
    # Save pdf to parent directory
    canvas.save()

# Source: https://www.30secondsofcode.org/python/s/num-to-range/
def num_to_range(num, inMin=0, inMax=255, outMin=0, outMax=1):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin))