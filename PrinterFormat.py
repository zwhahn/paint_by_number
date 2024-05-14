from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.graphics import shapes

pa_logo_path = "./images/pa_logo.png"
color_square_size = 0.5*inch

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
        pos_x = num*inch
        pos_y = 1*inch

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