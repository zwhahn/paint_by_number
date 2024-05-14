from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.graphics import shapes

pa_logo_path = "./images/pa_logo.png"

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
        print("color:", color)
        print("num:", num)
        pos_x = num*inch
        pos_y = 1*inch
        R_map = num_to_range(color[2])
        G_map = num_to_range(color[1])
        B_map = num_to_range(color[0])
        canvas.setFillColorRGB(R_map, G_map, B_map)
        canvas.rect(pos_x, pos_y, 0.5*inch, 0.5*inch, fill=True)
    
    # Save pdf to parent directory
    canvas.save()

# Source: https://www.30secondsofcode.org/python/s/num-to-range/
def num_to_range(num, inMin=0, inMax=255, outMin=0, outMax=1):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin))