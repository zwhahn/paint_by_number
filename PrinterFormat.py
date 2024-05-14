from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch

pa_logo_path = "./images/pa_logo.png"

def CreatePDF(img):
    # Create blank PDF 
    canvas = Canvas("PAintByNumber.pdf", pagesize=LETTER)

    # Add the paint-by-number image
    canvas.drawImage(img, 30, 310, width=550, preserveAspectRatio=True)

    # Add PA logo 
    canvas.drawImage(pa_logo_path, 0, 0, preserveAspectRatio=True)

    # Save pdf to parent directory
    canvas.save()