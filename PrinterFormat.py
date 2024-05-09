from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import LETTER

def CreatePDF(img):
    # Create blank PDF 
    canvas = Canvas("PAintByNumber.pdf", pagesize=LETTER)

    # Add the paint-by-number image
    canvas.drawImage(img, 30, 310, width=550, preserveAspectRatio=True)

    # Add PA logo 
    canvas.drawImage("./images/pa_logo.png", 0, 0, width=150, preserveAspectRatio=True)

    # Save pdf to parent directory
    canvas.save()