from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import LETTER

def CreatePDF(img):
    # Create blank PDF 
    canvas = Canvas("PAintByNumber.pdf", pagesize=LETTER)
    canvas.drawString(50, 50,"Hello World")

    canvas.drawImage(img, 30, 310, width=550, preserveAspectRatio=True)

    # Save pdf to parent directory
    canvas.save()