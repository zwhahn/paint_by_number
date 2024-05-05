from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import LETTER

# Create blank PDF 
canvas = Canvas("PAintByNumber.pdf", pagesize=LETTER)

canvas.drawString(50, 50,"Hello World")

# Save pdf
canvas.save()

