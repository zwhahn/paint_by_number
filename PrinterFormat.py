from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import LETTER

# Create blank PDF 
canvas = Canvas("PAintByNumber.pdf", pagesize=LETTER)

# Save pdf
canvas.save()

