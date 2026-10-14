import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
import pymupdf
doc=pymupdf.open(HERE/'audit_sources/DeepSeek_V41_Tech_Report.pdf')
for i in [0,1,2,3,4,5,6,7,8,9]:
    page=doc[i]
    page.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(HERE/'audit_sources'/('report_page_'+str(i+1)+'.png'))
print('Rendered report pages 1–10')
