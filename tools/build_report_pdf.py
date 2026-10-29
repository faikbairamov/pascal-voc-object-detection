"""Typeset the checked-in technical report; requires ReportLab (authoring only)."""
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'output/pdf/technical_report.pdf'


def inline(text):
    text = text.replace('—','-').replace('–','-').replace('’',"'").replace('“','"').replace('”','"')
    text = escape(text)
    text = re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    return re.sub(r'`([^`]+)`',r'<font name="Courier" size="8">\1</font>',text)


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#d4dfdf'))
    canvas.line(42,35,A4[0]-42,35)
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#54706f'))
    canvas.drawString(42,22,'Faik Bairamov | Pascal VOC object detection | 29 October 2026')
    canvas.drawRightString(A4[0]-42,22,f'{doc.page}')


def main():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportTitle',fontName='Helvetica-Bold',fontSize=23,leading=27,
                              textColor=colors.HexColor('#123d38'),spaceAfter=15))
    styles.add(ParagraphStyle('ReportHeading',fontName='Helvetica-Bold',fontSize=12,leading=15,
                              textColor=colors.HexColor('#176f60'),spaceBefore=14,spaceAfter=7,
                              keepWithNext=True))
    styles.add(ParagraphStyle('ReportBody',fontName='Helvetica',fontSize=9.5,leading=13.2,
                              textColor=colors.HexColor('#253b3b'),spaceAfter=7))
    styles.add(ParagraphStyle('Cell',parent=styles['ReportBody'],fontSize=8.5,leading=11,spaceAfter=0))
    styles.add(ParagraphStyle('Caption',parent=styles['ReportBody'],fontSize=9,leading=12,
                              textColor=colors.HexColor('#54706f')))
    lines=(ROOT/'reports/technical_report.md').read_text().splitlines()
    story=[];i=0
    while i < len(lines):
        line=lines[i]
        if not line.strip(): i+=1;continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]),styles['ReportTitle']));i+=1;continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]),styles['ReportHeading']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):
                    rows.append([Paragraph(inline(c),styles['Cell']) for c in cells])
                i+=1
            width=A4[0]-84
            n=len(rows[0]); widths=[width*.52]+[width*.48/(n-1)]*(n-1)
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2f1ec')),
                                       ('VALIGN',(0,0),(-1,-1),'TOP'),
                                       ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7fa99e')),
                                       ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d7e2df')),
                                       ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.extend([table,Spacer(1,10)]);continue
        paragraph=[]
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|')):
            paragraph.append(lines[i]);i+=1
        story.append(Paragraph(inline(' '.join(paragraph)),styles['ReportBody']))
    story.extend([Spacer(1,12),Paragraph('Performance by category',styles['ReportHeading'])])
    for name,caption in [('dev_per_category.png','Development AP by category. Additional precision-recall curves and error images accompany the report under reports/figures/.')]:
        image=Image(str(ROOT/'reports/figures'/name))
        scale=min((A4[0]-100)/image.imageWidth,310/image.imageHeight)
        image.drawWidth=image.imageWidth*scale;image.drawHeight=image.imageHeight*scale
        story.extend([image,Paragraph(caption,styles['Caption']),Spacer(1,10)])
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(DESTINATION),pagesize=A4,rightMargin=42,leftMargin=42,
                          topMargin=40,bottomMargin=48,title='Pascal VOC Object Detection - Technical Report',
                          author='Faik Bairamov')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(DESTINATION)


if __name__ == '__main__': main()
