"""Generates two small fictional "Demo Bank" annual reports so the pipeline can be tried before downloading real ones.

Every figure here is made up. Delete these PDFs from data/raw once you add the real DBS, OCBC and UOB reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# made-up numbers for each year, so the two reports differ and year filtering can be tested
FIGURES = {
    2024: {"profit": "4.12", "income": "9.80", "roe": "15.1", "cet1": "14.2", "nim": "2.05", "dividend": "1.60", "staff": "18,400", "green": "21"},
    2025: {"profit": "4.57", "income": "10.45", "roe": "16.3", "cet1": "14.8", "nim": "2.11", "dividend": "1.85", "staff": "19,050", "green": "27"},
}


def build(year: int, out_dir: Path) -> Path:
    f = FIGURES[year]
    prev = FIGURES.get(year - 1)
    path = out_dir / f"demo_{year}.pdf"
    styles = getSampleStyleSheet()
    body, h1 = styles["BodyText"], styles["Heading1"]

    def footer(canvas, doc):
        # a running header and footer on every page, like real annual reports have
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 815, f"Demo Bank Annual Report {year}")
        canvas.drawString(520, 25, str(doc.page))

    story = [
        Paragraph("Chairman's Statement", h1),
        Paragraph(
            f"Demo Bank delivered a record year in {year}. Net profit rose to SGD {f['profit']} billion, "
            f"supported by higher fee income and disciplined cost management. The Board has proposed a total "
            f"dividend of SGD {f['dividend']} per share for the financial year.", body),
        Spacer(1, 12),
        Paragraph(
            "Our strategy remains focused on digital banking, wealth management for the region, and a strong "
            "balance sheet. We continued to invest in technology, including generative AI tools that help "
            "relationship managers prepare client reviews faster.", body),
        PageBreak(),
        Paragraph("Financial Highlights", h1),
        Paragraph("The table below summarises the key financial results for the year.", body),
        Spacer(1, 8),
    ]

    rows = [["Metric", str(year), str(year - 1) if prev else "Prior year"],
            ["Net profit (SGD billion)", f["profit"], prev["profit"] if prev else "3.70"],
            ["Total income (SGD billion)", f["income"], prev["income"] if prev else "9.10"],
            ["Return on equity (%)", f["roe"], prev["roe"] if prev else "14.0"],
            ["Common Equity Tier 1 ratio (%)", f["cet1"], prev["cet1"] if prev else "13.9"],
            ["Net interest margin (%)", f["nim"], prev["nim"] if prev else "1.98"]]
    table = Table(rows, colWidths=[220, 90, 90])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story += [table, PageBreak()]

    story += [
        Paragraph("Risk Management", h1),
        Paragraph(
            "Credit risk is the largest risk the bank faces. We manage it through borrower limits, sector "
            "concentration limits and regular stress testing. Non-performing loans stayed low during the year "
            "and the allowance coverage remained above 100 percent.", body),
        Spacer(1, 12),
        Paragraph(
            "Technology risk is overseen by the Board Risk Committee. Major system changes go through "
            "independent review, and critical services are tested for recovery twice a year.", body),
        PageBreak(),
        Paragraph("Sustainability", h1),
        Paragraph(
            f"Sustainable financing commitments reached SGD {f['green']} billion by the end of {year}. "
            "The bank aims to reach net zero operational emissions by 2030 and net zero financed emissions by 2050.", body),
        PageBreak(),
        Paragraph("Our People", h1),
        Paragraph(
            f"Demo Bank employed {f['staff']} people at the end of {year}. Every employee completed at least "
            "20 hours of training, with a focus on data skills and responsible use of AI.", body),
    ]
    SimpleDocTemplate(str(path), pagesize=A4, topMargin=60).build(story, onFirstPage=footer, onLaterPages=footer)
    return path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    for y in FIGURES:
        print("wrote", build(y, target))
