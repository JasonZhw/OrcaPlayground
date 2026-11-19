"""把技术报告 Markdown 转成 Word（供分发/粘贴到飞书）。

用法：
    python docs/build_report_docx.py
    # 输入 docs/medication_sorting_technical_report.md
    # 输出 docs/medication_sorting_technical_report.docx

只覆盖本报告用到的语法：标题、段落、列表、表格、代码块、图片、引用、粗体、行内代码。
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "medication_sorting_technical_report.md"
DST = ROOT / "medication_sorting_technical_report.docx"

CN_FONT = "微软雅黑"
MONO_FONT = "Consolas"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
SUB_ACCENT = RGBColor(0x2E, 0x74, 0xB5)
GREY = RGBColor(0x60, 0x60, 0x60)


def _cjk(run, name: str = CN_FONT) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def add_runs(paragraph, text: str, size: float = 10.5) -> None:
    """按 **粗体** 与 `代码` 拆分并写入 run。"""
    for token in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            run.font.size = Pt(size)
            _cjk(run)
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = MONO_FONT
            run.font.size = Pt(size - 1)
            run.font.color.rgb = RGBColor(0xB0, 0x30, 0x30)
        else:
            run = paragraph.add_run(token)
            run.font.size = Pt(size)
            _cjk(run)


def build() -> Path:
    doc = Document()
    section = doc.sections[0]
    section.left_margin = section.right_margin = Cm(2.2)
    section.top_margin = section.bottom_margin = Cm(2.0)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    i, in_code, code_buf = 0, False, []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if not in_code:
                in_code, code_buf = True, []
            else:
                paragraph = doc.add_paragraph()
                paragraph.paragraph_format.left_indent = Cm(0.4)
                paragraph.paragraph_format.space_after = Pt(8)
                pPr = paragraph._p.get_or_add_pPr()
                pPr.append(pPr.makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:fill"): "F4F4F4"}))
                run = paragraph.add_run("\n".join(code_buf))
                run.font.name = MONO_FONT
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
                in_code = False
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if not line.strip() or line.strip() == "---":
            i += 1
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            level, text = len(heading.group(1)), heading.group(2)
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.space_before = Pt(14 if level == 1 else 10)
            paragraph.paragraph_format.space_after = Pt(6)
            run = paragraph.add_run(text)
            run.bold = True
            run.font.size = Pt(20 if level == 1 else 14 if level == 2 else 11.5)
            run.font.color.rgb = ACCENT if level <= 2 else SUB_ACCENT
            _cjk(run)
            i += 1
            continue

        if line.startswith(">"):
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.6)
            add_runs(paragraph, line.lstrip("> ").strip(), size=9.5)
            for run in paragraph.runs:
                run.font.color.rgb = GREY
            i += 1
            continue

        image = re.match(r"^!\[[^\]]*\]\(([^)]+)\)", line.strip())
        if image:
            path = (ROOT / image.group(1)).resolve()
            if path.exists():
                doc.add_picture(str(path), width=Cm(15.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        if line.strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") and c for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                table = doc.add_table(rows=0, cols=ncol)
                table.style = "Light Grid Accent 1"
                for ridx, row in enumerate(rows):
                    cells = table.add_row().cells
                    for cidx in range(ncol):
                        cells[cidx].text = ""
                        paragraph = cells[cidx].paragraphs[0]
                        add_runs(paragraph, row[cidx] if cidx < len(row) else "", size=9)
                        if ridx == 0:
                            for run in paragraph.runs:
                                run.bold = True
                doc.add_paragraph()
            continue

        bullet = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if bullet:
            paragraph = doc.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.left_indent = Cm(0.7 + 0.5 * (len(bullet.group(1)) // 2))
            paragraph.paragraph_format.space_after = Pt(3)
            add_runs(paragraph, bullet.group(2))
            i += 1
            continue

        numbered = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if numbered:
            paragraph = doc.add_paragraph(style="List Number")
            paragraph.paragraph_format.space_after = Pt(3)
            add_runs(paragraph, numbered.group(3))
            i += 1
            continue

        # 普通段落：合并后续软换行
        buffer = [line.strip()]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            stripped = nxt.strip()
            if (not stripped or stripped.startswith(("#", "|", "-", "*", ">", "```"))
                    or re.match(r"^\d+\.\s", stripped) or stripped == "---"):
                break
            prev = buffer[-1]
            joiner = " " if (prev[-1:].isascii() and prev[-1:].isalnum()
                             and stripped[:1].isascii() and stripped[:1].isalnum()) else ""
            buffer[-1] = prev + joiner + stripped
            i += 1
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(6)
        paragraph.paragraph_format.line_spacing = 1.35
        add_runs(paragraph, "".join(buffer))

    doc.save(DST)
    return DST


if __name__ == "__main__":
    out = build()
    print(f"已生成 {out}")
