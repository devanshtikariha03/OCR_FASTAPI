#extractors.py

import os
import pandas as pd
from typing import List, Tuple

from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
import pdfplumber
import camelot

def is_scanned_pdf(filepath: str) -> bool:
    try:
        txt = extract_text(filepath)
        return len((txt or "").strip()) == 0
    except Exception:
        return True

def ocr_page_to_df(pil_image):
    df = pytesseract.image_to_data(pil_image, lang="eng", output_type=Output.DATAFRAME)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    raw_text = "\n".join(
        df.groupby(["page_num","block_num","par_num","line_num"])["text"]
          .apply(lambda s: " ".join(str(x) for x in s if str(x).strip()))
          .tolist()
    )
    return raw_text, df

def extract_textpdf(filepath: str):
    raw_text_pages: List[str] = []
    words_sheets: List[Tuple[str, pd.DataFrame]] = []
    table_sheets: List[Tuple[str, pd.DataFrame]] = []

    with pdfplumber.open(filepath) as pdf:
        for p_idx, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            words_df = pd.DataFrame(words)
            words_sheets.append((f"WORDS_Page{p_idx}", words_df))

            raw_text_pages.append(page.extract_text() or "")

            tables = page.extract_tables()
            for t_idx, table in enumerate(tables, start=1):
                df = pd.DataFrame(table)
                table_sheets.append((f"TABLE_Page{p_idx}_T{t_idx}", df))

    # Try Camelot for better tables (text PDFs only)
    try:
        cam_tables = camelot.read_pdf(filepath, pages="all")
        for i, t in enumerate(cam_tables, start=1):
            table_sheets.append((f"TABLE_Camelot_T{i}", t.df))
    except Exception:
        pass
    return raw_text_pages, words_sheets, table_sheets

def extract_scanned(filepath: str, dpi: int = 300):
    raw_text_pages: List[str] = []
    words_sheets: List[Tuple[str, pd.DataFrame]] = []
    table_sheets: List[Tuple[str, pd.DataFrame]] = []  # OCR table detection is non-trivial; omit for MVP

    # Windows requires Poppler path if not on PATH
    images = convert_from_path(filepath, dpi=dpi)
    for p_idx, img in enumerate(images, start=1):
        raw_txt, words_df = ocr_page_to_df(img)
        raw_text_pages.append(raw_txt)
        words_sheets.append((f"WORDS_Page{p_idx}", words_df))

    return raw_text_pages, words_sheets, table_sheets

def extract_all(filepath: str):
    """
    Returns:
      raw_text_pages: List[str]
      words_sheets: List[(sheet_name, DataFrame)]
      table_sheets: List[(sheet_name, DataFrame)]
    """
    if is_scanned_pdf(filepath):
        return extract_scanned(filepath)
    else:
        return extract_textpdf(filepath)
