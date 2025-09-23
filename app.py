# app.py — JSON-only extraction API for PDF + Excel + Images (NaN/Inf-safe)
# Run:  python -m uvicorn app:app --reload

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import io
import os
import math

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from uuid import uuid4  # <-- added

# ---- PDF text tools ----
import pdfplumber                          # text/words/tables from text PDFs
from pdfminer.high_level import extract_text as pdf_extract_text  # detect scanned
# If your import was actually: from pdfminer.high_level import extract_text
# then keep that one instead and remove this alias line.

# ---- Optional scanned PDF OCR (safe to omit in deps) ----
try:
    from pypdfium2 import PdfDocument
    _HAVE_PDFIUM = True
except Exception:
    _HAVE_PDFIUM = False

try:
    from PIL import Image
    import pytesseract
    from pytesseract import Output
    _HAVE_TESS = True
except Exception:
    _HAVE_TESS = False

# ---- Optional table extractor (text PDFs) ----
try:
    import camelot
    _HAVE_CAMELOT = True
except Exception:
    _HAVE_CAMELOT = False


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="PDF + Excel + Image Extraction API (JSON)")

# -----------------------
# In-memory store for results (dev/demo)
# -----------------------
STORED_RESULTS: Dict[str, Any] = {}  # job_id -> payload

# -----------------------
# JSON-safe helpers
# -----------------------
def df_to_json_records_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return JSON-safe records: NaN/±Inf -> None (=> JSON null)."""
    if df is None or len(df) == 0:
        return []
    # Replace ±Inf with NA, then convert NA to None
    df2 = (
        df.replace([np.inf, -np.inf], pd.NA)
          .where(pd.notnull(df), None)         # turn NA/NaN into None
    )
    # Ensure plain Python scalars (avoid numpy types sneaking in)
    return df2.astype(object).to_dict(orient="records")

def is_scanned_pdf(filepath: str) -> bool:
    """Heuristic: if pdfminer finds no text, treat as scanned."""
    try:
        txt = pdf_extract_text(filepath) or ""
        return len(txt.strip()) == 0
    except Exception:
        return True

def render_pdf_to_images(filepath: str, dpi: int = 300) -> List["Image.Image"]:
    """Render pages to PIL using pypdfium2 (if available)."""
    if not _HAVE_PDFIUM:
        raise RuntimeError("pypdfium2 not installed")
    imgs: List["Image.Image"] = []
    scale = dpi / 72.0
    pdf = PdfDocument(filepath)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            pil_img = (
                page.render(scale=scale, draw_annots=True, may_draw_forms=True)
                    .to_pil()
                    .convert("RGB")
            )
            imgs.append(pil_img)
            page.close()
    finally:
        pdf.close()
    return imgs

def ocr_page_to_df(pil_image: "Image.Image") -> Tuple[str, pd.DataFrame]:
    """OCR one PIL image to raw text + word boxes (pytesseract)."""
    if not _HAVE_TESS:
        raise RuntimeError("pytesseract/Pillow not installed or Tesseract binary unavailable")
    df = pytesseract.image_to_data(pil_image, lang="eng", output_type=Output.DATAFRAME)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    # Build line text
    if {"block_num", "par_num", "line_num"}.issubset(df.columns):
        grouped = df.groupby(["block_num", "par_num", "line_num"])["text"]
        raw_text = "\n".join(
            grouped.apply(lambda s: " ".join(str(x) for x in s if str(x).strip())).tolist()
        )
    else:
        raw_text = " ".join(str(x) for x in df["text"].tolist())
    return raw_text, df

# -----------------------
# PDF extractors
# -----------------------
def extract_textpdf(filepath: str):
    """Text PDF: pdfplumber for text/words/tables; optional Camelot tables."""
    raw_text_pages: List[str] = []
    words_sheets: List[Tuple[str, pd.DataFrame]] = []
    table_sheets: List[Tuple[str, pd.DataFrame]] = []

    with pdfplumber.open(filepath) as pdf:
        for p_idx, page in enumerate(pdf.pages, start=1):
            words = page.extract_words() or []
            words_df = pd.DataFrame(words)
            words_sheets.append((f"WORDS_Page{p_idx}", words_df))

            raw_text_pages.append(page.extract_text() or "")

            for t_idx, table in enumerate(page.extract_tables() or [], start=1):
                df = pd.DataFrame(table)
                table_sheets.append((f"TABLE_Page{p_idx}_T{t_idx}", df))

    if _HAVE_CAMELOT:
        try:
            cam_tables = camelot.read_pdf(filepath, pages="all")
            for i, t in enumerate(cam_tables, start=1):
                table_sheets.append((f"TABLE_Camelot_T{i}", t.df))
        except Exception:
            pass

    return raw_text_pages, words_sheets, table_sheets

def extract_scanned(filepath: str, dpi: int = 300):
    """Scanned PDF: render with pypdfium2, OCR with pytesseract."""
    raw_text_pages: List[str] = []
    words_sheets: List[Tuple[str, pd.DataFrame]] = []
    table_sheets: List[Tuple[str, pd.DataFrame]] = []

    images = render_pdf_to_images(filepath, dpi=dpi)
    for p_idx, img in enumerate(images, start=1):
        raw_txt, words_df = ocr_page_to_df(img)
        raw_text_pages.append(raw_txt)
        words_sheets.append((f"WORDS_Page{p_idx}", words_df))

    return raw_text_pages, words_sheets, table_sheets

def extract_all_pdf(filepath: str):
    return extract_scanned(filepath) if is_scanned_pdf(filepath) else extract_textpdf(filepath)

# -----------------------
# Optional LLM normalize (PDF only)
# -----------------------
def try_llm_normalize(raw_text_pages: List[str], table_sheets: List[Tuple[str, pd.DataFrame]]) -> Dict[str, Any]:
    """
    Call OpenAI to normalize to {fields, line_items, tables}.
    Requires `openai` and OPENAI_API_KEY. Returns {"error": "..."} on failure.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        return {"error": f"openai package not installed: {e}"}

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)

    # keep input compact
    sample_tables = []
    for name, df in (table_sheets[:3] if table_sheets else []):
        sample_tables.append({
            "name": name,
            "columns": list(df.columns) if df is not None else [],
            "rows": (df.astype(str).values.tolist()[:10] if df is not None else []),
        })

    system = (
        "You are an invoice parser. Extract a normalized JSON with keys:\n"
        "fields (object of key invoice-level fields),\n"
        "line_items (array of {description, quantity, unit_price, line_total}),\n"
        "tables (array of {name, columns[], rows[][]}).\n"
        "Return STRICT JSON."
    )
    user = (
        "RAW_TEXT_PAGES (first 5):\n"
        + "\n\n---\n\n".join(raw_text_pages[:5])
        + "\n\nTABLE_SAMPLES:\n"
        + pd.Series(sample_tables).to_json(orient="values")
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        import json
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": f"LLM normalize failed: {e}"}

# -----------------------
# API
# -----------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "POST a file to /process (field: file). Supports PDF, Images (JPEG/PNG/TIFF/BMP/WEBP), and Excel. Optional: use_llm=true (PDF only).",
    }

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    use_llm: Optional[bool] = Form(False),
):
    # Read upload
    content = await file.read()
    filename = file.filename or "upload"
    ctype = (file.content_type or "").lower()

    # ---------- Images (JPEG/PNG/TIFF/BMP/WEBP) ----------
    if (
        ctype.startswith("image/")
        or filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"))
    ):
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")  # PIL accepts file-like objects
            raw_txt, words_df = ocr_page_to_df(img)
        except Exception as e:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"Image OCR failed: {e}"})

        words_json = [{"name": "WORDS_Image", "rows": df_to_json_records_safe(words_df)}]
        payload: Dict[str, Any] = {
            "ok": True,
            "type": "image",
            "source": {"filename": filename, "content_type": ctype or "image/*"},
            "is_scanned_pdf": False,
            "raw_text_pages": [raw_txt],  # single-page analogue
            "words": words_json,
            "tables": [],                  # none by default for images
        }
        job_id = uuid4().hex
        STORED_RESULTS[job_id] = payload
        return {"job_id": job_id, **payload}

    # ---------- Excel ----------
    if (
        "excel" in ctype
        or filename.lower().endswith((".xlsx", ".xlsm", ".xltx", ".xltm", ".xls", ".xlsb", ".ods", ".odf", ".odt"))
    ):
        try:
            # engine auto-detect; for .xls may need `xlrd`, for .xlsb `pyxlsb`
            excel = pd.read_excel(io.BytesIO(content), sheet_name=None)
        except Exception as e:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"Excel read failed: {e}"})

        sheets = []
        for sheet_name, df in excel.items():
            sheets.append({
                "name": sheet_name,
                "columns": list(df.columns),
                "rows": df_to_json_records_safe(df),
            })

        payload: Dict[str, Any] = {
            "ok": True,
            "type": "excel",
            "source": {"filename": filename, "content_type": ctype or "application/vnd.ms-excel"},
            "sheets": sheets,
        }

        # store + return with job_id
        job_id = uuid4().hex
        STORED_RESULTS[job_id] = payload
        return {"job_id": job_id, **payload}

    # ---------- PDF ----------
    if "pdf" in ctype or filename.lower().endswith(".pdf"):
        tmp_dir = BASE_DIR / ".tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_pdf_path = tmp_dir / filename
        with open(tmp_pdf_path, "wb") as f:
            f.write(content)

        scanned = is_scanned_pdf(str(tmp_pdf_path))
        try:
            raw_text_pages, words_sheets, table_sheets = extract_all_pdf(str(tmp_pdf_path))
        except Exception as e:
            detail = ""
            if scanned and (not _HAVE_PDFIUM or not _HAVE_TESS):
                detail = " (scanned PDFs require pypdfium2 + pytesseract + Tesseract binary)"
            return JSONResponse(status_code=400, content={"ok": False, "error": f"PDF extraction failed: {e}{detail}"})

        words_json = [{"name": name, "rows": df_to_json_records_safe(df)} for name, df in words_sheets]
        tables_json = [{
            "name": name,
            "columns": (list(df.columns) if df is not None else []),
            "rows": df_to_json_records_safe(df)
        } for name, df in table_sheets]

        payload: Dict[str, Any] = {
            "ok": True,
            "type": "pdf",
            "source": {"filename": filename, "content_type": ctype or "application/pdf"},
            "is_scanned_pdf": bool(scanned),
            "raw_text_pages": raw_text_pages,
            "words": words_json,
            "tables": tables_json,
        }
        if use_llm:
            payload["llm"] = try_llm_normalize(raw_text_pages, table_sheets)

        # store + return with job_id
        job_id = uuid4().hex
        STORED_RESULTS[job_id] = payload
        return {"job_id": job_id, **payload}

    # ---------- Unsupported ----------
    return JSONResponse(
        status_code=415,
        content={"ok": False, "error": f"Unsupported content_type '{ctype}'. Please upload an Image, PDF, or Excel file."},
    )

# -----------------------
# Retrieval endpoint
# -----------------------
@app.get("/results/{job_id}")
def get_results(job_id: str):
    payload = STORED_RESULTS.get(job_id)
    if payload is None:
        return JSONResponse(status_code=404, content={"ok": False, "error": "Job ID not found"})
    return payload
