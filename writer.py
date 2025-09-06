# src/writer.py
import io
import pandas as pd

def _write_frames(writer, raw_text_pages, words_sheets, table_sheets, normalized=None):
    # RAW text
    pd.DataFrame({
        "page": list(range(1, len(raw_text_pages) + 1)),
        "text": raw_text_pages
    }).to_excel(writer, sheet_name="RAW_TEXT", index=False)

    # Words (OCR/words/coords/confidence)
    for name, df in words_sheets:
        if df is None or df.empty:
            pd.DataFrame({"note": ["no data"]}).to_excel(writer, sheet_name=name, index=False)
        else:
            df.to_excel(writer, sheet_name=name, index=False)

    # Tables (as detected)
    for name, df in table_sheets:
        if df is None or df.empty:
            pd.DataFrame({"note": ["no data"]}).to_excel(writer, sheet_name=name, index=False)
        else:
            df.to_excel(writer, sheet_name=name, index=False)

    # Normalized (LLM) — fully defensive
    if isinstance(normalized, dict):
        err = normalized.get("error")

        # Fields
        fields = normalized.get("fields")
        if isinstance(fields, dict) and fields:
            pd.DataFrame([fields]).to_excel(writer, sheet_name="LLM_Fields", index=False)
        elif err:
            pd.DataFrame([{"error": err}]).to_excel(writer, sheet_name="LLM_Error", index=False)

        # Line items
        li = normalized.get("line_items")
        if isinstance(li, list) and li:
            pd.DataFrame(li).to_excel(writer, sheet_name="Line_Items", index=False)

        # Tables
        tables = normalized.get("tables") or []
        for i, t in enumerate(tables, start=1):
            cols = (t.get("columns") or [])
            rows = (t.get("rows") or [])
            if cols and rows:
                df = pd.DataFrame(rows, columns=cols)
                sheet = f'LLM_{str(t.get("name", f"Table{i}"))[:25]}'
                df.to_excel(writer, sheet_name=sheet, index=False)


def write_all(out_xlsx, raw_text_pages, words_sheets, table_sheets, normalized=None):
    with pd.ExcelWriter(out_xlsx) as writer:
        _write_frames(writer, raw_text_pages, words_sheets, table_sheets, normalized)


def build_excel_bytes(raw_text_pages, words_sheets, table_sheets, normalized=None) -> bytes:
    buf = io.BytesIO()
    # Explicit openpyxl engine is fine; pandas will also choose a default if installed.
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _write_frames(writer, raw_text_pages, words_sheets, table_sheets, normalized)
    buf.seek(0)
    return buf.read()
