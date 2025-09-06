# src/llm_normalize.py

import os, json
from typing import List, Tuple
from openai import OpenAI
from utils import load_env

SCHEMA = {
  "name": "extraction_normalized",
  "schema": {
    "type": "object",
    "properties": {
      "fields": {
        "type": "object",
        "additionalProperties": {"type": "string"},
        "description": "Key invoice-level fields (invoice_id, dates, totals, vendor, customer, etc.)."
      },
      "line_items": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "description": {"type": "string"},
            "quantity": {"type": ["string", "number"]},
            "unit_price": {"type": ["string", "number"]},
            "line_total": {"type": ["string", "number"]}
          },
          "required": ["description", "quantity", "unit_price", "line_total"],
          "additionalProperties": False
        }
      },
      "tables": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "rows": {
              "type": "array",
              "items": {"type": "array", "items": {"type": ["string","number","null"]}}
            }
          },
          "required": ["columns", "rows"]
        }
      }
    },
    "required": ["fields", "line_items", "tables"],
    "additionalProperties": False
  }
}


def normalize(raw_text_pages: List[str], table_sheets: List[Tuple[str, any]]):
    api_key, model = load_env()
    client = OpenAI(api_key=api_key)

    prompt = ("""
        You are a smart invoice extractor. From the raw pages and table samples, extract:
        - Key invoice-level fields: invoice_id, invoice_date, due_date, vendor_name, vendor_address, vendor_tax_id, customer_name, customer_address, po_number, subtotal, tax_amount, discount, shipping_charges, payment_terms, payment_mode, invoice_status, notes.
        - Line items: list each item with description, quantity, unit_price, line_total.
        - Any other tables.
        Return valid JSON following the provided schema.
        """
    )

    input_text = (
        "RAW_TEXT_PAGES:\n" + "\n\n---\n\n".join(raw_text_pages[:5]) +
        "\n\nTABLE_SAMPLES:\n" +
        json.dumps([
            {"name": n, "columns": df.columns.tolist(), "rows": df.astype(str).values.tolist()[:10]}
            for n, df in table_sheets[:3]
        ])
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": SCHEMA
        }
    )

    text = resp.choices[0].message.content
    return json.loads(text)
