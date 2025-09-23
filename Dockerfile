# syntax=docker/dockerfile:1
FROM python:3.11-slim

# --- Environment hygiene ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# --- System packages ---
# - tesseract-ocr + tesseract-ocr-eng: OCR engine + English model
# - ghostscript: optional backend for camelot-py table extraction
# - curl: used by HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng ghostscript curl \
  && rm -rf /var/lib/apt/lists/*

# --- App setup ---
WORKDIR /app
COPY requirements.txt ./
# Gunicorn isn’t in your requirements; add it here during build
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# --- (Optional) non-root user for safety ---
RUN useradd -m appuser
USER appuser

# --- Healthcheck hits your root endpoint ---
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8000}/ || exit 1

# --- Expose for local runs (Render will pass $PORT) ---
EXPOSE 8000

# --- Start server, binding to Render's $PORT (fallback 8000 locally) ---
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:${PORT:-8000} app:app"]
