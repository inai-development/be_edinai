FROM python:3.11-slim

# Basic environment configuration
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

WORKDIR /app

# Copy only requirements first to leverage Docker layer cache
COPY backend/requirements.txt ./requirements.txt

# Install system dependencies: Tesseract, Ghostscript, ffmpeg, etc.
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        libpq-dev \
        libpq5 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        ghostscript \
        ffmpeg \
        curl \
        ca-certificates && \
    mkdir -p /usr/share/tesseract-ocr/5/tessdata/ && \
    curl -L -o /usr/share/tesseract-ocr/5/tessdata/hin.traineddata \
        https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/hin.traineddata && \
    curl -L -o /usr/share/tesseract-ocr/5/tessdata/guj.traineddata \
        https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/guj.traineddata && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "gunicorn>=22" "uvicorn[standard]>=0.30.0" && \
    # Install these without deps to avoid pip ResolutionImpossible conflicts
    # Install these without deps to avoid pip ResolutionImpossible conflicts
    pip install --no-cache-dir --no-deps pytesseract ocrmypdf pydub edge-tts && \
    # Remove only build tools & curl, keep PostgreSQL libs (libpq5) installed
    apt-get purge -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the backend code
COPY backend ./backend

EXPOSE 8000

CMD ["gunicorn","app.main:app","-k", "uvicorn.workers.UvicornWorker","--workers", "2","--timeout", "600","--bind", "0.0.0.0:8000","--access-logfile", "-","--error-logfile", "-"]

