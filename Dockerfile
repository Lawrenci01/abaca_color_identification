# Use lightweight Python 3.10 image
FROM python:3.10-slim
# Set working directory inside container
WORKDIR /app
# Prevent Python from buffering logs (important for HF Spaces)
ENV PYTHONUNBUFFERED=1
# Hugging Face Spaces expects app on port 7860
ENV PORT=7860
# Install system dependencies (minimal set) and clean cache
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*
# Upgrade pip for latest dependency resolution
RUN pip install --upgrade pip
# Copy requirements first (better Docker caching)
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy main application files
COPY app.py features.py db.py segment.py ./
# Copy model downloader (fetches models from lawrencio/abaca-models at startup)
COPY download_models.py ./
# Copy service worker
COPY sw.js ./
# Copy HTML entry point
COPY templates/index.html ./templates/index.html
# Copy static assets (CSS, JS, HTML partials)
COPY static/ ./static/
# NOTE: abaca_pipeline/ is NOT copied here anymore.
# Models are downloaded from HF at container startup by download_models.py.
# To update the model, push new files to lawrencio/abaca-models/abaca_pipeline/
# and restart the Space — no code changes needed.
# Expose the port required by HF Spaces
EXPOSE 7860
# Download models from HF, then start the app
CMD ["sh", "-c", "python -u download_models.py && python -u app.py"]