FROM python:3.11-slim

WORKDIR /app

# System deps sometimes needed by chromadb / audio tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent data paths (mount a volume at /app/data in production if available)
ENV DOWNLOADS_DIR=/app/downloads \
    VECTOR_DB_DIR=/app/vector_db

CMD ["python", "bot.py"]
