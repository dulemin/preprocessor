# --- Runtime image ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System-Tools nur minimal (für Healthcheck & Debug)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code
COPY app ./app

# FastAPI / Uvicorn
EXPOSE 8000

# Optionaler Healthcheck (n8n wartet so auf "ready")
HEALTHCHECK --interval=20s --timeout=3s --retries=5 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

# Ein Prozess, sauber per exec gestartet
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
