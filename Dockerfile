FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages kept minimal for faster builds.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for better layer caching.
COPY requirements.txt ./

# requirements.txt includes a non-pip line: "python == 3.13.5".
# Filter it out during install so image builds reliably.
RUN grep -viE '^\s*python\s*==' requirements.txt > requirements.docker.txt \
    && pip install --upgrade pip \
    && pip install -r requirements.docker.txt

# Copy project files.
COPY . .

ENV PORT=7860

EXPOSE 7860

# Default: FastAPI API server (Hugging Face Spaces uses port 7860)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]