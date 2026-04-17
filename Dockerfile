FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN grep -viE '^\s*python\s*==' requirements.txt > requirements.docker.txt \
    && pip install --upgrade pip \
    && pip install -r requirements.docker.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]