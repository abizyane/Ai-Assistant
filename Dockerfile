FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./rag/ /app/rag

RUN pip install --no-cache-dir -r /app/rag/requirements.txt

ARG GEMINI_API_KEY
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

ENV PYTHONPATH=/app

CMD ["python", "/app/rag/rag.py"]