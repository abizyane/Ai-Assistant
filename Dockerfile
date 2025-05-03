FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./rag/requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

CMD ["python", "/rag/rag.py"]