FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY rag_chat_app/requirements.txt /rag_chat_app/requirements.txt

RUN pip install --no-cache-dir -r /rag_chat_app/requirements.txt

COPY rag_chat_app /rag_chat_app

WORKDIR /rag_chat_app

# CMD ["python", "/rag_chat_app/rag.py"]
CMD ["bash"]