FROM python:3.10-slim


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
  && pip install --no-cache-dir pymupdf


COPY ingest /app/ingest

# data and chroma_db will be mounted as volumes
WORKDIR /app/ingest

CMD ["python", "ingest.py"]
