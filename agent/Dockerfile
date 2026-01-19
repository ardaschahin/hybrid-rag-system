FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agent /app/agent
COPY chroma_db /app/chroma_db

EXPOSE 8001
CMD ["uvicorn", "agent.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
