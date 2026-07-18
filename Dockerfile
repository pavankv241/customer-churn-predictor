FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/
COPY web/ web/
COPY models/ models/
COPY reports/ reports/
COPY data/ data/
COPY app.py train_model.py pytest.ini ./

ENV PYTHONUNBUFFERED=1
ENV API_URL=http://api:8000

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
