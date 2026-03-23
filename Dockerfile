FROM python:3.11-slim

# System deps for PyMuPDF, OpenCV, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=api.settings

EXPOSE 8000 8501
