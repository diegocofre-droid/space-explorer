FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000
EXPOSE 8000
# 1 worker: los trabajos viven en memoria del proceso
CMD gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:${PORT} app:app
