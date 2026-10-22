# onnxruntime has no musl wheels, so use Debian-based Python rather than the Alpine HA base.
FROM python:3.12-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY baby_detect.py yolov8n.onnx ./

CMD ["python", "-u", "/app/baby_detect.py"]
