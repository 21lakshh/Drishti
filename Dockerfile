FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set HF cache dir BEFORE any model downloads happen
ENV HF_HOME=/app/models
ENV MODEL_SERVER_URL=http://127.0.0.1:8000
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Install Sarvam plugin from git
RUN uv pip install --system --no-deps \
    "livekit-plugins-sarvam @ git+https://github.com/livekit/agents.git#subdirectory=livekit-plugins/livekit-plugins-sarvam"

# Copy ALL files first
COPY . .

# Pre-download models so they are baked into the Docker image
RUN mkdir -p /app/models && python -c "\
from ultralytics import YOLO; YOLO('yolo11n.pt');\n\
from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2');\n\
from transformers import ZoeDepthForDepthEstimation; ZoeDepthForDepthEstimation.from_pretrained('Intel/zoedepth-nyu-kitti')"

# Make startup script executable
RUN chmod +x start.sh

EXPOSE 4000
EXPOSE 8000

CMD ["./start.sh"]
