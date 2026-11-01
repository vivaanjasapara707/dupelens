# DupeLens web app for Hugging Face Spaces (Docker SDK) or any container host. CPU only.
# Secrets (set on the host, never baked in): SERPER_API_KEY (live search), GROQ_API_KEY (AI assistant).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf PORT=7860
WORKDIR /app

# CPU build of the pinned torch first (much smaller than the CUDA build), then the rest of the pins.
COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

# Download the three pretrained encoders at build time so the first search is not a 3 GB download.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('clip-ViT-B-32'); S('paraphrase-multilingual-mpnet-base-v2'); \
from transformers import AutoImageProcessor, AutoModel; AutoImageProcessor.from_pretrained('facebook/dinov2-small'); AutoModel.from_pretrained('facebook/dinov2-small')"

COPY . .
RUN mkdir -p data cache/live cache/uploads && chmod -R 777 data cache .hf
EXPOSE 7860
CMD ["python", "-m", "src.app"]
