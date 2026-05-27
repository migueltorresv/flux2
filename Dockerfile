FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . fastapi uvicorn[standard] \
    --extra-index-url https://download.pytorch.org/whl/cu129

# Copy source code
COPY src/ src/
COPY api.py .

# Copy model weights (downloaded by Cloud Build step before docker build)
COPY models/ /models/

# Point to local model files
ENV KLEIN_4B_MODEL_PATH=/models/flux-2-klein-4b.safetensors
ENV AE_MODEL_PATH=/models/ae.safetensors
ENV QWEN3_4B_PATH=/models/Qwen3-4B
ENV MODEL_NAME=flux.2-klein-4b
ENV PYTHONPATH=src
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
