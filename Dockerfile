FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

# Install gcloud CLI to download models from GCS during build
RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . fastapi uvicorn[standard] \
    --extra-index-url https://download.pytorch.org/whl/cu129

# Copy source code
COPY src/ src/
COPY api.py .

# Download model weights from GCS (Cloud Build service account needs read access)
ARG GCS_BUCKET
RUN mkdir -p /models/Qwen3-4B && \
    gsutil cp gs://${GCS_BUCKET}/models/flux-2-klein-4b.safetensors /models/ && \
    gsutil cp gs://${GCS_BUCKET}/models/ae.safetensors /models/ && \
    gsutil -m rsync -r gs://${GCS_BUCKET}/models/Qwen3-4B/ /models/Qwen3-4B/

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
