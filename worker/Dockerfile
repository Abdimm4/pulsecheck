# Image for both the API and the worker (docker-compose.yml picks the command).
FROM python:3.12-slim

# opencv-contrib-python (required by mediapipe) needs the system GL libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 pulsecheck
WORKDIR /app
COPY backend/ backend/
COPY configs/ configs/
COPY scripts/download_models.py scripts/
COPY worker/ worker/
RUN pip install --no-cache-dir -e "backend[api,postgres]" \
    && python scripts/download_models.py \
    && mkdir -p data/storage && chown -R pulsecheck data

USER pulsecheck
CMD ["python", "worker/entrypoint.py", "worker"]
