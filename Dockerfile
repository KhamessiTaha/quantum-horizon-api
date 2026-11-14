FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QH_CACHE_DIR=/data/cache

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts/entrypoint.sh /srv/entrypoint.sh

# The committed region cache ships with the image and is copied onto the volume on first boot, so the demo targets
# are warm from the first request instead of being fetched live while someone is watching.
COPY cache /srv/seed-cache

RUN useradd --create-home --uid 10001 qh \
    && mkdir -p /data/cache \
    && sed -i 's/\r$//' /srv/entrypoint.sh \
    && chmod +x /srv/entrypoint.sh \
    && chown -R qh /data /srv/seed-cache

EXPOSE 8080
ENTRYPOINT ["/srv/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
