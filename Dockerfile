FROM python:3.12-slim

# Version shown by /health and the API docs; the release workflow passes the git tag.
ARG APP_VERSION=dev

ENV APP_VERSION=${APP_VERSION} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAX_UPLOAD_MB=50

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements.txt .
# invisible-watermark without dependencies: otherwise PyTorch (~1 GB) would be installed,
# although only the dwtDct method is used.
RUN pip install -r requirements.txt \
 && pip install --no-deps invisible-watermark==0.2.0

COPY app ./app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
