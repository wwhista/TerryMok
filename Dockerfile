FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TERRYMOK_MODEL_PATH=/models/selected-model.joblib
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir '.[api]' \
    && useradd --create-home --uid 10001 appuser
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "terrymok.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
