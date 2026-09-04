FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY src ./src
COPY scripts ./scripts
COPY corpus ./corpus
COPY config ./config
COPY eval ./eval
COPY app.py README.md ./

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
