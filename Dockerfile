FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev
COPY src ./src
COPY corpus ./corpus
COPY app.py README.md ./
ENV PYTHONPATH=/app/src
EXPOSE 8501
CMD [".venv/bin/streamlit", "run", "app.py", "--server.address=0.0.0.0"]
