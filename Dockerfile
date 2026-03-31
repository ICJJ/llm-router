FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY llm_router/ llm_router/
RUN pip install --no-cache-dir .

COPY config.example.yaml ./config.yaml

RUN useradd -r -s /bin/false appuser
USER appuser

EXPOSE 8100
CMD ["python", "-m", "llm_router", "serve", "--config", "config.yaml"]
