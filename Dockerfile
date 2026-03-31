# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Gateway / shared deps
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Legacy MVP app deps (optional — only if present)
COPY legacy_mvp/app/requirements.txt ./legacy_mvp_app_requirements.txt
RUN pip install --prefix=/install --no-cache-dir -r legacy_mvp_app_requirements.txt || true

# Identity API deps (optional — only if present)
COPY legacy_mvp/services/identity_api/requirements.txt ./identity_api_requirements.txt
RUN pip install --prefix=/install --no-cache-dir -r identity_api_requirements.txt || true


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Default entry point — docker-compose overrides CMD per service
CMD ["uvicorn", "apps.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
