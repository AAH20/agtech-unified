# ── Builder stage ─────────────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

# ── Platform args (multi-arch support) ────────────────────────────────
ARG TARGETARCH
ARG TARGETPLATFORM

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # setuptools vendors wheel-0.45.1 (CVE-2026-24049); upgrading
    # setuptools replaces the vendored copy. jaraco.context patched by
    # the direct upgrade below.
    && pip install --no-cache-dir --upgrade "setuptools>=84.0.0" "wheel>=0.46.2" "jaraco.context>=6.1.0"

# ── Runtime stage ─────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Platform args (multi-arch support) ────────────────────────────────
ARG TARGETARCH
ARG TARGETPLATFORM

# ── Patch base-image OS vulnerabilities (util-linux family) ───────────
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

LABEL maintainer="Hassan, Ahmed"
LABEL description="AgTech Unified - Precision Agriculture & Genetic Engineering Platform"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Purge vulnerable copies pip cannot remove:
# 1. setuptools <84 vendored wheel-0.45.1 (CVE-2026-24049) — the dist-info
#    directory can be left behind even after setuptools is upgraded, and
#    Trivy scans it as python-pkg METADATA.
# 2. Stale jaraco.context 5.3.0 dist-info copy (CVE-2026-23949).
# Repair-package versions come in clean via the COPY from builder
# (wheel 0.48.0, jaraco.context 6.1.2, setuptools 84.0.0 verified in the
# build logs of run 37474735186).
RUN for d in /usr/local/lib/python3.11/site-packages/setuptools/_vendor/wheel-0.45.1.dist-info \
             /usr/local/lib/python3.11/site-packages/jaraco_context-5.3.0.dist-info; do \
        rm -rf "$d"; \
    done

# ── Platform-specific runtime dependencies ────────────────────────────
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo "ARM64 detected: $TARGETPLATFORM"; \
    elif [ "$TARGETARCH" = "amd64" ]; then \
        echo "AMD64 detected: $TARGETPLATFORM"; \
    else \
        echo "Unknown arch $TARGETARCH for $TARGETPLATFORM"; \
    fi

# Only copy application source (not tests/, docs/, etc.)
COPY src/ src/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.decision_support.api_gateway:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
