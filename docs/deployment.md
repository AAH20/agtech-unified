# Deployment Guide

## Table of Contents

- [Quick Start](#quick-start)
- [Docker](#docker)
- [Docker Compose](#docker-compose)
- [Kubernetes](#kubernetes)
- [Bare-Metal](#bare-metal)
- [Edge Deployment (Jetson / Raspberry Pi)](#edge-deployment-jetson--raspberry-pi)
- [Environment Variables](#environment-variables)
- [Health Checks](#health-checks)
- [Resource Requirements](#resource-requirements)
- [Scaling](#scaling)
- [Monitoring & Observability](#monitoring--observability)
- [SSL/TLS Configuration](#tls-configuration)
- [Rollback Procedures](#rollback-procedures)

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/AAH20/agtech-unified.git
cd agtech-unified
pip install -e .

# Run tests
pytest

# Start API server
uvicorn src.decision_support.api_gateway:create_app --factory --host 0.0.0.0 --port 8000
```

---

## Docker

```bash
docker build -t agtech-unified .
docker run -p 8000:8000 agtech-unified
```

With custom environment:

```bash
docker run -p 8000:8000 \
  -e JWT_SECRET=change-me \
  -e DATABASE_URL=postgresql://user:pass@db:5432/agtech \
  -e AGTECH_ENV=production \
  agtech-unified
```

---

## Docker Compose

```bash
docker-compose up -d
```

The `docker-compose.yml` includes:
- `agtech-unified` app container
- PostgreSQL database
- MQTT broker (Eclipse Mosquitto)
- Kafka + Zookeeper
- TimescaleDB for time-series data

---

## Kubernetes

### Prerequisites

- Kubernetes cluster (v1.25+)
- `kubectl` configured
- Helm (optional, for ingress controller)

### Deploy

```bash
# Apply all manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Verify
kubectl get pods -l app=agtech-unified
kubectl get svc agtech-unified
```

### Port Forwarding (local testing)

```bash
kubectl port-forward svc/agtech-unified 8000:8000
```

---

## Bare-Metal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn src.decision_support.api_gateway:create_app --factory --host 0.0.0.0 --port 8000
```

---

## Edge Deployment (Jetson / Raspberry Pi)

### NVIDIA Jetson (Orin / Xavier)

```bash
# Use the CUDA-enabled Docker image
docker build -f Dockerfile.jetson -t agtech-unified:jetson .
docker run --runtime nvidia -p 8000:8000 agtech-unified:jetson
```

### Raspberry Pi (ARM64)

```bash
# Build for ARM64
docker buildx build --platform linux/arm64 -t agtech-unified:arm64 .
docker run -p 8000:8000 agtech-unified:arm64
```

### Edge Configuration

Set these environment variables for edge deployments:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGTECH_ENV` | `production` | Environment name |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MQTT_BROKER_URL` | `localhost` | MQTT broker address |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `TIMESCALEDB_URL` | `localhost:5432` | TimescaleDB host |
| `GPU_ENABLED` | `false` | Enable GPU acceleration |
| `BATCH_SIZE` | `32` | Inference batch size |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET` | Yes | — | JWT signing secret (min 32 chars) |
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `MQTT_BROKER_URL` | No | `localhost` | MQTT broker hostname |
| `KAFKA_BOOTSTRAP_SERVERS` | No | `localhost:9092` | Kafka bootstrap servers |
| `TIMESCALEDB_URL` | No | `localhost:5432` | TimescaleDB hostname |
| `API_PORT` | No | `8000` | API server port |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `AGTECH_ENV` | No | `development` | Environment (development/staging/production) |
| `GPU_ENABLED` | No | `false` | Enable GPU acceleration |
| `BATCH_SIZE` | No | `32` | Inference batch size |
| `MAX_WORKERS` | No | `4` | Thread pool workers |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated) |
| `RATE_LIMIT_PER_MINUTE` | No | `60` | API rate limit per minute |

See `.env.example` for a template.

---

## Health Checks

The API gateway exposes health endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe |
| `GET /health/ready` | Readiness probe (checks DB, MQTT, Kafka) |
| `GET /metrics` | Prometheus metrics |

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 15

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

---

## Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| API Gateway | 0.5–2 cores | 512 MB–2 GB | 1 GB |
| PostgreSQL | 0.5–1 core | 512 MB–1 GB | 10 GB |
| TimescaleDB | 0.5–2 cores | 1–4 GB | 50 GB |
| Kafka | 0.5–1 core | 1–2 GB | 20 GB |
| MQTT Broker | 0.1–0.5 core | 128–512 MB | 1 GB |
| GPU Worker (optional) | 2–8 cores | 4–16 GB | 5 GB |

---

## Scaling

### Horizontal Pod Autoscaler (HPA)

The `k8s/hpa.yaml` manifest configures autoscaling:

- **Min replicas**: 2
- **Max replicas**: 10
- **Target CPU utilization**: 70%
- **Target memory utilization**: 80%

### Manual Scaling

```bash
kubectl scale deployment agtech-unified --replicas=5
```

### Database Scaling

- Use read replicas for PostgreSQL
- Partition TimescaleDB hypertables by time
- Use Kafka consumer groups for parallel processing

---

## Monitoring & Observability

### Prometheus Metrics

The `/metrics` endpoint exposes:

- `agtech_requests_total` — Total HTTP requests
- `agtech_request_duration_seconds` — Request latency histogram
- `agtech_active_connections` — Active WebSocket connections
- `agtech_iot_messages_total` — IoT messages processed
- `agtech_optimization_duration_seconds` — Optimization solve time

### Grafana Dashboard

Import the provided dashboard JSON (see `monitoring/grafana-dashboard.json`).

### Logging

Structured JSON logging is enabled in production:

```json
{"timestamp": "2026-10-04T12:00:00Z", "level": "INFO", "module": "api_gateway", "message": "Request completed", "duration_ms": 42}
```

### Distributed Tracing

OpenTelemetry tracing is supported via `OTEL_EXPORTER_OTLP_ENDPOINT`.

---

## TLS Configuration

### Using cert-manager (recommended)

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: agtech-unified-tls
  namespace: default
spec:
  secretName: agtech-unified-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - agtech.example.com
```

### Manual TLS

```bash
# Generate self-signed cert (dev only)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Mount into container
docker run -p 443:443 -v $(pwd)/cert.pem:/cert.pem -v $(pwd)/key.pem:/key.pem agtech-unified
```

### Ingress TLS

See `k8s/ingress.yaml` for TLS termination at the ingress controller.

---

## Rollback Procedures

### Kubernetes Rollback

```bash
# Check rollout history
kubectl rollout history deployment/agtech-unified

# Rollback to previous revision
kubectl rollout undo deployment/agtech-unified

# Rollback to specific revision
kubectl rollout undo deployment/agtech-unified --to-revision=3
```

### Docker Rollback

```bash
# List available images
docker images agtech-unified

# Run previous version
docker run -p 8000:8000 agtech-unified:previous-tag
```

### Database Rollback

```bash
# Restore from backup
pg_dump agtech_unified > backup_$(date +%Y%m%d).sql
psql agtech_unified < backup_20261003.sql
```

### Blue-Green Deployment

```bash
# Deploy new version alongside old
kubectl apply -f k8s/deployment-green.yaml

# Switch service to green
kubectl patch service agtech-unified -p '{"spec":{"selector":{"version":"green"}}}'

# Rollback: switch back to blue
kubectl patch service agtech-unified -p '{"spec":{"selector":{"version":"blue"}}}'
```
