# Deployment Guide

## Docker

```bash
docker build -t agtech-unified .
docker run -p 8000:8000 agtech-unified
```

Or with docker-compose:

```bash
docker-compose up -d
```

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl port-forward svc/agtech-unified 8000:8000
```

## Bare-Metal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn decision_support.api_gateway:app --host 0.0.0.0 --port 8000
```

## Environment

Set `AGTECH_ENV` to `development`, `staging`, or `production`.
