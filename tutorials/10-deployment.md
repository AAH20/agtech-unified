# Tutorial 10: Deployment

## Overview

Learn how to deploy agtech-unified to production.

## Docker Deployment

```bash
# Build image
docker build -t agtech-unified:1.0.0 .

# Run with environment variables
docker run -d \
  --name agtech-unified \
  -p 8000:8000 \
  -e JWT_SECRET=your-secret-key \
  -e DATABASE_URL=postgresql://user:pass@db:5432/agtech \
  -e AGTECH_ENV=production \
  agtech-unified:1.0.0
```

## Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Verify
kubectl get pods -l app=agtech-unified
kubectl get svc agtech-unified
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Fill in your secrets
3. Never commit `.env` to version control

## Monitoring

- Health checks: `GET /health`, `GET /health/ready`
- Metrics: `GET /metrics`
- Logs: Structured JSON logging

## Next Steps

- [Deployment Guide](../docs/deployment.md)
- [API Reference](../README.md)
