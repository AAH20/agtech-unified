# Tutorial 07: API Gateway

## Overview

Learn how to use the REST API gateway to expose farm data and control actuators.

## Starting the Server

```bash
uvicorn src.decision_support.api_gateway:create_app --factory --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Create Farm

```bash
curl -X POST http://localhost:8000/farms \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"name": "Green Valley Farm", "location": {"lat": 40.7, "lon": -74.0}}'
```

### Submit Sensor Reading

```bash
curl -X POST http://localhost:8000/sensors/readings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"sensor_id": "soil-01", "temperature": 25.3, "soil_moisture": 0.65}'
```

### Get Recommendations

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"soil_moisture": 0.15, "temperature": 40.0}'
```

## Authentication

```python
from src.decision_support.security import ZeroTrustAuth

auth = ZeroTrustAuth(secret_key="your-secret-key")
token = auth.generate_token(user_id="farmer-1", roles=["admin"])
print(f"Token: {token}")
```

## Next Steps

- [Tutorial 08: Genomics](08-genomics.md)
- [Tutorial 09: End-to-End Farm](09-end-to-end-farm.md)
