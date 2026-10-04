# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-04

### Added

- **Optimization Module**
  - TSP solver with Christofides algorithm and nearest-neighbor heuristic
  - VRP solver with Clarke-Wright savings algorithm
  - GPU-accelerated TSP/VRP solvers using PyTorch
  - 2-opt local search improvement
  - GPU fallback to CPU on CUDA failure
  - Edge AI inference engine backed by ONNX Runtime
  - Crop vision pipeline: disease detection, weed classification, yield prediction

- **IoT Module**
  - MQTT client for sensor data ingestion
  - Kafka stream processing
  - TimescaleDB storage for time-series data
  - FIWARE NGSI-LD context broker integration
  - IoT data validation: schema validation, range checks, anomaly detection
  - IoT device management: registration, provisioning, health monitoring
  - IoT sensor placement optimization (greedy Set Cover)
  - FIWARE smart data models for agriculture

- **Multi-Agent Module**
  - Swarm coordination for agricultural robots
  - Byzantine fault tolerance consensus
  - Task allocation using greedy and Hungarian algorithms
  - Multi-agent collision avoidance
  - Fault tolerance: task reassignment and leader election

- **Digital Twin Module**
  - Digital twin simulation engine
  - Agricultural knowledge graph with entity management
  - AGROVOC ontology integration
  - Nutrient cycling model
  - Water balance and irrigation model
  - Graph algorithms for agricultural network analysis

- **Decision Support Module**
  - Decision engine for agricultural recommendations
  - Alert management system
  - Zero-trust authentication (JWT, RBAC, rate limiting)
  - Unified REST API Gateway (FastAPI)
  - Real-time farm monitoring dashboard with WebSocket
  - GPS anti-spoofing detection
  - ML model interfaces and yield prediction
  - Notification channels (email, SMS, webhook)
  - Audit logging for compliance

- **Integration Module**
  - Cross-module event bus with pub/sub
  - Shared FarmState model with validation
  - Unified optimizer interface wrapping TSP/VRP/GPU solvers

- **Path Planning Module**
  - Coverage path planning (boustrophedon pattern)
  - Obstacle field for path planning

- **Onboarding Module**
  - Module registry mapping tiers to onboarding modules
  - Organization profiling and tier recommendation

- **Infrastructure**
  - Docker and Docker Compose support
  - Kubernetes manifests (deployment, service, configmap, HPA, ingress)
  - CI/CD pipeline (GitHub Actions)
  - 632+ tests across 33 modules
  - Property-based testing with Hypothesis
  - GPU acceleration support with CPU fallback

### Security

- Zero-trust authentication with JWT tokens
- Role-based access control (RBAC)
- Rate limiting on API endpoints
- GPS anti-spoofing detection
- Audit logging for compliance
- Password policy enforcement
- TOTP two-factor authentication support
