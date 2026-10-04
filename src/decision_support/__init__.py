"""Decision support module: recommendations, alerts, API gateway, security, and dashboards."""

from src.decision_support.alerts import Alert, AlertManager, AlertSeverity, Threshold
from src.decision_support.alerts import NotificationChannel as AlertNotificationChannel
from src.decision_support.api_gateway import APIGateway, TenantRegistry
from src.decision_support.audit import AuditEvent, AuditEventType, AuditLogger
from src.decision_support.dashboard import DashboardConfig, FarmDashboard
from src.decision_support.gps_antispoof import GPSReading, GPSSpoofingDetector
from src.decision_support.ml_models import (
    FeatureVector,
    LinearRegressionModel,
    PredictionResult,
    SimpleMLModel,
    YieldPredictor,
)
from src.decision_support.notifications import (
    EmailNotification,
    NotificationChannel,
    NotificationManager,
    NotificationResult,
    NotificationStatus,
    SMSNotification,
    WebhookNotification,
)
from src.decision_support.recommender import DecisionEngine, FarmState, RecommendationResult
from src.decision_support.security import AuthenticationError, AuthorizationError, ZeroTrustAuth

__all__ = [
    "FarmState",
    "RecommendationResult",
    "DecisionEngine",
    "AlertSeverity",
    "Alert",
    "Threshold",
    "AlertManager",
    "AlertNotificationChannel",
    "AuthenticationError",
    "AuthorizationError",
    "ZeroTrustAuth",
    "APIGateway",
    "TenantRegistry",
    "AuditEventType",
    "AuditEvent",
    "AuditLogger",
    "DashboardConfig",
    "FarmDashboard",
    "GPSReading",
    "GPSSpoofingDetector",
    "FeatureVector",
    "PredictionResult",
    "SimpleMLModel",
    "LinearRegressionModel",
    "YieldPredictor",
    "NotificationStatus",
    "NotificationResult",
    "NotificationChannel",
    "EmailNotification",
    "SMSNotification",
    "WebhookNotification",
    "NotificationManager",
]
