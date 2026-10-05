"""Tests for infrastructure gap fixes - Wave 2.

Covers:
- DKR-001: Multi-arch Docker build support
- K8S-008: GPU node support
- K8S-009: Canary deployment
- K8S-003: NetworkPolicy
- K8S-002: PodDisruptionBudget
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text()


# ── DKR-001: Multi-arch Docker build ─────────────────────────────────


class TestDKR001MultiArchDocker:
    """DKR-001: Dockerfile must support multi-arch builds."""

    def test_dockerfile_has_targetarch_arg(self):
        """Dockerfile must define ARG TARGETARCH."""
        content = _read("Dockerfile")
        assert "ARG TARGETARCH" in content, "Dockerfile must define ARG TARGETARCH"

    def test_dockerfile_has_targetplatform_arg(self):
        """Dockerfile must define ARG TARGETPLATFORM."""
        content = _read("Dockerfile")
        assert "ARG TARGETPLATFORM" in content, "Dockerfile must define ARG TARGETPLATFORM"

    def test_dockerfile_uses_buildplatform_in_from(self):
        """Builder stage must use --platform=$BUILDPLATFORM."""
        content = _read("Dockerfile")
        assert (
            "--platform=$BUILDPLATFORM" in content
        ), "Builder stage must use --platform=$BUILDPLATFORM for multi-arch builds"

    def test_dockerfile_uses_targetarch_for_platform_specific_installs(self):
        """Dockerfile must use TARGETARCH for platform-specific logic."""
        content = _read("Dockerfile")
        assert "TARGETARCH" in content, "Dockerfile must use TARGETARCH for platform-specific logic"


# ── K8S-008: GPU node support ────────────────────────────────────────


class TestK8S008GPUNodeSupport:
    """K8S-008: k8s/gpu.yaml must configure GPU node scheduling."""

    def test_gpu_yaml_exists(self):
        """k8s/gpu.yaml must exist."""
        assert (PROJECT_ROOT / "k8s" / "gpu.yaml").is_file()

    def test_gpu_yaml_has_deployment(self):
        """gpu.yaml must define a Deployment."""
        content = _read("k8s/gpu.yaml")
        assert "kind: Deployment" in content

    def test_gpu_yaml_has_node_selector(self):
        """gpu.yaml must have nodeSelector targeting GPU nodes."""
        content = _read("k8s/gpu.yaml")
        assert "nodeSelector" in content
        assert "gpu" in content.lower()

    def test_gpu_yaml_has_gpu_resource_limit(self):
        """gpu.yaml must request nvidia.com/gpu resource."""
        content = _read("k8s/gpu.yaml")
        assert "nvidia.com/gpu" in content

    def test_gpu_yaml_has_tolerations(self):
        """gpu.yaml must tolerate GPU node taints."""
        content = _read("k8s/gpu.yaml")
        assert "tolerations" in content

    def test_gpu_yaml_has_gpu_taint_toleration(self):
        """gpu.yaml must tolerate nvidia.com/gpu taint."""
        content = _read("k8s/gpu.yaml")
        assert "nvidia.com/gpu" in content
        assert "NoSchedule" in content or "NoExecute" in content


# ── K8S-009: Canary deployment ────────────────────────────────────────


class TestK8S009CanaryDeployment:
    """K8S-009: k8s/canary.yaml must define canary deployment strategy."""

    def test_canary_yaml_exists(self):
        """k8s/canary.yaml must exist."""
        assert (PROJECT_ROOT / "k8s" / "canary.yaml").is_file()

    def test_canary_yaml_has_canary_resource(self):
        """canary.yaml must define a Canary resource."""
        content = _read("k8s/canary.yaml")
        assert "Canary" in content or "canary" in content

    def test_canary_yaml_has_weight_based_traffic(self):
        """canary.yaml must support gradual traffic shifting."""
        content = _read("k8s/canary.yaml")
        assert "weight" in content.lower() or "steps" in content.lower()

    def test_canary_yaml_has_analysis(self):
        """canary.yaml must have automated analysis/rollback."""
        content = _read("k8s/canary.yaml")
        assert "analysis" in content.lower() or "threshold" in content.lower()


# ── K8S-003: NetworkPolicy ───────────────────────────────────────────


class TestK8S003NetworkPolicy:
    """K8S-003: k8s/network-policy.yaml must restrict pod network traffic."""

    def test_network_policy_yaml_exists(self):
        """k8s/network-policy.yaml must exist."""
        assert (PROJECT_ROOT / "k8s" / "network-policy.yaml").is_file()

    def test_network_policy_has_correct_kind(self):
        """network-policy.yaml must define a NetworkPolicy."""
        content = _read("k8s/network-policy.yaml")
        assert "kind: NetworkPolicy" in content

    def test_network_policy_selects_agtech_pods(self):
        """NetworkPolicy must select agtech-unified pods."""
        content = _read("k8s/network-policy.yaml")
        assert "app: agtech-unified" in content or "app=agtech-unified" in content

    def test_network_policy_has_ingress_rules(self):
        """NetworkPolicy must define ingress rules."""
        content = _read("k8s/network-policy.yaml")
        assert "ingress" in content

    def test_network_policy_has_egress_rules(self):
        """NetworkPolicy must define egress rules."""
        content = _read("k8s/network-policy.yaml")
        assert "egress" in content

    def test_network_policy_restricts_to_known_services(self):
        """NetworkPolicy must restrict traffic to known services."""
        content = _read("k8s/network-policy.yaml")
        assert "8000" in content or "port" in content


# ── K8S-002: PodDisruptionBudget ─────────────────────────────────────


class TestK8S002PodDisruptionBudget:
    """K8S-002: k8s/pdb.yaml must ensure minimum availability."""

    def test_pdb_yaml_exists(self):
        """k8s/pdb.yaml must exist."""
        assert (PROJECT_ROOT / "k8s" / "pdb.yaml").is_file()

    def test_pdb_has_correct_kind(self):
        """pdb.yaml must define a PodDisruptionBudget."""
        content = _read("k8s/pdb.yaml")
        assert "kind: PodDisruptionBudget" in content

    def test_pdb_selects_agtech_pods(self):
        """PDB must select agtech-unified pods."""
        content = _read("k8s/pdb.yaml")
        assert "app: agtech-unified" in content or "app=agtech-unified" in content

    def test_pdb_has_min_available(self):
        """PDB must define minAvailable."""
        content = _read("k8s/pdb.yaml")
        assert "minAvailable" in content

    def test_pdb_min_available_is_at_least_1(self):
        """PDB minAvailable must be at least 1."""
        content = _read("k8s/pdb.yaml")
        assert (
            "minAvailable: 1" in content
            or 'minAvailable: "1"' in content
            or "minAvailable: 2" in content
        ), "PDB minAvailable must be at least 1"
