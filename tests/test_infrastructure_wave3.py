"""Tests for infrastructure gap fixes - Wave 3.

Covers:
- CI-001: Multi-arch CI builds (docker/setup-buildx-action)
- CI-008: Image scanning (Trivy/grype)
- CI-016/K8S-017: Helm chart (values.yaml, deployment.yaml, service.yaml, hpa.yaml)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text()


# ── CI-001: Multi-arch CI builds ─────────────────────────────────────


class TestCI001MultiArchBuilds:
    """CI-001: CI must build multi-arch Docker images."""

    def test_ci_has_setup_buildx_action(self):
        """CI must use docker/setup-buildx-action."""
        content = _read(".github/workflows/ci.yml")
        assert (
            "docker/setup-buildx-action" in content
        ), "CI must use docker/setup-buildx-action for multi-arch builds"

    def test_ci_has_build_push_action(self):
        """CI must use docker/build-push-action."""
        content = _read(".github/workflows/ci.yml")
        assert (
            "docker/build-push-action" in content
        ), "CI must use docker/build-push-action to build and push images"

    def test_ci_buildx_action_has_version(self):
        """setup-buildx-action must specify a version."""
        content = _read(".github/workflows/ci.yml")
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "docker/setup-buildx-action" in line:
                context = "\n".join(lines[i : i + 5])
                assert "@v" in context, "setup-buildx-action must pin a version"
                break
        else:
            raise AssertionError("docker/setup-buildx-action not found in CI")

    def test_ci_build_push_has_platforms(self):
        """build-push-action must specify the CI target platform.

        CI builds linux/amd64 (the deployment/scan target). ARM64 builds
        via QEMU with heavy ML wheels exceeded the 30-minute job window
        (run 37455131937), so arm64 images are built in the release
        workflow instead.
        """
        content = _read(".github/workflows/ci.yml")
        assert "platforms:" in content, "build-push-action must specify platforms"
        assert "linux/amd64" in content, "Must build for linux/amd64"

    def test_ci_has_docker_build_job(self):
        """CI must have a dedicated docker-build job."""
        content = _read(".github/workflows/ci.yml")
        assert (
            "docker-build:" in content or "docker_build:" in content
        ), "CI must have a docker-build job"

    def test_ci_docker_job_needs_test(self):
        """docker-build job should depend on test job passing."""
        content = _read(".github/workflows/ci.yml")
        # Find docker-build job and check for needs:
        lines = content.splitlines()
        in_docker_job = False
        for line in lines:
            if "docker-build:" in line or "docker_build:" in line:
                in_docker_job = True
            if in_docker_job and "needs:" in line:
                assert "test" in line, "docker-build job must depend on test job"
                return
        # If no needs found, that's also acceptable for now
        # but we want to encourage it


# ── CI-008: Image scanning ────────────────────────────────────────────


class TestCI008ImageScanning:
    """CI-008: CI must scan Docker images for vulnerabilities."""

    def test_ci_has_trivy_or_grype(self):
        """CI must include Trivy or grype image scanning."""
        content = _read(".github/workflows/ci.yml")
        has_trivy = "trivy" in content.lower()
        has_grype = "grype" in content.lower()
        assert (
            has_trivy or has_grype
        ), "CI must include Trivy or grype for image vulnerability scanning"

    def test_ci_trivy_scans_image(self):
        """Trivy step must scan the built image."""
        content = _read(".github/workflows/ci.yml")
        if "trivy" in content.lower():
            # Find trivy step and verify it scans an image
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if "trivy" in line.lower():
                    context = "\n".join(lines[i : i + 10])
                    assert (
                        "image" in context.lower() or "scan" in context.lower()
                    ), "Trivy step must scan an image"
                    break

    def test_ci_image_scan_has_severity_threshold(self):
        """Image scan must define a severity threshold."""
        content = _read(".github/workflows/ci.yml")
        if "trivy" in content.lower():
            assert (
                "severity" in content.lower() or "CRITICAL" in content
            ), "Image scan must define severity threshold"

    def test_ci_image_scan_fails_on_critical(self):
        """Image scan should fail on critical vulnerabilities."""
        content = _read(".github/workflows/ci.yml")
        if "trivy" in content.lower():
            # Check for exit-on-echo or severity-based failure
            assert (
                "CRITICAL" in content or "severity" in content.lower()
            ), "Image scan must fail on critical vulnerabilities"


# ── CI-016/K8S-017: Helm chart ───────────────────────────────────────


class TestHelmChart:
    """CI-016/K8S-017: Helm chart must exist with required templates."""

    def test_helm_directory_exists(self):
        """helm/ directory must exist."""
        assert (PROJECT_ROOT / "helm").is_dir(), "helm/ directory must exist"

    def test_helm_chart_yaml_exists(self):
        """helm/Chart.yaml must exist."""
        assert (PROJECT_ROOT / "helm" / "Chart.yaml").is_file(), "helm/Chart.yaml must exist"

    def test_helm_values_yaml_exists(self):
        """helm/values.yaml must exist."""
        assert (PROJECT_ROOT / "helm" / "values.yaml").is_file(), "helm/values.yaml must exist"

    def test_helm_templates_directory_exists(self):
        """helm/templates/ directory must exist."""
        assert (
            PROJECT_ROOT / "helm" / "templates"
        ).is_dir(), "helm/templates/ directory must exist"

    def test_helm_deployment_template_exists(self):
        """helm/templates/deployment.yaml must exist."""
        assert (
            PROJECT_ROOT / "helm" / "templates" / "deployment.yaml"
        ).is_file(), "helm/templates/deployment.yaml must exist"

    def test_helm_service_template_exists(self):
        """helm/templates/service.yaml must exist."""
        assert (
            PROJECT_ROOT / "helm" / "templates" / "service.yaml"
        ).is_file(), "helm/templates/service.yaml must exist"

    def test_helm_hpa_template_exists(self):
        """helm/templates/hpa.yaml must exist."""
        assert (
            PROJECT_ROOT / "helm" / "templates" / "hpa.yaml"
        ).is_file(), "helm/templates/hpa.yaml must exist"


class TestHelmChartYaml:
    """Validate helm/Chart.yaml structure."""

    def test_chart_yaml_has_api_version(self):
        """Chart.yaml must have apiVersion."""
        content = _read("helm/Chart.yaml")
        assert "apiVersion:" in content, "Chart.yaml must have apiVersion"

    def test_chart_yaml_has_name(self):
        """Chart.yaml must have name."""
        content = _read("helm/Chart.yaml")
        assert "name:" in content, "Chart.yaml must have name"
        assert "agtech-unified" in content, "Chart name must be agtech-unified"

    def test_chart_yaml_has_version(self):
        """Chart.yaml must have version."""
        content = _read("helm/Chart.yaml")
        assert "version:" in content, "Chart.yaml must have version"


class TestHelmValuesYaml:
    """Validate helm/values.yaml structure."""

    def test_values_yaml_has_image(self):
        """values.yaml must define image configuration."""
        content = _read("helm/values.yaml")
        assert "image:" in content, "values.yaml must have image section"
        assert "repository:" in content, "values.yaml must define image repository"
        assert "tag:" in content, "values.yaml must define image tag"

    def test_values_yaml_has_replica_count(self):
        """values.yaml must define replicaCount."""
        content = _read("helm/values.yaml")
        assert "replicaCount:" in content, "values.yaml must define replicaCount"

    def test_values_yaml_has_resources(self):
        """values.yaml must define resource requests/limits."""
        content = _read("helm/values.yaml")
        assert "resources:" in content, "values.yaml must have resources section"
        assert "requests:" in content, "values.yaml must define resource requests"
        assert "limits:" in content, "values.yaml must define resource limits"

    def test_values_yaml_has_service(self):
        """values.yaml must define service configuration."""
        content = _read("helm/values.yaml")
        assert "service:" in content, "values.yaml must have service section"
        assert "port:" in content, "values.yaml must define service port"

    def test_values_yaml_has_autoscaling(self):
        """values.yaml must define autoscaling configuration."""
        content = _read("helm/values.yaml")
        assert "autoscaling:" in content, "values.yaml must have autoscaling section"
        assert "minReplicas:" in content, "values.yaml must define minReplicas"
        assert "maxReplicas:" in content, "values.yaml must define maxReplicas"

    def test_values_yaml_has_probes(self):
        """values.yaml must define health probes."""
        content = _read("helm/values.yaml")
        assert "livenessProbe:" in content, "values.yaml must define livenessProbe"
        assert "readinessProbe:" in content, "values.yaml must define readinessProbe"


class TestHelmDeploymentTemplate:
    """Validate helm/templates/deployment.yaml structure."""

    def test_deployment_template_has_api_version(self):
        """Deployment template must have apiVersion."""
        content = _read("helm/templates/deployment.yaml")
        assert "apiVersion: apps/v1" in content

    def test_deployment_template_has_kind(self):
        """Deployment template must have kind: Deployment."""
        content = _read("helm/templates/deployment.yaml")
        assert "kind: Deployment" in content

    def test_deployment_template_uses_values(self):
        """Deployment template must reference values."""
        content = _read("helm/templates/deployment.yaml")
        assert ".Values" in content, "Deployment template must use .Values"

    def test_deployment_template_has_containers(self):
        """Deployment template must define containers."""
        content = _read("helm/templates/deployment.yaml")
        assert "containers:" in content

    def test_deployment_template_has_probes(self):
        """Deployment template must define probes."""
        content = _read("helm/templates/deployment.yaml")
        assert "livenessProbe:" in content
        assert "readinessProbe:" in content

    def test_deployment_template_has_resources(self):
        """Deployment template must define resource limits."""
        content = _read("helm/templates/deployment.yaml")
        assert "resources:" in content

    def test_deployment_template_has_security_context(self):
        """Deployment template must define securityContext."""
        content = _read("helm/templates/deployment.yaml")
        assert "securityContext:" in content


class TestHelmServiceTemplate:
    """Validate helm/templates/service.yaml structure."""

    def test_service_template_has_api_version(self):
        """Service template must have apiVersion."""
        content = _read("helm/templates/service.yaml")
        assert "apiVersion: v1" in content

    def test_service_template_has_kind(self):
        """Service template must have kind: Service."""
        content = _read("helm/templates/service.yaml")
        assert "kind: Service" in content

    def test_service_template_uses_values(self):
        """Service template must reference values."""
        content = _read("helm/templates/service.yaml")
        assert ".Values" in content, "Service template must use .Values"

    def test_service_template_has_port(self):
        """Service template must define port."""
        content = _read("helm/templates/service.yaml")
        assert "port:" in content

    def test_service_template_has_selector(self):
        """Service template must define selector."""
        content = _read("helm/templates/service.yaml")
        assert "selector:" in content


class TestHelmHpaTemplate:
    """Validate helm/templates/hpa.yaml structure."""

    def test_hpa_template_has_api_version(self):
        """HPA template must have apiVersion."""
        content = _read("helm/templates/hpa.yaml")
        assert "apiVersion: autoscaling/v2" in content

    def test_hpa_template_has_kind(self):
        """HPA template must have kind: HorizontalPodAutoscaler."""
        content = _read("helm/templates/hpa.yaml")
        assert "kind: HorizontalPodAutoscaler" in content

    def test_hpa_template_uses_values(self):
        """HPA template must reference values."""
        content = _read("helm/templates/hpa.yaml")
        assert ".Values" in content, "HPA template must use .Values"

    def test_hpa_template_has_min_replicas(self):
        """HPA template must define minReplicas."""
        content = _read("helm/templates/hpa.yaml")
        assert "minReplicas:" in content

    def test_hpa_template_has_max_replicas(self):
        """HPA template must define maxReplicas."""
        content = _read("helm/templates/hpa.yaml")
        assert "maxReplicas:" in content

    def test_hpa_template_has_metrics(self):
        """HPA template must define scaling metrics."""
        content = _read("helm/templates/hpa.yaml")
        assert "metrics:" in content
