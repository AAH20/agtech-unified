"""Tests for k8s manifests and example file content."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_k8s_deployment_yaml_valid():
    content = (ROOT / "k8s" / "deployment.yaml").read_text()
    assert "apiVersion: apps/v1" in content
    assert "kind: Deployment" in content
    assert "agtech-unified" in content
    assert "replicas" in content
    assert "containers" in content


def test_k8s_service_yaml_valid():
    content = (ROOT / "k8s" / "service.yaml").read_text()
    assert "apiVersion: v1" in content
    assert "kind: Service" in content
    assert "agtech-unified" in content
    assert "port" in content


def test_k8s_configmap_yaml_valid():
    content = (ROOT / "k8s" / "configmap.yaml").read_text()
    assert "apiVersion: v1" in content
    assert "kind: ConfigMap" in content
    assert "AGTECH_ENV" in content


def test_k8s_hpa_yaml_valid():
    content = (ROOT / "k8s" / "hpa.yaml").read_text()
    assert "apiVersion: autoscaling/v2" in content
    assert "kind: HorizontalPodAutoscaler" in content
    assert "agtech-unified" in content


def test_k8s_ingress_yaml_valid():
    content = (ROOT / "k8s" / "ingress.yaml").read_text()
    assert "apiVersion: networking.k8s.io/v1" in content
    assert "kind: Ingress" in content
    assert "agtech-unified" in content


def test_example_files_have_docstrings():
    """All example files must have a module docstring."""
    examples_dir = ROOT / "examples"
    for py_file in examples_dir.glob("*.py"):
        content = py_file.read_text().strip()
        assert content.startswith('"""') or content.startswith(
            "'''"
        ), f"{py_file.name} missing module docstring"


def test_example_files_have_main_guard():
    """All example files should have if __name__ == '__main__' guard."""
    examples_dir = ROOT / "examples"
    for py_file in examples_dir.glob("*.py"):
        content = py_file.read_text()
        assert (
            "__name__" in content and "__main__" in content
        ), f"{py_file.name} missing __main__ guard"


def test_example_files_import_from_src():
    """Example files should import from src modules."""
    examples_dir = ROOT / "examples"
    for py_file in examples_dir.glob("*.py"):
        content = py_file.read_text()
        # Should have sys.path manipulation or direct src imports
        assert "src" in content or "sys.path" in content, f"{py_file.name} doesn't reference src"


def test_swarm_coordination_example_content():
    content = (ROOT / "examples" / "swarm_coordination.py").read_text()
    assert "SwarmCoordinator" in content
    assert "Agent" in content


def test_byzantine_consensus_example_content():
    content = (ROOT / "examples" / "byzantine_consensus.py").read_text()
    assert "ByzantineConsensus" in content


def test_iot_pipeline_example_content():
    content = (ROOT / "examples" / "iot_pipeline.py").read_text()
    assert "MQTTClient" in content or "SensorReading" in content


def test_digital_twin_example_content():
    content = (ROOT / "examples" / "digital_twin_sim.py").read_text()
    assert "DigitalTwin" in content


def test_security_example_content():
    content = (ROOT / "examples" / "security_demo.py").read_text()
    assert "ZeroTrustAuth" in content


def test_api_gateway_example_content():
    content = (ROOT / "examples" / "api_gateway_demo.py").read_text()
    assert "APIGateway" in content or "FastAPI" in content


def test_crispr_example_content():
    content = (ROOT / "examples" / "crispr_demo.py").read_text()
    assert "CRISPRDesigner" in content


def test_end_to_end_farm_example_content():
    content = (ROOT / "examples" / "end_to_end_farm.py").read_text()
    # Should reference multiple modules
    assert "TSP" in content or "tsp" in content
    assert "IoT" in content or "iot" in content or "sensor" in content.lower()
