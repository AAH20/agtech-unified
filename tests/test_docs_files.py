"""Tests for documentation files: CONTRIBUTING, SECURITY, CHANGELOG, .env.example, etc."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_contributing_md_exists():
    assert (ROOT / "CONTRIBUTING.md").is_file()


def test_contributing_md_has_content():
    content = (ROOT / "CONTRIBUTING.md").read_text()
    assert len(content) > 500
    assert "pull request" in content.lower() or "PR" in content
    assert "test" in content.lower()


def test_security_md_exists():
    assert (ROOT / "SECURITY.md").is_file()


def test_security_md_has_content():
    content = (ROOT / "SECURITY.md").read_text()
    assert len(content) > 300
    assert "report" in content.lower() or "disclosure" in content.lower()


def test_changelog_md_exists():
    assert (ROOT / "CHANGELOG.md").is_file()


def test_changelog_md_has_content():
    content = (ROOT / "CHANGELOG.md").read_text()
    assert len(content) > 200
    assert "1.0.0" in content or "initial" in content.lower()


def test_code_of_conduct_md_exists():
    assert (ROOT / "CODE_OF_CONDUCT.md").is_file()


def test_code_of_conduct_has_content():
    content = (ROOT / "CODE_OF_CONDUCT.md").read_text()
    assert len(content) > 200


def test_env_example_exists():
    assert (ROOT / ".env.example").is_file()


def test_env_example_has_content():
    content = (ROOT / ".env.example").read_text()
    assert "JWT_SECRET" in content
    assert "DATABASE_URL" in content
    assert "API_PORT" in content


def test_pr_template_exists():
    assert (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").is_file()


def test_issue_templates_exist():
    issue_dir = ROOT / ".github" / "ISSUE_TEMPLATE"
    assert issue_dir.is_dir()
    assert (issue_dir / "bug_report.md").is_file()
    assert (issue_dir / "feature_request.md").is_file()


def test_adr_directory_exists():
    adr_dir = ROOT / "docs" / "adr"
    assert adr_dir.is_dir()
    assert (adr_dir / "0001-in-memory-backends.md").is_file()
    assert (adr_dir / "0002-pbft-consensus-simulation.md").is_file()
    assert (adr_dir / "0003-ngsi-ld-integration.md").is_file()


def test_testing_guide_exists():
    assert (ROOT / "docs" / "testing.md").is_file()


def test_testing_guide_has_content():
    content = (ROOT / "docs" / "testing.md").read_text()
    assert len(content) > 300
    assert "pytest" in content.lower()


def test_tutorials_directory_exists():
    tutorials_dir = ROOT / "tutorials"
    assert tutorials_dir.is_dir()
    assert (tutorials_dir / "01-getting-started.md").is_file()
    assert (tutorials_dir / "02-optimization-basics.md").is_file()
    assert (tutorials_dir / "03-swarm-coordination.md").is_file()
    assert (tutorials_dir / "04-iot-data-pipeline.md").is_file()
    assert (tutorials_dir / "05-digital-twin-simulation.md").is_file()
    assert (tutorials_dir / "06-decision-support.md").is_file()
    assert (tutorials_dir / "07-api-gateway.md").is_file()
    assert (tutorials_dir / "08-genomics.md").is_file()
    assert (tutorials_dir / "09-end-to-end-farm.md").is_file()
    assert (tutorials_dir / "10-deployment.md").is_file()


def test_k8s_manifests_exist():
    k8s_dir = ROOT / "k8s"
    assert k8s_dir.is_dir()
    assert (k8s_dir / "deployment.yaml").is_file()
    assert (k8s_dir / "service.yaml").is_file()
    assert (k8s_dir / "configmap.yaml").is_file()
    assert (k8s_dir / "hpa.yaml").is_file()
    assert (k8s_dir / "ingress.yaml").is_file()


def test_deployment_doc_expanded():
    content = (ROOT / "docs" / "deployment.md").read_text()
    lines = content.splitlines()
    assert len(lines) > 100, f"deployment.md only {len(lines)} lines, need 100+"
    assert "health" in content.lower()
    assert "rollback" in content.lower()
    assert "monitor" in content.lower()
    assert "SSL" in content or "TLS" in content
    assert "environment" in content.lower()


def test_deployment_doc_has_env_vars():
    content = (ROOT / "docs" / "deployment.md").read_text()
    assert "JWT_SECRET" in content
    assert "DATABASE_URL" in content
    assert "AGTECH_ENV" in content


def test_deployment_doc_has_resource_requirements():
    content = (ROOT / "docs" / "deployment.md").read_text()
    assert "resource" in content.lower() or "CPU" in content or "memory" in content.lower()


def test_deployment_doc_has_scaling():
    content = (ROOT / "docs" / "deployment.md").read_text()
    assert "scal" in content.lower()


def test_deployment_doc_has_edge_deployment():
    content = (ROOT / "docs" / "deployment.md").read_text()
    assert (
        "jetson" in content.lower() or "raspberry" in content.lower() or "edge" in content.lower()
    )


def test_examples_directory_has_multiple_files():
    examples_dir = ROOT / "examples"
    py_files = list(examples_dir.glob("*.py"))
    assert len(py_files) >= 10, f"Only {len(py_files)} example files, need 10+"


def test_example_swarm_coordination_exists():
    assert (ROOT / "examples" / "swarm_coordination.py").is_file()


def test_example_byzantine_consensus_exists():
    assert (ROOT / "examples" / "byzantine_consensus.py").is_file()


def test_example_task_allocation_exists():
    assert (ROOT / "examples" / "task_allocation.py").is_file()


def test_example_iot_pipeline_exists():
    assert (ROOT / "examples" / "iot_pipeline.py").is_file()


def test_example_context_broker_exists():
    assert (ROOT / "examples" / "context_broker_demo.py").is_file()


def test_example_digital_twin_exists():
    assert (ROOT / "examples" / "digital_twin_sim.py").is_file()


def test_example_knowledge_graph_exists():
    assert (ROOT / "examples" / "knowledge_graph_demo.py").is_file()


def test_example_ontology_exists():
    assert (ROOT / "examples" / "ontology_demo.py").is_file()


def test_example_decision_support_exists():
    assert (ROOT / "examples" / "decision_support.py").is_file()


def test_example_api_gateway_exists():
    assert (ROOT / "examples" / "api_gateway_demo.py").is_file()


def test_example_security_exists():
    assert (ROOT / "examples" / "security_demo.py").is_file()


def test_example_gps_antispoof_exists():
    assert (ROOT / "examples" / "gps_antispoof_demo.py").is_file()


def test_example_edge_ai_exists():
    assert (ROOT / "examples" / "edge_ai_demo.py").is_file()


def test_example_crop_vision_exists():
    assert (ROOT / "examples" / "crop_vision_demo.py").is_file()


def test_example_crispr_exists():
    assert (ROOT / "examples" / "crispr_demo.py").is_file()


def test_example_protein_exists():
    assert (ROOT / "examples" / "protein_demo.py").is_file()


def test_example_path_planning_exists():
    assert (ROOT / "examples" / "path_planning_demo.py").is_file()


def test_example_collision_avoidance_exists():
    assert (ROOT / "examples" / "collision_avoidance.py").is_file()


def test_example_fault_tolerance_exists():
    assert (ROOT / "examples" / "fault_tolerance_demo.py").is_file()


def test_example_unified_optimizer_exists():
    assert (ROOT / "examples" / "unified_optimizer_demo.py").is_file()


def test_example_event_bus_exists():
    assert (ROOT / "examples" / "event_bus_demo.py").is_file()


def test_example_onboarding_exists():
    assert (ROOT / "examples" / "onboarding_demo.py").is_file()


def test_example_end_to_end_farm_exists():
    assert (ROOT / "examples" / "end_to_end_farm.py").is_file()


def test_example_basic_tsp_exists():
    assert (ROOT / "examples" / "basic_tsp.py").is_file()


def test_example_basic_vrp_exists():
    assert (ROOT / "examples" / "basic_vrp.py").is_file()
