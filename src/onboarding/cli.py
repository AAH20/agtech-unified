"""Onboarding CLI for agtech-unified.

Provides `agtech onboard init/status/validate` commands for headless
or CI/CD onboarding workflows.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.onboarding.dependencies import ModuleDependencyGraph
from src.onboarding.health import SystemHealth
from src.onboarding.modules import ModuleRegistry
from src.onboarding.sizing import OrganizationProfiler
from src.onboarding.templates import ConfigTemplate


def _cmd_init(args: argparse.Namespace) -> int:
    """Handle `agtech onboard init`."""
    if args.employees is None:
        print("Error: --employees is required", file=sys.stderr)
        return 2
    if args.revenue is None:
        print("Error: --revenue is required", file=sys.stderr)
        return 2
    if args.employees < 0:
        print("Error: --employees must be non-negative", file=sys.stderr)
        return 1
    if args.revenue < 0:
        print("Error: --revenue must be non-negative", file=sys.stderr)
        return 1

    profiler = OrganizationProfiler()
    registry = ModuleRegistry()
    graph = ModuleDependencyGraph.from_registry(registry)

    score_result = profiler.score_organization(
        employees=args.employees,
        revenue=args.revenue,
        fields=args.fields or 0,
        sensors=args.sensors or 0,
        has_dedicated_it=args.dedicated_it,
    )
    tier = profiler.recommend_tier_scored(
        employees=args.employees,
        revenue=args.revenue,
        fields=args.fields or 0,
        sensors=args.sensors or 0,
        has_dedicated_it=args.dedicated_it,
    )
    modules = registry.get_modules(tier)
    install_order = graph.get_install_order(modules)

    if args.json:
        output = {
            "tier": tier,
            "total_score": score_result["total_score"],
            "factors": score_result["factors"],
            "modules": modules,
            "install_order": install_order,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Tier: {tier}")
        print(f"Score: {score_result['total_score']}")
        print(f"Modules: {', '.join(modules)}")
        print(f"Install order: {' -> '.join(install_order)}")

    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Handle `agtech onboard status`."""
    health = SystemHealth()
    status = health.get_status()

    print("Onboarding Status")
    print("-----------------")
    print(f"Health: {status.value}")
    print("Profile: not initialized")
    print("Progress: 0%")
    print("")
    print("Run 'agtech onboard init --employees N --revenue R' to begin.")

    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle `agtech onboard validate`."""
    templates = ConfigTemplate()

    if args.config:
        try:
            with open(args.config) as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error reading config: {exc}", file=sys.stderr)
            return 1
    else:
        # Validate all default templates
        all_valid = True
        for tier in templates.get_all_templates():
            config = templates.get_template(tier)
            errors = templates.validate_template(config)
            if errors:
                all_valid = False
                print(f"Config for tier '{tier}' is INVALID:")
                for err in errors:
                    print(f"  - {err}")
            else:
                print(f"Config for tier '{tier}': OK")
        return 0 if all_valid else 1

    errors = templates.validate_template(config)
    if errors:
        print("Config is INVALID:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Config is valid.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the onboarding CLI."""
    parser = argparse.ArgumentParser(
        prog="agtech onboard",
        description="AgTech Unified onboarding CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize onboarding")
    init_parser.add_argument(
        "--employees",
        type=int,
        default=None,
        help="Number of employees (required)",
    )
    init_parser.add_argument(
        "--revenue",
        type=float,
        default=None,
        help="Annual revenue in USD (required)",
    )
    init_parser.add_argument(
        "--fields",
        type=int,
        default=None,
        help="Number of fields/managed plots",
    )
    init_parser.add_argument(
        "--sensors",
        type=int,
        default=None,
        help="Number of sensors",
    )
    init_parser.add_argument(
        "--dedicated-it",
        action="store_true",
        help="Has dedicated IT staff",
    )
    init_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    init_parser.set_defaults(func=_cmd_init)

    # status
    status_parser = subparsers.add_parser("status", help="Show onboarding status")
    status_parser.set_defaults(func=_cmd_status)

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (validates all templates if omitted)",
    )
    validate_parser.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the onboarding CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 for success, 1 for errors, 2 for usage errors.
    """
    if argv and argv[0] == "onboard":
        argv = argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(2)

    sys.exit(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
