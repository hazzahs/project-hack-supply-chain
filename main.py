import argparse
import sys


def _run_with_passthrough(module_entry, argv: list[str]) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *argv]
        module_entry()
    finally:
        sys.argv = original_argv


def _run_dash_app(app_factory, debug: bool = True) -> None:
    app_factory().run(debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for forecast modeling, PCA workflows, and dashboards."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "forecast-model",
        help="Run logistic forecast failure model (passes args to forecast_failure_model.py).",
        add_help=False,
    )

    subparsers.add_parser(
        "pca-analysis",
        help="Run PCA analysis (passes args to pca_analysis.py).",
        add_help=False,
    )

    subparsers.add_parser(
        "pca-linear-workflow",
        help="Run PCA -> linear workflow (passes args to pca_linear_workflow.py).",
        add_help=False,
    )

    subparsers.add_parser(
        "pca-visuals",
        help="Build Plotly visuals from workflow outputs (passes args to plotly_pca_linear_visuals.py).",
        add_help=False,
    )

    dash_upload_parser = subparsers.add_parser(
        "dashboard-upload",
        help="Launch the scored/raw upload dashboard.",
    )
    dash_upload_parser.add_argument("--debug", action="store_true", help="Run Dash app in debug mode")

    dash_pca_parser = subparsers.add_parser(
        "dashboard-pca",
        help="Launch the PCA + linear upload dashboard.",
    )
    dash_pca_parser.add_argument("--debug", action="store_true", help="Run Dash app in debug mode")

    subparsers.add_parser(
        "legacy-logistic",
        help="Run legacy logistic wrapper (delegates to forecast-model).",
        add_help=False,
    )

    args, passthrough = parser.parse_known_args()

    if args.command == "forecast-model":
        from forecast_failure_model import main as forecast_main

        _run_with_passthrough(forecast_main, passthrough)
        return

    if args.command == "pca-analysis":
        from pca_analysis import main as pca_main

        _run_with_passthrough(pca_main, passthrough)
        return

    if args.command == "pca-linear-workflow":
        from pca_linear_workflow import main as workflow_main

        _run_with_passthrough(workflow_main, passthrough)
        return

    if args.command == "pca-visuals":
        from plotly_pca_linear_visuals import main as visuals_main

        _run_with_passthrough(visuals_main, passthrough)
        return

    if args.command == "dashboard-upload":
        from plotly_upload_dashboard import app

        _run_dash_app(lambda: app, debug=bool(args.debug))
        return

    if args.command == "dashboard-pca":
        from plotly_pca_linear_upload_dashboard import app

        _run_dash_app(lambda: app, debug=bool(args.debug))
        return

    if args.command == "legacy-logistic":
        from logistic_regression_model import main as legacy_main

        _run_with_passthrough(legacy_main, passthrough)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
