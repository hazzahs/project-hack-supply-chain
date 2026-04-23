from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_hack_supply_chain.forecast_failure import main as forecast_failure_main


def main() -> None:
    print(
        "Deprecated: logistic_regression_model.py is now a compatibility wrapper. "
        "Use `python scripts/forecast_failure_model.py` or `uv run forecast-failure-model`."
    )
    forecast_failure_main()


if __name__ == "__main__":
    main()
