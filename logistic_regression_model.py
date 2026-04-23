from forecast_failure_model import main as forecast_failure_main


def main() -> None:
    print(
        "Deprecated: logistic_regression_model.py is now a compatibility wrapper. "
        "Use 'python main.py forecast-model ...' for unified execution."
    )
    forecast_failure_main()


if __name__ == "__main__":
    main()