from src.app import app


def main() -> None:
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
