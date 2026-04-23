from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
REPO_ROOT = SRC_DIR.parent
DATA_DIR = REPO_ROOT / "data"
ASSETS_DIR = REPO_ROOT / "assets"
PROMPTS_DIR = REPO_ROOT / "prompts"
OUTPUTS_DIR = REPO_ROOT / "outputs"
