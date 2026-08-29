from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "paper.yaml"
TASK_ORDER = ("perception", "imagination")
LABEL_TO_TASK = {0: "perception", 1: "imagination"}
TASK_TO_LABEL = {value: key for key, value in LABEL_TO_TASK.items()}
