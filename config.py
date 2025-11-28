# config.py

from pathlib import Path

# local project path (change later)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "/Users/yifan/Desktop/Learning Project/MLRAG/mt-rag-benchmark/human/generation_tasks"

# dataset path
TASK_A_FILE = DATA_DIR / "reference.jsonl"

# output path
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TASK_A_PREDICTIONS = OUTPUT_DIR / "taskA_predictions.jsonl"
TASK_C_PREDICTIONS = OUTPUT_DIR / "taskC_predictions.jsonl"
# model
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 或你喜欢的别的 instruct 模型
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9
