# run_task_a.py

import json
import argparse
from pathlib import Path

from tqdm import tqdm

from config import TASK_A_FILE, TASK_A_PREDICTIONS, DEFAULT_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE, TOP_P
from data_loader import load_taskA
from prompts import build_taskA_prompt
from model import LocalLLM



def main():
    parser = argparse.ArgumentParser(description="Run MT-RAG Task A (Reference) generation.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(TASK_A_FILE),
        help="Path to reference.jsonl",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(TASK_A_PREDICTIONS),
        help="Where to save predictions jsonl.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HF model id for local LLM.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit number of tasks for quick test.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    tasks = load_taskA(data_path, limit=args.max_tasks)

    print(f"[main] Loading LLM: {args.model_name}")

    llm = LocalLLM(
    model_name=args.model_name,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    )

    print(f"[main] Generating answers for {len(tasks)} tasks...")
    with open(output_path, "w", encoding="utf-8") as fout:
        for task in tqdm(tasks):
            prompt = build_taskA_prompt(task)
            raw_answer = llm.generate(prompt)
            answer = raw_answer

            record = {
                "task_id": task["task_id"],
                "conversation_id": task["conversation_id"],
                "turn": task["turn"],
                "collection": task["collection"],
                "dataset": task["dataset"],
                "answer": answer,
                "gold_answers": task["gold_answers"],
                # keep raw task to track
                "raw_task": task["raw"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[main] Done. Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
