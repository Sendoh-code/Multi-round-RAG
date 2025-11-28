# data_loader.py

import json
from typing import List, Dict, Any, Iterable


def read_jsonl(path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_taskA(path, limit: int = None) -> List[Dict[str, Any]]:
    """
    read taskA sample from human/generation_tasks/reference.jsonl 
    

   return:
    {
        "task_id": str,
        "conversation_id": str,
        "turn": int,
        "collection": str,
        "dataset": str,
        "contexts": [ {document_id, text, title, url, score, ...}, ...],
        "input_turns": [ {speaker, text, metadata, ...}, ...],
        "last_user_query": str,
        "history": [ {speaker, text}, ...  ],
        "gold_answers": [str]  
    }
    """
    tasks = []
    for i, obj in enumerate(read_jsonl(path)):
        input_turns = obj.get("input", []) or []

        # history + current query
        last_user_query = ""
        history = []
        for turn in input_turns:
            if turn.get("speaker") == "user":
                last_user_query = turn.get("text", "")
            # store all the  history/context in prepare for building the prompt
            history.append(
                {
                    "speaker": turn.get("speaker", ""),
                    "text": turn.get("text", ""),
                }
            )

        # gold answers: retract from target
        targets = obj.get("targets", []) or []
        gold_answers = [t.get("text", "") for t in targets]

        tasks.append(
            {
                "task_id": obj.get("task_id"),
                "conversation_id": obj.get("conversation_id"),
                "turn": obj.get("turn"),
                "collection": obj.get("collection"),
                "dataset": obj.get("dataset"),
                "contexts": obj.get("contexts", []) or [],
                "input_turns": input_turns,
                "last_user_query": last_user_query,
                "history": history[:-1] if history else [],
                "gold_answers": gold_answers,
                "raw": obj, 
            }
        )

        if limit is not None and len(tasks) >= limit:
            break

    print(f"[load_taskA] Loaded {len(tasks)} tasks from {path}")
    return tasks
