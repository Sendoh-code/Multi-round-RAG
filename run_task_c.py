# run_task_a.py

import json
import argparse
from pathlib import Path
from rag_retriever import FaissRetriever
from tqdm import tqdm

from config import TASK_A_FILE, TASK_C_PREDICTIONS, DEFAULT_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE, TOP_P
from data_loader import load_taskA
from prompts import build_taskA_prompt, build_taskC_prompt
from model import LocalLLM
from conversation_memory import ConversationMemory

#memory = ConversationMemory(max_turns=6, use_summary=False)


def runC(memory=None,conv_id=None,task=None,llm=None):
    retriever = FaissRetriever(
    index_path="faiss_index.bin",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    metadata_path="passages.jsonl",
    top_k=5
)
    if memory!=None:
        context_memory = memory.build_context(conv_id)
    else:
        context_memory = ''

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
        default=str(TASK_C_PREDICTIONS),
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

    with open(output_path, "a", encoding="utf-8") as fout:
        
        query_text = task.get("last_user_query", "")
        docs = retriever.retrieve(query_text)
        prompt = (build_taskC_prompt(task,docs)+ '\n\n' +
        str(context_memory) +
        "Now answer the user's final question based on the passages above.\n"+
        f"User's final question: {query_text}\n" + '[ENDOFPROMPT]\n\n')
        if not True: #llm.check_answerability(query_text, docs):
            answer= "I'm sorry, but I don't have the answer to your question."

            # 否则继续正常 RAG
        else:
            answer = llm.generate(prompt)

        record = {
            "task_id": task["task_id"],
            "conversation_id": task["conversation_id"],
            "turn": task["turn"],
            "answer": answer,
            "gold_answers": task["gold_answers"],
            # keep raw task to track
            "raw_task": task.get("last_user_query", ""),
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[main] Done. Conversation {conv_id} Predictions saved to output_path")
    return record

if __name__ == "__main__":
    llm = LocalLLM(
    model_name=DEFAULT_MODEL_NAME,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    )
    tasks = load_taskA(str(TASK_A_FILE), limit=1)
    task = tasks[0]
    runC(task=task)
