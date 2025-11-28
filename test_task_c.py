from conversation_memory import ConversationMemory
from run_task_c import runC
from data_loader import load_taskA
from model import LocalLLM
from config import TASK_A_FILE, TASK_C_PREDICTIONS, DEFAULT_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE, TOP_P

if __name__=='__main__':
    llm = LocalLLM(
    model_name=DEFAULT_MODEL_NAME,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    )
    all_tasks = load_taskA(str(TASK_A_FILE),limit=5)

    # for every conversation id, build its own memory
    memory = ConversationMemory()
    current_coversation=0

    for task in all_tasks:
        if current_coversation != task["conversation_id"]:
            memory.reset(task["conversation_id"])
            current_coversation = task["conversation_id"]
        record = runC(llm=llm, memory=memory, conv_id=current_coversation, task=task)
        memory.add_turn(conv_id = current_coversation, 
                        user_utt = task.get("last_user_query", ""), 
                        assistant_utt = record['answer'])
