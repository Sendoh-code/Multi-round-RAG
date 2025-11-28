# prompts.py

from typing import Dict, List


def format_contexts(contexts: List[Dict]) -> str:
    if not contexts:
        return "No reference passages are provided.\n"
    lines = []
    for i, ctx in enumerate(contexts, start=1):
        title = ctx.get("title") or ""
        text = ctx.get("text") or ""
        if title:
            header = f"[{i}] {title}"
        else:
            header = f"[{i}] Passage {i}"
        lines.append(header)
        lines.append(text.strip())
        lines.append("")  
    return "\n".join(lines)


def format_history(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history:
        speaker = turn.get("speaker", "user")
        txt = turn.get("text", "").strip()
        if not txt:
            continue
        if speaker == "user":
            lines.append(f"User: {txt}")
        else:
            lines.append(f"Assistant: {txt}")
    return "\n".join(lines)


def build_taskA_prompt(task: Dict) -> str:
    """
    create a complete input for LLM based on contexts
    """

    contexts = task.get("contexts", [])
    history = task.get("history", [])
    query = task.get("last_user_query", "")

    context_block = format_contexts(contexts)
    history_block = format_history(history)

    system_instructions = (
        "You are a helpful, factual assistant. "
        "Use ONLY the reference passages to answer the user's final question. "
        "If the passages do not contain the answer, say that you don't know.\n"
    )

    parts = [system_instructions]

    parts.append("Reference passages:\n")
    parts.append(context_block)

    if history_block:
        parts.append("Conversation so far:\n")
        parts.append(history_block)
        parts.append("")

    parts.append("Now answer the user's final question based on the passages above.\n")
    parts.append(f"User's final question: {query}\n")
    parts.append("Answer:")

    return "\n".join(parts)


def build_taskC_prompt(task: Dict, retrieved_docs:list=None) -> str:
    """
    create a complete input for LLM based on contexts
    """

    if retrieved_docs:
        docs = "\n\nRetrieved Passages:\n" + "\n".join(
            [f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)]
        ) + "\n\n"

    #contexts = task.get("contexts", [])
    #history = task.get("history", [])
    #query = task.get("last_user_query", "")

    #context_block = format_contexts(contexts)
    #history_block = format_history(history)

    system_instructions = (
        '''You are a helpful, factual assistant. 
        INSTRUCTIONS:
        - Answer the user question in 1–3 sentences.
        - Give only the final answer.
        - Do NOT provide explanations, reasoning steps, or background context.
        - Do NOT repeat or summarize the provided documents.
        - Do NOT mention the existence of documents.
        - Do NOT cite, list, or refer to sources.
        '''
    )

    parts = [system_instructions]

    parts.append(docs)

    #parts.append("Reference passages:\n")
    #parts.append(context_block)

    #if history_block:
     #   parts.append("Conversation so far:\n")
      #  parts.append(history_block)
       # parts.append("")

    #parts.append("Now answer the user's final question based on the passages above.\n")
    #parts.append(f"User's final question: {query}\n")


    return "\n".join(parts)
