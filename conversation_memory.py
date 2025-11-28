# conversation_memory.py
from typing import List, Dict

class ConversationMemory:
    """
    Simple multi-turn memory module.
    Supports:
        - raw history (full text)
        - summarization 
    """

    def __init__(self, max_turns: int = 10, use_summary: bool = False):
        self.max_turns = max_turns
        self.use_summary = use_summary
        self.memory: Dict[str, List[str]] = {}   # conversation_id -> list of turns
        self.summary: Dict[str, str] = {}        # conversation_id -> summary text

    def reset(self, conv_id: str):
        self.memory[conv_id] = []
        self.summary[conv_id] = ""

    def add_turn(self, conv_id: str, user_utt: str, assistant_utt: str = ""):
        """ Store one multi-turn record. """
        if conv_id not in self.memory:
            self.reset(conv_id)

        entry = f"User: {user_utt}"
        if assistant_utt:
            entry += f"\nAssistant: {assistant_utt}"

        self.memory[conv_id].append(entry)

        # clip to max_turns
        if len(self.memory[conv_id]) > self.max_turns:
            self.memory[conv_id] = self.memory[conv_id][-self.max_turns:]

    def build_context(self, conv_id: str) -> str:
        """Return the memory as a block of text."""
        if conv_id not in self.memory:
            return ""

        raw_hist = "\n".join(self.memory[conv_id])

        if not self.use_summary:
            return raw_hist

        # optional summary mechanism (not implemented yet)
        summary = self.summary.get(conv_id, "")
        context = f"[SUMMARY]\n{summary}\n\n[RECENT TURNS]\n{raw_hist}"
        return context

    def update_summary(self, conv_id: str, text: str):
        """Optional summarization function."""
        self.summary[conv_id] = text
