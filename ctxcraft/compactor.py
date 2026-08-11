from typing import List, Dict, Optional
from ctxcraft.allocator import TokenCounter, ContextBudget

class Compactor:
    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self.counter = token_counter or TokenCounter()

    def prune_history(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        preserve_n: int = 0
    ) -> List[Dict[str, str]]:

        # messages:
        # max_tokens:
        # preserve_n:

        if not messages:
            return []

        total_tokens = self.counter.count_messages(messages)
        if total_tokens <= max_tokens:
            return messages

        anchored = messages[:preserve_n]
        anchored_tokens = self.counter.count_messages(anchored) if anchored else 0
        rem_budget = max_tokens - anchored_tokens
        if rem_budget <= 0:
            return self._fit_subset(anchored, max_tokens)

        recent_candidates = messages[preserve_n:]
        selected_recent = []
        accumulated_tokens = 0

        for msg in reversed(recent_candidates):
            msg_tokens = self.counter.count_messages([msg])
            if accumulated_tokens + msg_tokens > rem_budget:
                break
            selected_recent.insert(0, msg)
            accumulated_tokens += msg_tokens

        return anchored + selected_recent

    def _fit_subset(self, messages, max_tokens):
        subset = []
        current = 0

        for msg in messages:
            x = self.counter.count_messages([msg])
            if current + x > max_tokens:
                break
            subset.append(msg)
            current += x

        return subset

