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

        preserve_n = min(preserve_n, len(messages))

        total_tokens = self.counter.count_messages(messages)
        if total_tokens <= max_tokens:
            return messages

        anchored = messages[:preserve_n]
        anchored_tokens = self.counter.count_messages(anchored) if anchored else 0
        rem_budget = max_tokens - anchored_tokens
        if rem_budget <= 0:
            if anchored_tokens > max_tokens:
                raise ValueError(
                    f"preserve_n = {preserve_n} message requires {anchored_tokens} tokens"
                    f"which exceeds maximum tokens= {max_tokens}. Increase max_tokens or reduce preserve_n"
                )
            return self._fit_subset(anchored, max_tokens)

        recent_candidates = messages[preserve_n:]
        selected_recent = []
        accumulated_tokens = 0

        for msg in reversed(recent_candidates):
            msg_tokens = self.counter.count_messages([msg])
            if accumulated_tokens + msg_tokens > rem_budget:
                remaining = rem_budget - accumulated_tokens
                if remaining > 20:
                    truncated_msg = self._truncate_message(msg, remaining)
                    if truncated_msg:
                        selected_recent.insert(0, truncated_msg)
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

    def _truncate_message(self, msg, token_budget):
        content = msg.get("content", "")
        if not isinstance(content, str) or not content:
            return None
        encoded = self.counter.encoder.encode(content)
        if len(encoded) <= token_budget:
            return msg

        suffix = "..[truncated]"
        suffix_tokens = len(self.counter.encoder.encode(suffix))
        keep = max(token_budget - suffix_tokens, 0)
        truncated_text = self.counter.encoder.decode(encoded[:keep]) + suffix
        return {**msg, "content": truncated_text}

# messages = [
#          {"role": "user", "content": "Turn 1: " + "brainrot " * 50},
#          {"role": "assistant", "content": "Reply 1: " + "meow " * 50},
#          {"role": "user", "content": "Turn 2: " + "aura_maxing " * 50},
#          {"role": "assistant", "content": "Reply 2: " + "bark " * 50},
#      ]
# new = Compactor()
# print(new.prune_history(messages=messages, max_tokens=115, preserve_n=1))


