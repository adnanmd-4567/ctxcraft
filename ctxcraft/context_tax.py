from typing import Dict, List, Optional, Union
from ctxcraft.allocator import TokenCounter

class ContextTaxEstimator:
    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self.counter = token_counter or TokenCounter()

    def calculate_tax(
        self,
        system_prompt: str = "",
        tools: Optional[List[Dict]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, int]:

            system_tokens = self.counter.count_text(system_prompt) if system_prompt else 0
            tools_tokens = 0

            if tools:
                for tool in tools:
                    tools_tokens += self.counter.count_text(str(tool))

            history_tokens = self.counter.count_messages(history) if history else 0
            total_tax = system_tokens + tools_tokens + history_tokens

            return {
                "system_prompt_tokens": system_tokens,
                "tools_tokens": tools_tokens,
                "history_tokens": history_tokens,
                "total_tax": total_tax,
            }
    
    def attention_to_noise_ratio(
        self,
        signal_content: Union[str, List[Dict[str, str]]],
        total_content: Union[str, List[Dict[str, str]]],
    ) -> float:

        signal_tokens = self._count_flexible(signal_content)
        total_tokens = self._count_flexible(total_content)

        if total_tokens == 0:
            return 0.0

        return round(signal_tokens / total_tokens, 4)

    def _count_flexible(self, content: Union[str, List[Dict[str, str]]]) -> int:
        if isinstance(content, str):
            return self.counter.count_text(content)
        if isinstance(content, list):
            return self.counter.count_messages(content)
        return 0