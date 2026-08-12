from typing import Dict, List, Union, Optional
import tiktoken

class TokenCounter:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def count_messages(self, messages: List[Dict[str,str]]) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += 3
            for key, value in message.items():
                if not isinstance(value, str):
                    continue
                num_tokens += self.count_text(value)
                if key == "name":
                    num_tokens -= 1
        num_tokens += 3
        return num_tokens

class ContextBudget:

    def __init__(
        self,
        max_con_win: int = 128000,
        res_out_tok: int = 4096,
        sys_p: float = 0.15,
        rag_p: float = 0.50,
        his_p: float = 0.35,
    ):
        # max_con_win: maximum_context_window
        # res_out_tok: reserved_output_tokens
        # sys_p: system_percentage
        # rag_p: rag_percentage
        # his_p: history_percentage
        
        total_pct = sys_p + rag_p + his_p
        if not (0.99 <= total_pct <= 1.01):
            raise ValueError("total_pct must be 1.0")

        self.max_con_win = max_con_win
        self.res_out_tok = res_out_tok

        self.total_input_budget = max_con_win - res_out_tok
        if self.total_input_budget <= 0:
            raise ValueError("max context window must be >= res_out_tok")

        self.sys_budget = int(self.total_input_budget * sys_p)
        self.rag_budget = int(self.total_input_budget * rag_p)
        self.his_budget = self.total_input_budget - self.sys_budget - self.rag_budget

    def summary(self) -> Dict[str,int]:
        return {
            "max_context_window": self.max_con_win,
            "reserved_output_tokens": self.res_out_tok,
            "total_input_budget": self.total_input_budget,
            "system_budget": self.sys_budget,
            "rag_budget": self.rag_budget,
            "history_budget": self.his_budget,
        }
# new = TokenCounter()
# print(new.encoder.encode("tiktoken is great!"))

   
