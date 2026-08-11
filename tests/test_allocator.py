import pytest
from ctxcraft.allocator import TokenCounter, ContextBudget

def test_token_counter_basic_text():
    counter = TokenCounter(model_name="gpt-4o")
    text = "No other sadness in the world would do"
    tokens = counter.count_text(text)
    assert tokens > 0
    assert counter.count_text("") == 0

def test_token_counter_messages():
    counter = TokenCounter(model_name="gpt-4o")
    messages = [
        {"role": "system", "content": "Receipts are being kept"},
        {"role": "user", "content": "I bow my head in gratitude"}
    ]
    tokens = counter.count_messages(messages)
    assert tokens > 10

def test_context_budget():
    budget = ContextBudget(
        max_con_win= 10000,
        res_out_tok= 2000,
        sys_p= 0.20,
        rag_p= 0.50,
        his_p= 0.30,
    )

    assert budget.total_input_budget == 8000
    assert budget.sys_budget == 1600
    assert budget.rag_budget == 4000
    assert budget.his_budget == 2400

def test_invalid_budget_percentage():
    with pytest.raises(ValueError):
        ContextBudget(sys_p=0.8, rag_p=0.8, his_p=0.8)

from typing import List, Dict, Optional
from ctxcraft.allocator import TokenCounter


