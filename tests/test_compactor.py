import pytest

from ctxcraft.allocator import TokenCounter, ContextBudget
from ctxcraft.compactor import Compactor

def test_prune_history_budget():
    compactor = Compactor()
    messages = [
        {"role": "user", "content": "Turn 1: " + "brainrot " * 50},
        {"role": "assistant", "content": "Reply 1: " + "meow " * 50},
        {"role": "user", "content": "Turn 2: " + "aura_maxing " * 50},
        {"role": "assistant", "content": "Reply 2: " + "bark " * 50},
    ]
    pruned = compactor.prune_history(messages, max_tokens=100)
    assert len(pruned) < len(messages)
    assert pruned[-1]["content"] == messages[-1]["content"]

def test_prune_history_anchored_turns():
    compactor = Compactor()
    messages = [
        {"role": "user", "content": "ANCHOR: Initial setup rules."},
        {"role": "assistant", "content": "Old middle turn " * 30},
        {"role": "user", "content": "Recent turn " * 10},
    ]

    pruned = compactor.prune_history(messages, max_tokens=80, preserve_n=1)
    assert pruned[0]["content"] == "ANCHOR: Initial setup rules."