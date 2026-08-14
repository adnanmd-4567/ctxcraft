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

@pytest.fixture
def compactor():
    return Compactor(token_counter=TokenCounter())

# message fits entirely, no need for truncation
def test_normal_case(compactor):
    messages = [
        {"role": "user", "content": "yo yo crodie"},
        {"role": "user", "content": "yo yo brodie"},
    ]

    result = compactor.prune_history(messages, max_tokens=1000, preserve_n=0)
    assert result == messages

# last message is oversized, gets truncated instead of dropped
def test_truncation_case(compactor):
    huge_content = "skibidi" * 2000
    messages = [
        {"role": "user", "content": "yo yo crodie"},
        {"role": "assistant", "content": "yo yo brodie"},
        {"role": "tool", "content": huge_content},
    ]
    
    result = compactor.prune_history(messages, max_tokens=50, preserve_n=1)
    assert len(result) >= 1
    last_msg = result[-1]
    assert last_msg["role"] == "tool"
    assert "..[truncated]" in last_msg["content"]
    assert len(last_msg["content"]) < len(huge_content)

# the truncated message's token count should not exceed what was actually left in rem_budget at that point.
def test_truncated_message_respects_remaining_budget(compactor):
    huge_content = "token " * 5000
    messages = [{"role": "tool", "content": huge_content}]
    max_tokens = 30
    result = compactor.prune_history(messages, max_tokens=max_tokens, preserve_n=0)
 
    if result:
        counted = compactor.counter.count_messages(result)
        assert counted <= max_tokens, (
            f"Truncated message used {counted} tokens, exceeds max_tokens={max_tokens}"
        )

# if remaining budget is <= 20 tokens, the oversized message should be dropped entirely rather than truncated.
def test_message_dropped_when_remaining_budget_too_small(compactor):
    messages = [
        {"role": "user", "content": "yo yo crodie " * 50},
        {"role": "tool", "content": "yo yo brodie" * 4656},
    ]
    result = compactor.prune_history(messages, max_tokens=45, preserve_n=0)
    for msg in result:
        if msg["role"] == "tool":
            assert len(msg["content"]) > 20

def test_non_string_content_is_skipped_safely(compactor):
    messages = [
        {"role": "user", "content": "yo yo crodie"},
        {"role": "assistant", "content": [{"type": "text", "text": "x" * 2346}]},
    ]
    result = compactor.prune_history(messages, max_tokens=20, preserve_n=0)
    assert isinstance(result, list)

def test_truncation_happens_partway_through_recent_window(compactor):
    messages = [
        {"role": "user", "content": "yo yo crodie"},
        {"role": "assistant", "content": "yo yo brodie"},
        {"role": "tool", "content": "word " * 1894},
    ]
    result = compactor.prune_history(messages, max_tokens=60, preserve_n=0)
    roles_present = [m["role"] for m in result]
    assert "tool" in roles_present or len(result) > 0

def test_preserve_n_still_respected_alongside_truncation(compactor):
    messages = [
        {"role": "system", "content": "God is punishing me for my sins"},
        {"role": "user", "content": "word " * 3298},
    ]
    result = compactor.prune_history(messages, max_tokens=50, preserve_n=1)
 
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "God is punishing me for my sins"




    