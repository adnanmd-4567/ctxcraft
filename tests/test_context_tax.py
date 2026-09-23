import pytest
from ctxcraft.context_tax import ContextTaxEstimator

@pytest.fixture
def estimator():
    return ContextTaxEstimator()

def test_calculate_tax_with_all_components(estimator):
    result = estimator.calculate_tax(
        system_prompt="You are a helpful assistant.",
        tools=[{"name": "search", "description": "Search the web"}],
        history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    assert result["system_prompt_tokens"] > 0
    assert result["tools_tokens"] > 0
    assert result["history_tokens"] > 0
    assert result["total_tax"] == (
        result["system_prompt_tokens"]
        + result["tools_tokens"]
        + result["history_tokens"]
    )

def test_calculate_tax_with_no_inputs_is_zero(estimator):
    result = estimator.calculate_tax()
    assert result["total_tax"] == 0
    assert result["system_prompt_tokens"] == 0
    assert result["tools_tokens"] == 0
    assert result["history_tokens"] == 0

def test_calculate_tax_with_only_system_prompt(estimator):
    result = estimator.calculate_tax(system_prompt="Be concise.")
    assert result["system_prompt_tokens"] > 0
    assert result["tools_tokens"] == 0
    assert result["history_tokens"] == 0
    assert result["total_tax"] == result["system_prompt_tokens"]

def test_attention_to_noise_ratio_high_signal(estimator):
    signal = "What is the capital of France?"
    total = signal 
    ratio = estimator.attention_to_noise_ratio(signal, total)
    assert ratio == 1.0

def test_attention_to_noise_ratio_low_signal(estimator):
    signal = "What is the capital of France?"
    total = signal + (" filler text " * 500) 
    ratio = estimator.attention_to_noise_ratio(signal, total)
    assert 0.0 < ratio < 0.1

def test_attention_to_noise_ratio_zero_total_is_safe(estimator):
    ratio = estimator.attention_to_noise_ratio("", "")
    assert ratio == 0.0

def test_attention_to_noise_ratio_accepts_message_lists(estimator):
    signal = [{"role": "user", "content": "What's the weather?"}]
    total = [
        {"role": "system", "content": "You are an assistant. " * 50},
        {"role": "user", "content": "What's the weather?"},
    ]
    ratio = estimator.attention_to_noise_ratio(signal, total)
    assert 0.0 < ratio < 1.0