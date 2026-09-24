# CtxCraft

CtxCraft is a Python toolkit that helps with budgeting, pruning and securing LLM context windows. It was built from scratch so that I can understand how libraries like LangChain work underneath.

## Why

LLM context windows have a fixed size. Hold everything: system prompts, tool definitions, conversation history and the user’s message. When this is not managed three problems appear. First conversations break when history overflows. Second untrusted user input can inject instructions, a problem known as prompt injection. Third the token and cost overhead is. Not obvious. CtxCraft solves all three problems.

## Features

- **`ContextBudget` / `TokenCounter`** — Allocates the context window across system, RAG and history budgets.
- **`Compactor`** — Trims conversation history so it fits the budget; it shortens messages that're too long instead of removing them completely.
- **`DelimiterManager`** — Surrounds content with XML tags. Cleans any injected tag patterns, which blocks prompt injection.
- **`ContextTaxEstimator`** — Calculates token overhead before the user’s message is added to the context.
- **`MemoryCache`** — A JSON scratchpad that keeps agent state across turns.

## Quick start

```bash
pip install git+https://github.com/adnanmd-4567/ctxcraft.git
```

```python
from ctxcraft.allocator import ContextBudget
from ctxcraft.compactor import Compactor
from ctxcraft.delimiter import DelimiterManager

budget = ContextBudget(max_con_win=32000, res_out_tok=1024)
pruned = Compactor().prune_to_budget(conversation_history, budget, preserve_n=2)

prompt = DelimiterManager().wrap_many({
    "system": "You are a helpful assistant.",
    "history": str(pruned),
    "user_input": user_message,  # sanitized even if it contains an injection attempt
})
```

## Demo

**Budget Simulator** — Adjust sliders for context window size and allocation percentages, and watch the budget breakdown update live.
![Budget Simulator](docs/screenshots/budget-simulator-1.jpg)
![Budget Simulator 2](docs/screenshots/budget-simulator-2.jpg)

**Pruning Preview** — Paste a conversation and a token limit, and see exactly which messages survive — with real token counts and percentage saved.
![Pruning Preview](docs/screenshots/pruning-preview-1.jpg)
![Pruning Preview 2](docs/screenshots/pruning-preview-2.jpg)

**Injection Sandbox** — Type an injection attempt yourself. This side-by-side shows the same malicious input handled two ways: raw concatenation (left, vulnerable — the fake <system> tag is real, structural markup) versus DelimiterManager (right, protected — the same tag is escaped into inert text).
![Injection Sandbox](docs/screenshots/injection-sandbox.jpg)

## Verified against a real LLM

Tested against a local `llama3` model via Ollama with an identical malicious prompt, sent two ways:

- **Unsanitized:** model leaked its system prompt verbatim.
- **Sanitized (DelimiterManager):** model refused — *"I'm unable to reveal my original system prompt."*

Same input, different structural handling, measurably different model behavior. Full transcript: [`examples/injection_test_results.txt`](examples/injection_test_results.txt).

*Scope note: this defends against structural tag-forgery injection specifically, not every possible injection technique.*

## Testing

```bash
pip install -e ".[dev]"
pytest -v
```

The tests target cases such, as oversized or malformed messages non‑string content and injection attempts. They do not focus on happy paths.
