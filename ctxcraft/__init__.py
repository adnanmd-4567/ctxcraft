"""CtxCraft: Lightweight Context Engineering Toolkit."""

from ctxcraft.allocator import TokenCounter, ContextBudget
from ctxcraft.compactor import Compactor

__version__ = "0.1.0"
__all__ = ["TokenCounter", "ContextBudget", "Compactor"]