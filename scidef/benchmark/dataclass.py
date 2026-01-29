from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkResult:
    """Simple benchmark result."""

    metric: str
    dataset: str
    result: Optional[List[Any]] = None
    correlation: Optional[float] = None
    correlations: Optional[List[float]] = None
    success_rate: float = 0.0
    error_count: int = 0
    total_count: int = 0
    model_name: Optional[str] = None
    prompt_type: Optional[str] = None
    system_prompt: Optional[str] = None
    meta: Optional[Dict] = None
