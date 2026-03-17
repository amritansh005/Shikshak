"""
In-process metrics for the TTS service.

Tracks:
  - Request counts (total, cached, errors)
  - Latency histogram (p50, p95, p99)
  - GPU / CPU memory (if available)
  - Queue depth (concurrent synthesis requests)

No external dependencies — uses pure Python.
Exposed via /metrics endpoint as JSON.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict


class TTSMetrics:
    def __init__(self, window: int = 500) -> None:
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._errors: int = 0
        self._active_requests: int = 0
        self._latencies: Deque[float] = deque(maxlen=window)
        self._started_at: float = time.time()

    def record_request_start(self) -> None:
        with self._lock:
            self._total_requests += 1
            self._active_requests += 1

    def record_request_end(self, latency_ms: float, *, cache_hit: bool = False, error: bool = False) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._latencies.append(latency_ms)
            if cache_hit:
                self._cache_hits += 1
            if error:
                self._errors += 1

    def snapshot(self) -> Dict:
        with self._lock:
            lats = sorted(self._latencies)
            n = len(lats)

            def _pct(p: float) -> float:
                if not lats:
                    return 0.0
                idx = int(p * n)
                return round(lats[min(idx, n - 1)], 1)

            gpu_info = _gpu_memory()

            return {
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "total_requests": self._total_requests,
                "cache_hits": self._cache_hits,
                "cache_hit_rate": round(self._cache_hits / max(self._total_requests, 1), 3),
                "errors": self._errors,
                "active_requests": self._active_requests,
                "latency_ms": {
                    "p50": _pct(0.50),
                    "p95": _pct(0.95),
                    "p99": _pct(0.99),
                    "samples": n,
                },
                "gpu": gpu_info,
            }


def _gpu_memory() -> Dict:
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            return {
                "allocated_gb": round(alloc, 2),
                "reserved_gb": round(reserved, 2),
                "device": torch.cuda.get_device_name(0),
            }
    except Exception:
        pass
    return {}
