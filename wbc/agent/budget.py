"""Budget guard: adds up what every model call costs and stops the agent before the credits run out.

Spend is saved to disk, so restarts and separate scripts share one running total.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


def _field(obj, *names, default=0):
    """Read the first present field from a dict or an SDK object (usage shapes differ by endpoint)."""
    if obj is None:
        return default
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return default


class Budget:
    def __init__(self, limit_usd: float, prices: dict, state_file: Path | None = None,
                 warn_usd: float | None = None):
        self.limit = float(limit_usd)
        self.warn = None if warn_usd is None else float(warn_usd)
        self.prices = prices
        self.path = Path(state_file) if state_file else None
        self._lock = threading.Lock()
        self.spent = self._load()
        # Already past the warning line from earlier runs: say so once, at the first chance.
        self._warning_due = self.warn is not None and self.spent >= self.warn
        self._warned = self._warning_due

    @classmethod
    def from_config(cls, models: dict, root: Path) -> "Budget":
        b = models["budget"]
        return cls(b["limit_usd"], b["prices"], root / b["state_file"], b.get("warn_usd"))

    def pop_warning(self) -> str | None:
        """The spend warning, once, after tracked spend first passes warn_usd."""
        with self._lock:
            if not self._warning_due:
                return None
            self._warning_due = False
        return (f"Spend has passed ${self.warn:.0f}: ${self.spent:.2f} of the ${self.limit:.0f} stop "
                f"(${self.remaining:.2f} left).")

    @property
    def remaining(self) -> float:
        return self.limit - self.spent

    def check(self) -> None:
        if self.spent >= self.limit:
            raise BudgetExceeded(
                f"Budget stop: tracked spend is ${self.spent:.2f}, at the ${self.limit:.2f} limit, so no "
                f"more paid calls. Check the Billing page first; raise budget.limit_usd in "
                f"config/models.yaml only if your credits allow it.")

    def add_tokens(self, model: str, usage) -> float:
        p = self._price(model)
        tokens_in = _field(usage, "input_tokens", "prompt_tokens")
        tokens_out = _field(usage, "output_tokens", "completion_tokens")
        details = _field(usage, "input_tokens_details", "prompt_tokens_details", default=None)
        cached = min(_field(details, "cached_tokens"), tokens_in)
        cost = ((tokens_in - cached) * p["in"] + cached * p.get("cached_in", p["in"]) + tokens_out * p["out"]) / 1e6
        return self._add(cost)

    def add_audio(self, model: str, seconds: float) -> float:
        return self._add(seconds / 3600 * self._price(model)["per_hour"])

    def add_images(self, model: str, count: int) -> float:
        return self._add(count / 1000 * self._price(model)["per_1k_images"])

    def _price(self, model: str) -> dict:
        try:
            return self.prices[model]
        except KeyError:
            raise KeyError(f"No price for {model!r} in budget.prices (config/models.yaml)") from None

    def _add(self, cost: float) -> float:
        with self._lock:
            self.spent += cost
            if self.warn is not None and not self._warned and self.spent >= self.warn:
                self._warned = self._warning_due = True
            self._save()
        return cost

    def _load(self) -> float:
        if self.path and self.path.exists():
            return float(json.loads(self.path.read_text())["spent_usd"])
        return 0.0

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"spent_usd": round(self.spent, 6)}))
