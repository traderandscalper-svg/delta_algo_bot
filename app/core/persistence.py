from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


class EnginePersistence:
    """Crash-safe JSON state persistence for DEMO/PAPER operation."""

    VERSION = 1

    def __init__(self, path: str = "data/state/engine_state.json") -> None:
        self.path = Path(path)
        self.logger = logging.getLogger("EnginePersistence")

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": self.VERSION, "state": state}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, self.path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
                return {}
            return payload["state"]
        except Exception:
            self.logger.exception("Unable to load persisted engine state.")
            return {}
