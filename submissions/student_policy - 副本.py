from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from submissions.agent import TaskAgent


class Policy:
    """Evaluation entry point for the vision/FSM/planner agent.

    The wrapper deliberately sanitizes the evaluator's `info` dictionary before
    calling the strategy: only the allowed inventory-like fields and numeric
    reward feedback are used. Hidden coordinates, map truth, object state, and
    other simulator internals are ignored here.
    """

    def __init__(self) -> None:
        self.task_id = "mathematical_logic/task_1"
        self.agent = TaskAgent(self.task_id)

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed
        if task_id is not None and task_id != self.task_id:
            self.task_id = task_id
            self.agent = TaskAgent(self.task_id)
        self.agent.reset()

    def act(self, obs: Any, info: dict[str, Any] | None = None) -> int:
        frame = _extract_frame(obs)
        inventory = _extract_inventory(obs, info)
        reward = _extract_reward(obs, info)
        if reward is not None:
            self.agent.update_reward(reward)
        return int(self.agent.step(frame, inventory))


def make_policy() -> Policy:
    return Policy()


def act(obs: Any, info: dict[str, Any] | None = None) -> int:
    return _GLOBAL_POLICY.act(obs, info)


def reset(seed: int | None = None, task_id: str | None = None) -> None:
    _GLOBAL_POLICY.reset(seed=seed, task_id=task_id)


def _extract_frame(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        if "frame" in obs:
            return np.asarray(obs["frame"])
        if "obs" in obs:
            return np.asarray(obs["obs"])
        raise KeyError("observation dict must contain 'frame' or 'obs'")
    return np.asarray(obs)


def _extract_inventory(obs: Any, info: dict[str, Any] | None) -> dict[str, Any]:
    raw: Any = None
    if isinstance(obs, dict):
        raw = obs.get("inventory", obs.get("inventory_ids"))
    if raw is None and isinstance(info, dict):
        raw = info.get("inventory", info.get("items"))
    return _normalize_inventory(raw)


def _normalize_inventory(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    result: dict[str, Any] = {}
    values = _as_iterable(raw)
    items: list[str] = []
    for value in values:
        if isinstance(value, str):
            items.append(value)
            result[value] = result.get(value, 0) + 1
        elif isinstance(value, (int, np.integer)):
            item_id = int(value)
            items.append(str(item_id))
            result[str(item_id)] = result.get(str(item_id), 0) + 1
    if items:
        result["items"] = items
    return result


def _extract_reward(obs: Any, info: dict[str, Any] | None) -> float | None:
    if isinstance(obs, dict) and isinstance(obs.get("reward"), (int, float, np.number)):
        return float(obs["reward"])
    if not isinstance(info, dict):
        return None
    for key in ("reward", "last_reward"):
        value = info.get(key)
        if isinstance(value, (int, float, np.number)):
            return float(value)
    return None


def _as_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return iter(value)
    except TypeError:
        return ()


_GLOBAL_POLICY = Policy()


__all__ = ["Policy", "make_policy", "act", "reset"]
