from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from .types import Trace


@dataclass(slots=True)
class Unit:
    id: str
    kind: str
    session_id: str
    trace_ids: list[str]


class TemporalHierarchy:
    """Session -> episode -> window -> turn/local mapping.

    Episode segmentation is an engineering choice because the paper does not fully specify
    an implementation algorithm. Here: session boundaries + configurable time gaps.
    """

    def __init__(self, window_size: int = 4, stride: int = 2, episode_gap_minutes: int = 30):
        self.window_size = window_size
        self.stride = stride
        self.episode_gap = timedelta(minutes=episode_gap_minutes)
        self.traces: dict[str, Trace] = {}
        self.session_ids: dict[str, list[str]] = defaultdict(list)
        self.episodes: dict[str, Unit] = {}
        self.windows: dict[str, Unit] = {}
        self.trace_to_episode: dict[str, str] = {}
        self.trace_to_windows: dict[str, list[str]] = defaultdict(list)

    def add_trace(self, trace: Trace) -> None:
        self.traces[trace.id] = trace
        ids = self.session_ids[trace.session_id]
        ids.append(trace.id)
        ids.sort(key=lambda tid: self.traces[tid].turn_index)
        self._rebuild_session(trace.session_id)

    def _rebuild_session(self, session_id: str) -> None:
        ids = self.session_ids[session_id]
        for key in [k for k, u in self.episodes.items() if u.session_id == session_id]:
            del self.episodes[key]
        for key in [k for k, u in self.windows.items() if u.session_id == session_id]:
            del self.windows[key]
        groups: list[list[str]] = []
        cur: list[str] = []
        prev: Trace | None = None
        for tid in ids:
            tr = self.traces[tid]
            split = bool(prev and tr.created_at and prev.created_at and tr.created_at - prev.created_at > self.episode_gap)
            if split and cur:
                groups.append(cur)
                cur = []
            cur.append(tid)
            prev = tr
        if cur:
            groups.append(cur)
        for ei, group in enumerate(groups):
            eid = f"episode::{session_id}::{ei}"
            self.episodes[eid] = Unit(eid, "episode", session_id, group)
            for tid in group:
                self.trace_to_episode[tid] = eid
            if len(group) <= self.window_size:
                starts = [0]
            else:
                starts = list(range(0, len(group) - self.window_size + 1, self.stride))
                if starts[-1] + self.window_size < len(group):
                    starts.append(len(group) - self.window_size)
            for wi, start in enumerate(starts):
                members = group[start:start + self.window_size]
                wid = f"window::{session_id}::{ei}::{wi}"
                self.windows[wid] = Unit(wid, "window", session_id, members)
                for tid in members:
                    self.trace_to_windows[tid].append(wid)

    def neighbors(self, trace_id: str, radius: int = 1) -> list[str]:
        tr = self.traces[trace_id]
        ids = self.session_ids[tr.session_id]
        pos = ids.index(trace_id)
        lo = max(0, pos - radius)
        hi = min(len(ids), pos + radius + 1)
        return [tid for tid in ids[lo:hi] if tid != trace_id]
