"""
ECB-Precedent benchmark construction (proposal Sec 5).

Relevance is derived, not annotated: two events (i, j) in E constitute a
relevant pair when ||m_i - m_j||_2 falls below a threshold calibrated
against the empirical pairwise-distance distribution (e.g. bottom decile),
yielding graded relevance suitable for rank-sensitive metrics.

"Graded relevance" here is implemented as: relevance = max(0, 1 - dist/threshold)
for pairs under the threshold, 0 otherwise. This gives a continuous score in
(0, 1] that decays with distance rather than a flat binary label -- but this
specific grading function is an implementation choice, not something stated
in the proposal. Flag it to your teammate as a decision that needs to be
either endorsed or replaced before it's used in the paper's evaluation.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from data import Event


@dataclasses.dataclass
class RelevancePair:
    query_event_id: str
    candidate_event_id: str
    distance: float
    relevance: float  # graded, in [0, 1]


def pairwise_distances(events: list[Event]) -> np.ndarray:
    m = np.stack([e.market_vector for e in events])
    diffs = m[:, None, :] - m[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def build_ecb_precedent(
    events: list[Event],
    percentile: float = 10.0,
) -> tuple[list[RelevancePair], float]:
    """Returns (relevance pairs, threshold used).

    percentile=10.0 reproduces "bottom decile" as stated in the proposal.
    """
    n = len(events)
    dist = pairwise_distances(events)
    off_diag = dist[~np.eye(n, dtype=bool)]
    threshold = float(np.percentile(off_diag, percentile))

    pairs: list[RelevancePair] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist[i, j]
            if d < threshold:
                relevance = max(0.0, 1.0 - d / threshold)
                pairs.append(
                    RelevancePair(
                        query_event_id=events[i].event_id,
                        candidate_event_id=events[j].event_id,
                        distance=float(d),
                        relevance=float(relevance),
                    )
                )
    return pairs, threshold


def relevance_lookup(pairs: list[RelevancePair]) -> dict[tuple[str, str], float]:
    return {(p.query_event_id, p.candidate_event_id): p.relevance for p in pairs}
