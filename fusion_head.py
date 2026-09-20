"""
Retrieval-augmented prediction head (proposal Sec 4 "Retrieval-augmented
prediction" + Sec 5 "Fusion head").

    Given the top-k retrieved precedents and their known outcomes
    {m_i1, ..., m_ik}, a fusion function g_phi combines e_q with an
    attention-weighted aggregation of retrieved outcomes to produce a
    stance estimate s_hat_q and an associated uncertainty sigma_hat_q.
    The parametric head remains trainable, so the model is not reducible
    to a non-parametric k-nearest-neighbor estimator; retrieval informs
    rather than replaces the prediction mechanism.

    Fusion head: Cross-attention over the k retrieved outcome vectors,
    weighted by sim(q, di), concatenated with e_q and passed through a
    regression head architecture ... evaluated both frozen and lightly
    fine-tuned.

WHAT THIS FILE IS: the trainable architecture g_phi, built exactly to that
spec, and verified to run forward on real retrieved data (see __main__).

WHAT THIS FILE IS NOT, YET: trained. Training needs a target label s_q (the
"stance" a query event is scored against) for every event, and the proposal
never says, anywhere, what s_q actually is numerically -- Appendix A calls
it "predicted stance for query q" but gives no formula. This is a REAL,
BLOCKING gap, not an oversight in this file: nothing computes gradients
against nothing. See the bottom of this docstring.

CROSS-ATTENTION MECHANISM, AS BUILT (a documented implementation choice,
since the proposal names the ingredients -- e_q, retrieved outcomes,
sim(q,di) -- but not the exact mechanism): e_q is projected to a query
vector; each retrieved market vector m_i is projected to a key and a value;
attention logits are the learned query-key compatibility PLUS the given
retrieval similarity sim(q,di) as an additive bias in log-space (so a
precedent retrieval already trusted more also gets more weight in the
fusion, rather than the fusion re-deriving relevance from scratch). Revisit
with whoever owns Sec 5 if a different mechanism was intended.

REGRESSION HEAD: "consistent with the existing per-signal head design" --
that design lives in the concurrent submission's codebase, which this
project does not have. What's here is a standard two-layer MLP producing
(mean, log-variance) -> (s_hat, sigma_hat). Swap it for the real per-signal
head design once you have it; nothing else in this file depends on its
internals.

=====================================================================
OPEN BLOCKER, put here so it can't be missed: what is s_q (the label)?
=====================================================================
Two live candidates, neither confirmed:
  (a) A market-derived scalar computed FROM each event's own market_vector
      m_i -- e.g. the sign/magnitude of the short-rate OIS surprise, or a
      principal component of it (this is what EA-MPD's original paper
      calls "Target/Timing/FG/QE factors" -- NOT the same as the raw
      columns in data.MARKET_VECTOR_COLUMNS, which are unreduced price
      changes, not factors).
  (b) A dictionary/text-based hawkish-minus-dovish score computed from the
      speech text itself (the older, human-judgment-adjacent approach the
      proposal's related-work section explicitly distances itself from).
  (c) Something the concurrent submission (the encoder-training paper)
      already defines, that this project is meant to reuse -- proposal
      Sec 6 says "reused evaluation machinery ... consistent with existing
      project specifications," implying a definition exists somewhere
      this codebase doesn't have visibility into.
DO NOT pick one of these silently. Confirm with whoever owns Sec 4, then
write the label-construction function this file currently has no
substitute for. Until then, `s_hat_q`/`sigma_hat_q` below are real
NUMBERS from a real forward pass through an UNTRAINED network -- i.e.
noise. They validate the architecture runs; they are not predictions.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import nn


@dataclasses.dataclass
class FusionOutput:
    s_hat: torch.Tensor       # (batch,) predicted stance
    sigma_hat: torch.Tensor   # (batch,) predicted uncertainty, > 0
    attn_weights: torch.Tensor  # (batch, k) for the explanation output (Sec 5)


class FusionHead(nn.Module):
    """g_phi: (e_q, retrieved market vectors, their sim scores) -> (s_hat, sigma_hat).

    embed_dim: dimension of the frozen encoder's output e_q (h in the
        proposal's notation -- 32 for the current stub encoder; must match
        whatever real encoder eventually replaces it).
    market_dim: dimension of a market-reaction vector m_i (9, matching
        data.MARKET_VECTOR_COLUMNS).
    """

    def __init__(
        self,
        embed_dim: int,
        market_dim: int,
        attn_hidden: int = 32,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.market_dim = market_dim

        self.query_proj = nn.Linear(embed_dim, attn_hidden)
        self.key_proj = nn.Linear(market_dim, attn_hidden)
        self.value_proj = nn.Linear(market_dim, attn_hidden)
        self._attn_scale = attn_hidden ** 0.5

        self.head = nn.Sequential(
            nn.Linear(embed_dim + attn_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 2),  # [mean, log_var]
        )

    def forward(
        self,
        e_q: torch.Tensor,            # (batch, embed_dim)
        retrieved_m: torch.Tensor,    # (batch, k, market_dim)
        sim_scores: torch.Tensor,     # (batch, k) -- hybrid_similarity() scores
    ) -> FusionOutput:
        q = self.query_proj(e_q).unsqueeze(1)          # (batch, 1, attn_hidden)
        k = self.key_proj(retrieved_m)                  # (batch, k, attn_hidden)
        v = self.value_proj(retrieved_m)                 # (batch, k, attn_hidden)

        learned_logits = (q * k).sum(-1) / self._attn_scale  # (batch, k)
        logits = learned_logits + sim_scores  # retrieval-biased, see docstring
        attn_weights = torch.softmax(logits, dim=-1)          # (batch, k)

        context = (attn_weights.unsqueeze(-1) * v).sum(1)    # (batch, attn_hidden)
        fused = torch.cat([e_q, context], dim=-1)
        out = self.head(fused)                                # (batch, 2)
        s_hat = out[:, 0]
        sigma_hat = torch.nn.functional.softplus(out[:, 1]) + 1e-6
        return FusionOutput(s_hat=s_hat, sigma_hat=sigma_hat, attn_weights=attn_weights)


class NonAugmentedHead(nn.Module):
    """The proposal's own required baseline (Sec 5 'Baselines'): a
    regression head with NO retrieval component, e_q -> (s_hat, sigma_hat).
    Same head architecture as FusionHead.head, for a fair comparison that
    isolates retrieval's marginal contribution rather than head capacity.
    """

    def __init__(self, embed_dim: int, head_hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, e_q: torch.Tensor) -> FusionOutput:
        out = self.head(e_q)
        s_hat = out[:, 0]
        sigma_hat = torch.nn.functional.softplus(out[:, 1]) + 1e-6
        batch = e_q.shape[0]
        return FusionOutput(
            s_hat=s_hat, sigma_hat=sigma_hat,
            attn_weights=torch.zeros(batch, 0),
        )


if __name__ == "__main__":
    # Smoke test on REAL retrieved data: proves the architecture is wired
    # correctly end-to-end (shapes, gradients flow, no NaNs) on the actual
    # 228-event real corpus. Does NOT prove the predictions mean anything
    # -- the network is untrained and there is no target to train against
    # yet (see the module docstring's OPEN BLOCKER section).
    from pathlib import Path

    from baselines import rank_candidates
    from data import load_real_corpus
    from similarity import make_stub_encoder

    DATA_DIR = Path(__file__).parent / "data"
    corpus = load_real_corpus(
        DATA_DIR / "all_ECB_speeches (1).csv",
        DATA_DIR / "speeches_bis.csv",
        DATA_DIR / "Dataset_EA-MPD.xlsx",
    )
    events = corpus.events
    encoder = make_stub_encoder()
    doc_text_by_event = {e.event_id: corpus.doc_by_id(e.doc_id).text for e in events}

    k = 5
    query_event = events[0]
    candidates = [e for e in events if e.event_id != query_event.event_id]
    ranked_ids = rank_candidates(
        query_event, candidates, "dense_text_only",
        encoder=encoder, doc_text_by_event=doc_text_by_event,
    )[:k]
    cand_by_id = {e.event_id: e for e in candidates}

    e_q = torch.tensor(
        encoder.embed(doc_text_by_event[query_event.event_id]), dtype=torch.float32
    ).unsqueeze(0)
    retrieved_m = torch.tensor(
        np.stack([cand_by_id[cid].market_vector for cid in ranked_ids]), dtype=torch.float32
    ).unsqueeze(0)
    # Recompute the sim scores for the retrieved set (rank_candidates only
    # returns ids, not scores) so the fusion head sees the same signal the
    # retriever ranked on.
    from similarity import hybrid_similarity
    sims = [
        hybrid_similarity(
            e_q.squeeze(0).numpy(),
            encoder.embed(doc_text_by_event[cid]),
            cand_by_id[cid].market_vector,
            None,  # m_q unobserved at prediction time (text-only fallback)
            lam=0.5, tau=1.0,
        ).score
        for cid in ranked_ids
    ]
    sim_scores = torch.tensor([sims], dtype=torch.float32)

    model = FusionHead(embed_dim=encoder.dim, market_dim=retrieved_m.shape[-1])
    out = model(e_q, retrieved_m, sim_scores)
    print("FusionHead smoke test (real retrieval, untrained network):")
    print("  query event:", query_event.event_id, query_event.date)
    print("  retrieved:", ranked_ids)
    print("  attn_weights:", out.attn_weights.detach().numpy().round(3))
    print("  s_hat (MEANINGLESS -- untrained, no target defined yet):",
          float(out.s_hat.item()))
    print("  sigma_hat:", float(out.sigma_hat.item()))

    loss = out.s_hat.pow(2).mean()
    loss.backward()
    has_grad = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("  gradients flow through all parameters:", has_grad)

    baseline = NonAugmentedHead(embed_dim=encoder.dim)
    base_out = baseline(e_q)
    print("  NonAugmentedHead runs too, s_hat:", float(base_out.s_hat.item()))
