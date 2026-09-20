"""
Trains and evaluates the retrieval-augmented prediction model (fusion_head.py)
on real data, following proposal Sec 6 "Downstream prediction quality":

  Directional accuracy, Spearman's rho, and R^2, each with bootstrap
  confidence intervals, comparing the retrieval-augmented head against the
  non-augmented head under the existing chronological split (train prior
  to 2020, validation 2020-2022, test from 2023 onward).

READ fusion_head.py's module docstring first -- the stance label used below
(data.stance_label, real 1-year OIS rate change) is a PRAGMATIC CHOICE made
to unblock training, confirmed with the user on 2026-09-13 after checking
that neither this project nor the concurrent encoder-training submission
has a stance definition yet. It is not from the proposal. Write it up in
the paper as an explicit, justified modeling decision, not as settled fact.

HOW REAL THIS RUN IS:
  - Corpus, market data, event linking: REAL (events = EA-MPD dates that have a
    press-conference statement; count printed at run time -- see data.py).
  - Stance label: REAL market data, but an unvalidated proxy definition
    (see above) -- not confirmed against any ground truth.
  - Encoder: STILL THE STUB (similarity.py::make_stub_encoder). Text
    representations carry no real semantic meaning yet. This means the
    fusion head's text pathway (e_q) is working with noise -- ONLY the
    retrieved market vectors carry real signal right now. Re-run this
    exact script once the real trained encoder is wired in.
  - n is small (train/val/test sizes printed at run time; val and test are a
    few dozen events at most). Confidence intervals on val/test metrics will be WIDE. That is the
    honest consequence of the small n, not a bug -- do not read a single point
    estimate from this run as a stable result.

RETRIEVAL LEAKAGE BOUNDARY, enforced here (not needed in run_demo.py/
run_real.py's retrieval-QUALITY evaluation, which is a symmetric pairwise
comparison, not a sequential prediction task): for a query event dated t,
only candidate events dated STRICTLY BEFORE t are retrievable. Predicting
event t using a "precedent" that happened after t would be looking at the
future -- silently invalidating the whole exercise. Early events with no
prior history are skipped, not padded with something invented.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data import load_real_corpus, stance_label
from fusion_head import FusionHead, NonAugmentedHead
from metrics import bootstrap_metric_ci, directional_accuracy, r_squared, spearman_rho
from similarity import hybrid_similarity, make_stub_encoder

DATA_DIR = Path(__file__).parent / "data"
K = 5
LAM = 0.5
TAU = 1.0
EPOCHS = 150
LR = 1e-3
SEED = 0


def build_examples(events, encoder, embeddings, k=K, lam=LAM, tau=TAU, leaky=False, label_fn=None):
    """For each event (in date order), retrieves up to k STRICTLY EARLIER
    events. Returns a list of dicts; events with zero available history are
    skipped (there is nothing to retrieve).

    LEAKAGE BOUNDARY (proposal Sec 4): at genuine prediction time m_q is
    unobserved, so retrieval must not use the query's own market vector --
    the stance label is computed from that vector, so using it to pick
    precedents hands the model a peek at the answer. With leaky=False
    (default) m_q=None is passed, so hybrid_similarity takes its explicit
    text-only fallback (retrieval by text similarity alone; the candidates'
    KNOWN outcomes still reach the fusion head as the retrieved evidence).
    leaky=True reproduces the original, invalid setup and exists ONLY to
    quantify how much the leak inflated earlier results -- never report
    numbers from it as results.
    """
    examples = []
    for i, q in enumerate(events):
        history = events[:i]  # strictly earlier, by construction (events is date-sorted)
        if not history:
            continue
        m_q = q.market_vector if leaky else None
        scored = []
        for c in history:
            res = hybrid_similarity(
                embeddings[q.event_id], embeddings[c.event_id],
                c.market_vector, m_q, lam=lam, tau=tau,
            )
            scored.append((res.score, c))
        scored.sort(key=lambda x: -x[0])
        top = scored[:k]
        examples.append({
            "event": q,
            "e_q": embeddings[q.event_id],
            "retrieved_m": np.stack([c.market_vector for _, c in top]),
            "sim_scores": np.array([s for s, _ in top], dtype=np.float32),
            "retrieved_ids": [c.event_id for _, c in top],
            "label": (label_fn or stance_label)(q.market_vector),
        })
    return examples


def to_batch(ex, embed_dim, market_dim, k):
    """Pads/truncates one example's retrieved set to exactly k rows (early
    examples may have fewer than k available precedents) and returns
    tensors of a fixed shape. Padding rows get sim_score = -inf so softmax
    assigns them ~0 attention weight -- they contribute nothing, rather
    than silently averaging in a fabricated "precedent".
    """
    m = np.zeros((k, market_dim), dtype=np.float32)
    s = np.full((k,), -1e9, dtype=np.float32)
    n = min(len(ex["retrieved_m"]), k)
    m[:n] = ex["retrieved_m"][:n]
    s[:n] = ex["sim_scores"][:n]
    return (
        torch.tensor(ex["e_q"], dtype=torch.float32).unsqueeze(0),
        torch.tensor(m, dtype=torch.float32).unsqueeze(0),
        torch.tensor(s, dtype=torch.float32).unsqueeze(0),
    )


def gaussian_nll(s_hat, sigma_hat, target):
    return 0.5 * (((target - s_hat) / sigma_hat) ** 2 + 2 * torch.log(sigma_hat))


def train_model(model, examples, embed_dim, market_dim, k, is_fusion, label_mean, label_std, epochs=EPOCHS, lr=LR, seed=SEED,
                use_uncertainty=True, freeze_attention=False, verbose=True):
    """use_uncertainty=False trains s_hat alone with squared error (the proposal's
    'absence of the uncertainty output' ablation). freeze_attention=True freezes the
    cross-attention projections at their random init and trains only the regression
    head (one reading of the proposal's 'frozen versus fine-tuned fusion' ablation --
    encoder fine-tuning itself is not possible with the stub encoder)."""
    torch.manual_seed(seed)
    if freeze_attention and is_fusion:
        for name in ("query_proj", "key_proj", "value_proj"):
            for prm in getattr(model, name).parameters():
                prm.requires_grad_(False)
    opt = torch.optim.Adam([prm for prm in model.parameters() if prm.requires_grad], lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        order = np.random.default_rng(seed + epoch).permutation(len(examples))
        for idx in order:
            ex = examples[idx]
            target = torch.tensor([(ex["label"] - label_mean) / label_std], dtype=torch.float32)
            if is_fusion:
                e_q, m, s = to_batch(ex, embed_dim, market_dim, k)
                out = model(e_q, m, s)
            else:
                e_q = torch.tensor(ex["e_q"], dtype=torch.float32).unsqueeze(0)
                out = model(e_q)
            loss = (gaussian_nll(out.s_hat, out.sigma_hat, target) if use_uncertainty
                    else (out.s_hat - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        if verbose and (epoch % 30 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:3d}  mean NLL {total_loss / len(examples):.4f}")
    return model


def predict(model, examples, embed_dim, market_dim, k, is_fusion, label_mean, label_std):
    preds = []
    with torch.no_grad():
        for ex in examples:
            if is_fusion:
                e_q, m, s = to_batch(ex, embed_dim, market_dim, k)
                out = model(e_q, m, s)
            else:
                e_q = torch.tensor(ex["e_q"], dtype=torch.float32).unsqueeze(0)
                out = model(e_q)
            preds.append(float(out.s_hat.item()) * label_std + label_mean)
    return preds


def evaluate(name, y_true, y_pred, n):
    print(f"  {name} (n={n}):")
    rec = {"n": n}
    for key, metric_name, fn in [
        ("dir_acc", "directional accuracy", directional_accuracy),
        ("spearman", "Spearman rho", spearman_rho),
        ("r2", "R^2", r_squared),
    ]:
        if key == "spearman" and np.ptp(np.asarray(y_pred, float)) < 1e-9:
            print(f"    {metric_name:<22}     n/a  (constant predictor)")
            rec[key] = None
            continue
        point, lo, hi = bootstrap_metric_ci(y_true, y_pred, fn)
        print(f"    {metric_name:<22} {point:>7.3f}  [{lo:.3f}, {hi:.3f}]")
        rec[key] = [point, lo, hi]
    return rec


def main():
    print("=" * 72)
    print("Training run -- REAL data, PRAGMATIC stance label (see module")
    print("docstring), encoder STILL A STUB. Read the caveats before citing")
    print("any number this script prints.")
    print("=" * 72)

    corpus = load_real_corpus(
        DATA_DIR / "all_ECB_speeches (1).csv",
        DATA_DIR / "speeches_bis.csv",
        DATA_DIR / "Dataset_EA-MPD.xlsx",
    )
    events = sorted(corpus.events, key=lambda e: e.date)
    doc_text = {e.event_id: corpus.doc_by_id(e.doc_id).text for e in events}

    encoder = make_stub_encoder()
    embeddings = {eid: encoder.embed(text) for eid, text in doc_text.items()}

    examples = build_examples(events, encoder, embeddings)
    print(f"\n{len(events)} real events -> {len(examples)} usable examples "
          f"({len(events) - len(examples)} skipped: no prior history to retrieve).")

    train_ex = [e for e in examples if e["event"].date.year < 2020]
    val_ex = [e for e in examples if 2020 <= e["event"].date.year <= 2022]
    test_ex = [e for e in examples if e["event"].date.year >= 2023]
    print(f"split: train={len(train_ex)}  val={len(val_ex)}  test={len(test_ex)}")

    train_labels = np.array([e["label"] for e in train_ex])
    label_mean, label_std = float(train_labels.mean()), float(train_labels.std() + 1e-8)
    print(f"train-set stance label (OIS_1Y bp): mean={label_mean:.3f} std={label_std:.3f}")

    embed_dim, market_dim = encoder.dim, len(events[0].market_vector)

    print("\nTraining FusionHead (retrieval-augmented) ...")
    torch.manual_seed(SEED)  # init is drawn at construction: seed first (reproducibility)
    fusion = FusionHead(embed_dim=embed_dim, market_dim=market_dim)
    train_model(fusion, train_ex, embed_dim, market_dim, K, True, label_mean, label_std)

    print("\nTraining NonAugmentedHead (baseline, no retrieval) ...")
    torch.manual_seed(SEED)
    baseline = NonAugmentedHead(embed_dim=embed_dim)
    train_model(baseline, train_ex, embed_dim, market_dim, K, False, label_mean, label_std)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    for split_name, split_ex in [("val", val_ex), ("test", test_ex)]:
        if not split_ex:
            print(f"\n{split_name}: empty, skipping.")
            continue
        y_true = [e["label"] for e in split_ex]
        print(f"\n--- {split_name} ---")
        y_pred_fusion = predict(fusion, split_ex, embed_dim, market_dim, K, True, label_mean, label_std)
        evaluate("FusionHead (retrieval-augmented)", y_true, y_pred_fusion, len(split_ex))
        y_pred_base = predict(baseline, split_ex, embed_dim, market_dim, K, False, label_mean, label_std)
        evaluate("NonAugmentedHead (baseline)", y_true, y_pred_base, len(split_ex))

    print("\n" + "=" * 72)
    print("REMINDERS (do not drop these when reporting the numbers above):")
    print("1. Stance label = 1yr OIS change, a pragmatic proxy chosen today,")
    print("   not a validated construct. Frame it as a modeling choice.")
    print("2. Encoder is the stub -- text signal is noise. Re-run once real.")
    print(f"3. val n={len(val_ex)}, test n={len(test_ex)}. CIs are wide because n is small -- that")
    print("   is correct behavior, not something to explain away.")
    print("=" * 72)


if __name__ == "__main__":
    main()
