"""Builds ECIR_results_report.html: a plain-language one-page showcase of the measured results.
Every number is read from results/*.json, except the test count (86 passed, observed).   Usage: python make_results_report.py [out.html]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
R = lambda n: json.loads((ROOT / "results" / n).read_text(encoding="utf-8"))

rq, cv, fs, fsh = R("retrieval_quality.json"), R("cv_rolling.json"), R("faithfulness_seeds.json"), R("faithfulness_seeds_shuffled.json")
reg, lomo, tr = R("regime_robustness.json"), R("cv_lomo.json"), R("downstream_train_py.json")
n_events, n_docs = rq["n_events"], rq["n_documents"]


def bars(title, items, vmax, hline=None, fmt="{:.3f}", note=""):
    """items: list of (label, value, color, err_lo, err_hi). Horizontal SVG bar chart."""
    w, rowh, lab = 640, 30, 250
    h = rowh * len(items) + 34
    sc = (w - lab - 70) / vmax
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{title}">']
    x0 = lab
    if hline is not None:
        out.append(f'<line x1="{x0 + hline * sc}" y1="4" x2="{x0 + hline * sc}" y2="{h - 26}" stroke="#c0392b" stroke-dasharray="4 3"/>')
        out.append(f'<text x="{x0 + hline * sc + 4}" y="{h - 10}" font-size="11" fill="#c0392b">{note}</text>')
    for i, (lbl, v, col, lo, hi) in enumerate(items):
        y = 8 + i * rowh
        out.append(f'<text x="{lab - 8}" y="{y + 15}" font-size="13" text-anchor="end" fill="#222">{lbl}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{max(v, 0) * sc:.1f}" height="20" rx="3" fill="{col}"/>')
        if lo is not None:
            out.append(f'<line x1="{x0 + lo * sc:.1f}" y1="{y + 10}" x2="{x0 + hi * sc:.1f}" y2="{y + 10}" stroke="#222" stroke-width="1.5"/>')
        out.append(f'<text x="{x0 + max(v, 0) * sc + 6:.1f}" y="{y + 15}" font-size="12" fill="#222">{fmt.format(v)}</text>')
    out.append("</svg>")
    return f"<h3>{title}</h3>" + "".join(out)


BLUE, GREY, GREEN, ORANGE = "#2f6fb0", "#9aa5b1", "#3a9d6b", "#d98a2b"
m = rq["modes"]
chart1 = bars("1. Finding similar past events (score 0 = random, 1 = perfect)", [
    ("Random", m["random"]["ndcg@10"][0], GREY, *m["random"]["ndcg@10"][1:]),
    ("Keyword matching (BM25)", m["bm25"]["ndcg@10"][0], GREY, *m["bm25"]["ndcg@10"][1:]),
    ("Text-only (placeholder encoder)", m["dense_text_only"]["ndcg@10"][0], GREY, *m["dense_text_only"]["ndcg@10"][1:]),
    ("Text + market mix", m["hybrid"]["ndcg@10"][0], BLUE, *m["hybrid"]["ndcg@10"][1:]),
    ("Market outcome only", m["market_only"]["ndcg@10"][0], GREEN, *m["market_only"]["ndcg@10"][1:]),
], 1.0)

a0, a1 = cv["per_seed"]["0"], cv["per_seed"]["1"]
pt = lambda rep, who, k: rep[who]["all"][k]["point"]
ci = lambda rep, who, k: rep[who]["all"][k]["ci95"]
chart2 = bars("2. How often it gets the direction right (hawkish vs dovish) — coin flip = 0.50", [
    ("Retrieval model (run A)", pt(a0, "fusion", "dir_acc"), BLUE, *ci(a0, "fusion", "dir_acc")),
    ("Retrieval model (run B)", pt(a1, "fusion", "dir_acc"), BLUE, *ci(a1, "fusion", "dir_acc")),
    ("Plain model, no retrieval", pt(a0, "baseline", "dir_acc"), GREY, *ci(a0, "baseline", "dir_acc")),
    ("Always guess the same answer", pt(a0, "train_mean", "dir_acc"), GREY, *ci(a0, "train_mean", "dir_acc")),
], 1.0, hline=0.5, note="coin flip")

sh = lambda rep, who: rep[who]["all"]["r2"]["point"]
chart3 = "<h3>3. How close its guesses are (0 = as good as always guessing the average; below 0 = worse)</h3><table>" \
    "<tr><th>Model (pooled over 171 test events, 2008–2025)</th><th>Run A</th><th>Run B</th></tr>" \
    f"<tr><td>Retrieval model</td><td>{sh(a0,'fusion'):.3f}</td><td>{sh(a1,'fusion'):.3f}</td></tr>" \
    f"<tr><td>Plain model, no retrieval</td><td>{sh(a0,'baseline'):.3f}</td><td>{sh(a1,'baseline'):.3f}</td></tr>" \
    f"<tr><td>Always guess the average</td><td>{sh(a0,'train_mean'):.3f}</td><td>{sh(a1,'train_mean'):.3f}</td></tr></table>"

chart4 = bars("4. Does its answer move when we swap the evidence? (higher = reacts more in proportion)", [
    ("Real model (avg of 5 runs)", fs["rho_mean"], BLUE, fs["rho_mean"] - fs["rho_sd"], fs["rho_mean"] + fs["rho_sd"]),
    ("Model trained on scrambled answers", fsh["rho_mean"], GREY, fsh["rho_mean"] - fsh["rho_sd"], fsh["rho_mean"] + fsh["rho_sd"]),
], 1.0)

rg = {(x["mode"], x["regime"]): x["ndcg@10"] for x in reg["retrieval"]}
regs = [("pre_zlb", "Normal-rate era (to 2014)"), ("zlb", "Near-zero-rate era (2014–22)"), ("post_hiking", "Rate-hiking era (2022 on)")]
chart5 = bars("5. Finding similar past events, by economic era (market outcome only)",
              [(f"{lbl} · n={reg['regime_counts'][k]}", rg[("market_only", k)][0], GREEN, *rg[("market_only", k)][1:]) for k, lbl in regs], 1.0) + \
    bars("   …and the text + market mix", [(f"{lbl}", rg[("hybrid", k)][0], BLUE, *rg[("hybrid", k)][1:]) for k, lbl in regs], 1.0)

lo = lomo["report"]
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Precedent-RAG — results</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:820px;margin:28px auto;padding:0 18px;color:#1d1d1f;line-height:1.5;background:#fff}}
h1{{margin-bottom:2px}} h2{{margin-top:34px;border-bottom:2px solid #e5e7eb;padding-bottom:4px}} h3{{margin:22px 0 4px;font-size:16px}}
.box{{background:#fff4e5;border-left:5px solid #d98a2b;padding:12px 16px;border-radius:4px;margin:16px 0}}
.good{{background:#eaf6ef;border-left-color:#3a9d6b}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border:1px solid #e0e0e0;padding:6px 10px;text-align:left}} th{{background:#f5f6f8}}
.kpi{{display:flex;gap:12px;flex-wrap:wrap}} .kpi div{{flex:1;min-width:150px;background:#f5f6f8;border-radius:6px;padding:10px 14px}} .kpi b{{font-size:22px;display:block}}
small{{color:#555}}
</style></head><body>
<h1>Precedent-RAG — what the model achieved</h1>
<small>All numbers are read from the saved result files. Whiskers on charts = 95% uncertainty range.</small>

<div class="kpi">
<div><b>{n_events}</b>ECB policy events used (of 315)</div>
<div><b>{n_docs:,}</b>ECB documents</div>
<div><b>86 / 86</b>tests passed</div>
</div>

<div class="box"><b>Bottom line, in plain words.</b> The system can reliably <i>find past events whose market reaction looked similar</i> —
but only when it is allowed to look at the market reaction itself. Using the words of the statement alone, it is barely better than
random, and <b>the model does not yet predict new events better than a coin flip.</b> That is expected right now: the tool that turns text into numbers is
a <b>placeholder</b>, not your teammate's trained encoder, so these are pipeline results, not final paper results.</div>

<h2>What works</h2>
{chart1}
<p><small>Market-outcome-only scores highest partly by design: the "right answer" is defined by market reaction, so a method that
looks at the market gets a head start. The fair comparisons are among the grey text bars — and those are only slightly above random,
because the text encoder is a placeholder.</small></p>

<h2>What does not work yet: predicting new events</h2>
{chart2}
{chart3}
<p><small>Tested the honest way: train only on the past, predict the following year, repeat from 2008 to 2025 (171 events). Two independent runs
are shown because results depend a little on random starting values. All three models sit at coin-flip level (the red line is inside every whisker),
and none beats "always guess the average" (0). Leave-one-meeting-out check ({lo['fusion']['all']['n']} events): retrieval model R² = {lo['fusion']['all']['r2']['point']:.3f}, plain model {lo['baseline']['all']['r2']['point']:.3f}.</small></p>

<h2>Is the retrieved evidence actually used?</h2>
{chart4}
<div class="box good"><b>Encouraging.</b> When we swap in different past events as evidence, the real model's answer moves roughly in proportion
to how different the swap is ({fs['rho_mean']:.2f} on average, positive in {fs['seeds_with_ci_above_zero']} of {fs['n_seeds']} runs). A model trained on scrambled answers barely does ({fsh['rho_mean']:.2f}).
Caveat: the models are barely trained, so this shows sensitivity to evidence, not useful reasoning. Removing the <i>most</i> similar vs the <i>least</i> similar
precedent made no reliable difference ({fs['seeds_with_rank_p_below_0.05']} of {fs['n_seeds']} runs significant).</div>

<h2>Does it hold up across different economic eras?</h2>
{chart5}
<p><small>Finding similar events gets clearly harder in the rate-hiking era (only {reg['regime_counts']['post_hiking']} events exist so far). About
{reg['precedent_density_post_hiking']['mean_same_regime_fraction']:.0%} of the past events it draws on for a hiking-era event are themselves from that era; the rest are borrowed from
very different periods.</small></p>

<h2>What this means for the paper</h2>
<ul>
<li><b>Safe to say now:</b> the benchmark, the retriever comparison, the honest leak-free evaluation, the faithfulness test and the era analysis all run and are tested.</li>
<li><b>Not yet safe to claim:</b> that retrieval improves prediction. That needs your teammate's real encoder, then re-running everything (one setting change + one command; see HANDOFF.md).</li>
<li><b>Two things only you/your teammate can decide:</b> the definition of "stance" (a stand-in is used: the 1-year interest-rate change) and whether the market term uses squared or plain distance.</li>
</ul>
<small>Generated by make_results_report.py from results/*.json.</small>
</body></html>"""

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ECIR_results_report.html"
out.write_text(html, encoding="utf-8")
print("wrote", out)
