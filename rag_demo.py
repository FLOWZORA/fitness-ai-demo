"""
app.py — FitCoach Pro AI (RAG demo) front end.

Design intent
-------------
This page has one job: let a visitor experience the moment the assistant
declines to answer, and understand why that is the valuable behaviour.

So the interface is built around a confidence scale rather than a chat
transcript. Every response renders its retrieval score against the
calibrated threshold, so "answered" and "declined" are visibly the same
mechanism producing two outcomes — not a feature and an error.

Everything the visitor does not need is removed: no mode banners, no
debug output, no library names. Operator diagnostics go to the audit
log (pipeline_logger), never to the screen.
"""
import html

import gradio as gr

import config
from rag_pipeline import build_vector_store, answer as rag_answer

# ── Build once at startup ────────────────────────────────────────────
try:
    from kb_change_detector import check_and_recalibrate
    check_and_recalibrate(verbose=False)
except Exception:
    pass  # never block the UI on maintenance tasks

try:
    from pipeline_logger import log
    log.startup()
except Exception:
    log = None

STORE = build_vector_store()
THRESHOLD = float(getattr(config, "MIN_RELEVANCE", 0.302))


# ── Example prompts ──────────────────────────────────────────────────
# The third one is expected to be declined. Saying so up front is the
# point of the demo, so it is labelled honestly rather than hidden.
EXAMPLES = [
    "How much creatine should I take daily?",
    "Is high protein safe with kidney disease?",
    "What is the stock price of Apple?",
]


# ── Rendering ────────────────────────────────────────────────────────
def _scale(score: float, declined: bool) -> str:
    """The signature element: score plotted against the threshold."""
    pos = max(0.0, min(1.0, score)) * 100
    tpos = max(0.0, min(1.0, THRESHOLD)) * 100
    state = "declined" if declined else "grounded"

    return f"""
    <div class="scale" role="img"
         aria-label="Retrieval confidence {score:.3f} against threshold {THRESHOLD:.3f}">
      <div class="scale-track">
        <div class="scale-zone-low" style="width:{tpos}%"></div>
        <div class="scale-thresh" style="left:{tpos}%"></div>
        <div class="scale-dot scale-dot--{state}" style="left:{pos}%"></div>
      </div>
      <div class="scale-legend">
        <span class="mono">0.000</span>
        <span class="scale-legend-mid mono" style="left:{tpos}%">
          threshold {THRESHOLD:.3f}
        </span>
        <span class="mono">1.000</span>
      </div>
    </div>
    """


def _sources(items) -> str:
    if not items:
        return ""
    seen, ordered = set(), []
    for s in items:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    chips = "".join(
        f'<span class="src mono">{html.escape(str(s))}</span>' for s in ordered
    )
    return f'<div class="sources"><span class="sources-label">from</span>{chips}</div>'


def _card(question: str, result: dict) -> str:
    declined = bool(result.get("below_threshold"))
    score = float(result.get("top_relevance") or 0.0)
    body = html.escape(str(result.get("answer", ""))).replace("\n", "<br>")
    state = "declined" if declined else "grounded"
    verdict = "Declined — not enough grounding" if declined else "Grounded in the library"

    return f"""
    <article class="card card--{state}">
      <p class="q">{html.escape(question)}</p>
      <div class="a">{body}</div>
      <div class="meta">
        <div class="verdict verdict--{state}">
          <span class="verdict-mark"></span>{verdict}
          <span class="verdict-score mono">{score:.3f}</span>
        </div>
        {_scale(score, declined)}
        {_sources(result.get("sources"))}
      </div>
    </article>
    """


EMPTY = """
<div class="empty">
  <p class="empty-lead">Ask something a fitness coach would know.</p>
  <p class="empty-sub">
    Answers come only from FitCoach&nbsp;Pro's coaching library, and each one
    shows the retrieval score behind it. Ask something outside the library and
    you will watch the assistant decline instead of guess.
  </p>
</div>
"""


def respond(question: str, history: str):
    q = (question or "").strip()
    if not q:
        return history or EMPTY, ""
    try:
        result = rag_answer(STORE, q)
    except Exception as exc:
        if log:
            log.error("ui_answer", exc)
        return (history or "") + (
            '<article class="card card--error"><p class="q">'
            + html.escape(q)
            + '</p><div class="a">Something failed on the retrieval side. '
              "The question was not answered — nothing was guessed.</div></article>"
        ), ""
    card = _card(q, result)
    prior = "" if (history or EMPTY) == EMPTY else (history or "")
    return card + prior, ""


# ── Styles ───────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --bg:#1C1A18; --surface:#232020; --surface-2:#2B2725;
  --rule:#3A3532; --rule-soft:#302C29;
  --ink:#EDEAE3; --ink-soft:#A8A29A; --ink-faint:#968F84;
  --sage:#6E9E78; --brass:#B8935A; --clay:#C87C65;
  --sans:'Inter',system-ui,sans-serif; --mono:'IBM Plex Mono',monospace;
}

.gradio-container{
  background:var(--bg)!important; max-width:820px!important;
  margin:0 auto!important; font-family:var(--sans)!important;
  color:var(--ink)!important;
}
.gradio-container *{font-family:var(--sans);}
.mono{font-family:var(--mono); font-variant-numeric:tabular-nums; color:inherit;}
footer, .footer, .show-api, .built-with{display:none!important;}

/* Masthead */
.mast{padding:38px 4px 22px; border-bottom:1px solid var(--rule);}
.mast-row{display:flex; align-items:baseline; justify-content:space-between; gap:16px; flex-wrap:wrap;}
.mast h1{font-size:29px; font-weight:700; letter-spacing:-.02em; margin:0; color:var(--ink);}
.mast-by{font-size:12px; color:var(--ink-faint); letter-spacing:.06em; text-transform:uppercase; white-space:nowrap;}
.mast p{margin:11px 0 0; font-size:16px; line-height:1.55; color:var(--ink-soft); max-width:60ch;}
.mast p b{color:var(--ink); font-weight:600;}

/* Prompt chips */
.chips-label{font-size:11px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--ink-faint); margin:26px 0 11px;}
.chips{display:flex; flex-wrap:wrap; gap:9px; margin-bottom:8px;}
.chips button{
  background:transparent!important; border:1px solid var(--rule)!important;
  color:var(--ink-soft)!important; border-radius:2px!important;
  padding:9px 13px!important; font-size:13px!important; font-weight:400!important;
  text-align:left!important; min-height:0!important; box-shadow:none!important;
  transition:border-color .15s, color .15s;
}
.chips button:hover{border-color:var(--ink-faint)!important; color:var(--ink)!important;}

/* Composer */
.composer{display:flex; gap:10px; align-items:stretch; margin:6px 0 30px;}
.composer .ask-in{flex:1;}
.composer textarea, .composer input[type=text]{
  background:var(--surface)!important; border:1px solid var(--rule)!important;
  border-radius:2px!important; color:var(--ink)!important;
  font-size:15px!important; padding:13px 15px!important; box-shadow:none!important;
}
.composer textarea:focus, .composer input[type=text]:focus{
  border-color:var(--sage)!important; outline:none!important;
}
.composer textarea::placeholder{color:var(--ink-faint)!important;}
.composer button{
  background:var(--sage)!important; color:#142016!important; border:none!important;
  border-radius:2px!important; font-weight:600!important; font-size:14px!important;
  padding:0 24px!important; box-shadow:none!important; white-space:nowrap;
}
.composer button:hover{background:#7DAD87!important;}

/* Empty state */
.empty{padding:34px 0 8px; border-top:1px solid var(--rule-soft);}
.empty-lead{margin:0 0 9px; font-size:17px; font-weight:600; color:var(--ink);}
.empty-sub{margin:0; font-size:14px; line-height:1.65; color:var(--ink-faint); max-width:56ch;}

/* Answer card */
.card{padding:26px 0 30px; border-top:1px solid var(--rule);}
.card .q{margin:0 0 15px; font-size:14px; color:var(--ink-faint);
  font-weight:500; letter-spacing:.01em;}
.card .q::before{content:"asked  "; color:var(--ink-faint); opacity:.6;}
.card .a{font-size:16px; line-height:1.68; color:var(--ink);}
.card--declined .a{color:var(--ink-soft);}
.card--error .a{color:var(--clay);}

.meta{margin-top:20px; padding-top:17px; border-top:1px dashed var(--rule-soft);}

/* Verdict line */
.verdict{display:flex; align-items:baseline; gap:8px; font-size:12.5px;
  letter-spacing:.02em; color:var(--ink-soft);}
.verdict-mark{width:6px; height:6px; position:relative; top:-2px; border-radius:50%; flex:none;}
.verdict--grounded .verdict-mark{background:var(--sage);}
.verdict--declined .verdict-mark{background:var(--clay);}
.verdict--grounded{color:var(--sage);}
.verdict--declined{color:var(--clay);}
.verdict-score{color:var(--ink-faint); font-size:12px; padding-left:2px;}

/* Signature: the confidence scale */
.scale{margin:19px 0 4px;}
.scale-track{position:relative; height:3px; background:var(--surface-2); border-radius:2px;}
.scale-zone-low{position:absolute; inset:0 auto 0 0; height:3px;
  background:rgba(168,101,82,.28); border-radius:2px 0 0 2px;}
.scale-thresh{position:absolute; top:-7px; width:1px; height:17px; background:var(--ink-soft);}
.scale-dot{position:absolute; top:50%; width:11px; height:11px; border-radius:50%;
  transform:translate(-50%,-50%); border:2px solid var(--bg);}
.scale-dot--grounded{background:var(--sage);}
.scale-dot--declined{background:var(--clay);}
.scale-legend{position:relative; margin-top:9px; height:14px;
  font-size:10.5px; color:var(--ink-faint);}
.scale-legend>span{color:var(--ink-faint);}
.scale-legend>span:first-child{position:absolute; left:0;}
.scale-legend>span:last-child{position:absolute; right:0;}
.scale-legend-mid{position:absolute; transform:translateX(-50%); white-space:nowrap; color:var(--ink-faint);}

/* Sources */
.sources{margin-top:15px; display:flex; align-items:center; gap:8px; flex-wrap:wrap;}
.sources-label{font-size:10.5px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--ink-faint);}
.src{font-size:12px; color:var(--ink-soft); background:var(--surface);
  border:1px solid var(--rule-soft); border-radius:2px; padding:4px 9px;}

/* Colophon */
.colophon{margin-top:40px; padding:18px 0 34px; border-top:1px solid var(--rule);
  display:flex; justify-content:space-between; gap:14px; flex-wrap:wrap;
  font-size:11.5px; color:var(--ink-faint);}
.colophon>span{color:var(--ink-faint);}
.colophon a{color:var(--ink-soft); text-decoration:none; border-bottom:1px solid var(--rule);}
.colophon a:hover{color:var(--ink);}

@media (max-width:640px){
  .mast h1{font-size:24px;} .mast p{font-size:15px;}
  .composer{flex-direction:column;} .composer button{padding:13px 0!important;}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important; animation:none!important;}}
"""


# ── Layout ───────────────────────────────────────────────────────────
with gr.Blocks(title="FitCoach Pro AI — grounded coaching answers",
               analytics_enabled=False) as demo:

    gr.HTML("""
    <header class="mast">
      <div class="mast-row">
        <h1>FitCoach&nbsp;Pro AI</h1>
        <span class="mast-by">Flowzora · RAG demo</span>
      </div>
      <p>Answers come only from FitCoach&nbsp;Pro's verified coaching library, and
      each one is shown with the retrieval score behind it. Below the calibrated
      threshold, <b>it declines instead of guessing.</b></p>
    </header>
    """)

    gr.HTML('<p class="chips-label">Try one — the last is outside the library</p>')
    with gr.Row(elem_classes="chips"):
        chip_btns = [gr.Button(q, elem_classes="chip") for q in EXAMPLES]

    with gr.Row(elem_classes="composer"):
        box = gr.Textbox(
            placeholder="Ask about training, nutrition or recovery…",
            lines=1, show_label=False, container=False, elem_classes="ask-in",
        )
        send = gr.Button("Ask", variant="primary")

    feed = gr.HTML(EMPTY)

    gr.HTML(f"""
    <div class="colophon">
      <span>Threshold <span class="mono">{THRESHOLD:.3f}</span>, calibrated against
      measured score distributions — not chosen by hand.</span>
      <span>Built by Vaibhav Tiwari · <a href="https://flowzora.github.io/intake"
        target="_blank" rel="noopener">Flowzora</a></span>
    </div>
    """)

    send.click(respond, [box, feed], [feed, box])
    box.submit(respond, [box, feed], [feed, box])
    for btn, q in zip(chip_btns, EXAMPLES):
        btn.click(lambda _f, _q=q: respond(_q, _f), [feed], [feed, box])


if __name__ == "__main__":
    demo.launch(ssr_mode=False, css=CSS)
