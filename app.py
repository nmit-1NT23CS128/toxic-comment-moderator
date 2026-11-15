%%writefile app.py
import os, re, html, threading, queue, time, zlib
from itertools import islice
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "MANYA12345/toxic-comment-distilbert"

try:   # YouTube look is a light theme; make sure Streamlit does not start in dark mode
    from streamlit import config as _cfg
    for _k, _v in {"theme.base": "light", "theme.primaryColor": "#065fd4", "theme.backgroundColor": "#ffffff",
                   "theme.secondaryBackgroundColor": "#f2f2f2", "theme.textColor": "#0f0f0f"}.items():
        _cfg.set_option(_k, _v)
except Exception:
    pass

st.set_page_config(page_title="Comment Studio", page_icon="▶️", layout="wide")

# ---------- Look & feel ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
:root{color-scheme:light;--ink:#0f0f0f;--mute:#606060;--bg:#fff;--soft:#f2f2f2;--line:#e5e5e5;
--hide:#cc0000;--review:#b45f06;--ok:#2e7d32;--accent:#065fd4;--yt:#ff0000;--card:#fff;}
html,body,.stApp,[class*="css"]{font-family:Roboto,Arial,sans-serif;}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#fff!important;color:var(--ink);}
header[data-testid="stHeader"]{background:transparent!important;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:3.4rem!important;max-width:1300px;}

/* text */
.stApp p,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,
[data-testid="stWidgetLabel"] *,[data-testid="stMarkdownContainer"] *{color:var(--ink);}
[data-testid="stCaptionContainer"] *{color:var(--mute)!important;}
h1{font-size:1.6rem!important;font-weight:500!important;}
h5{font-size:1.05rem!important;font-weight:500!important;margin-bottom:0!important;}

/* top bar */
.ytbar{height:52px;background:#fff;border-bottom:1px solid var(--line);margin-bottom:14px;
display:flex;align-items:center;justify-content:space-between;padding:0 4px;}
.brand{display:flex;align-items:center;gap:8px;}
.play{width:30px;height:21px;background:var(--yt);border-radius:6px;position:relative;display:inline-block;}
.play:after{content:"";position:absolute;left:12px;top:5.5px;border-left:9px solid #fff;
border-top:5px solid transparent;border-bottom:5px solid transparent;}
.brand .name{font-size:1.25rem;font-weight:700;letter-spacing:-.04em;color:var(--ink);}
.brand .sub{font-size:.8rem;color:var(--mute);margin-left:2px;}
.ytbar .right{display:flex;align-items:center;gap:14px;}
.status{font-size:.78rem;font-weight:500;border-radius:14px;padding:4px 12px;background:var(--soft);color:var(--mute);}
.status.on{background:#fde8e8;color:var(--hide);}
.me{width:32px;height:32px;border-radius:50%;background:#7c3aed;color:#fff;display:flex;align-items:center;
justify-content:center;font-weight:500;font-size:.9rem;}

/* buttons: grey pills, blue for the main action, black for the selected chip */
.stButton>button,.stDownloadButton>button{border-radius:18px;font-weight:500;font-size:14px;padding:.3rem 1rem;
border:none;background:var(--soft);color:var(--ink);white-space:nowrap;box-shadow:none;}
.stButton>button *,.stDownloadButton>button *{color:inherit!important;}
.stButton>button:hover{background:#e3e3e3;color:var(--ink);}
.stButton>button:disabled{background:transparent!important;border:1px solid #cfcfcf!important;color:#8a8a8a!important;opacity:1;cursor:not-allowed;}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"],
.stDownloadButton>button{background:var(--accent);color:#fff!important;}
.stButton>button[kind="primary"]:hover,.stButton>button[data-testid="stBaseButton-primary"]:hover,
.stDownloadButton>button:hover{background:#0556bf;color:#fff!important;}
[class*="st-key-nav"] .stButton>button,[class*="st-key-filter"] .stButton>button{border-radius:8px;padding:.3rem .6rem;}
[class*="st-key-nav"] .stButton>button[kind="primary"],[class*="st-key-nav"] .stButton>button[data-testid="stBaseButton-primary"],
[class*="st-key-filter"] .stButton>button[kind="primary"],[class*="st-key-filter"] .stButton>button[data-testid="stBaseButton-primary"]{
background:var(--ink);color:#fff!important;}

/* inputs */
[data-testid$="RootElement"]{background:#fff!important;border:1px solid #cfcfcf!important;border-radius:20px!important;}
[data-testid="stTextAreaRootElement"]{border-radius:12px!important;}
[data-testid$="RootElement"]:focus-within{border-color:var(--accent)!important;}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="select"]>div{background:#fff!important;
border-color:#cfcfcf!important;border-radius:20px!important;}
[data-baseweb="textarea"]{background:#fff!important;border-color:#cfcfcf!important;border-radius:12px!important;}
.stTextInput input,.stTextArea textarea,.stSelectbox input{background:#fff!important;color:var(--ink)!important;
-webkit-text-fill-color:var(--ink)!important;}
input::placeholder,textarea::placeholder{color:#8a8a8a!important;-webkit-text-fill-color:#8a8a8a!important;opacity:1!important;}
[data-baseweb="input"] button,[data-baseweb="base-input"] button{background:#fff!important;}
[data-baseweb="input"] button svg{fill:var(--mute)!important;color:var(--mute)!important;}
[data-baseweb="input"]:focus-within,[data-baseweb="textarea"]:focus-within{border-color:var(--accent)!important;}
[data-baseweb="checkbox"] > span:first-child{background:#fff!important;border:2px solid #606060!important;}
[data-baseweb="checkbox"] input:checked + span{background:var(--accent)!important;border-color:var(--accent)!important;}
[data-testid="stSlider"] [role="slider"]{background:var(--yt)!important;}
[data-testid="stSlider"] [data-testid="stTickBar"] *,[data-testid="stSliderThumbValue"]{color:var(--mute)!important;}
[data-testid="stChatInput"]{background:#fff!important;border:1px solid #cfcfcf!important;border-radius:22px;}
[data-testid="stChatInput"] > div,[data-testid="stChatInput"] textarea{background:#fff!important;color:var(--ink)!important;
-webkit-text-fill-color:var(--ink)!important;}
[data-testid="stChatInput"] button{background:var(--accent)!important;border-radius:50%;}
[data-testid="stChatInput"] button svg{fill:#fff!important;color:#fff!important;}
[data-testid="stBottom"],[data-testid="stBottomBlockContainer"]{background:#fff!important;}
[data-testid="stExpander"] details{background:#fff;border:1px solid var(--line);border-radius:12px;}
[data-testid="stExpander"] summary{background:#fff!important;}
[data-testid="stFileUploaderDropzone"]{background:#fff;border:1px dashed #cfcfcf;}
section[data-testid="stSidebar"]{background:#f9f9f9!important;border-right:1px solid var(--line);}
div[data-testid="stMetric"]{background:#f9f9f9;border:1px solid var(--line);border-radius:12px;padding:14px 18px;}
[data-testid="stMetricValue"],[data-testid="stMetricLabel"] *{color:var(--ink)!important;}

/* comments, the way YouTube draws them */
.chead{display:flex;align-items:center;gap:28px;margin:18px 0 10px;}
.chead .cnt{font-size:1.25rem;font-weight:700;color:var(--ink);}
.chead .sort{font-size:.88rem;font-weight:500;color:var(--ink);}
.yt{display:flex;gap:16px;padding:12px 10px;border-radius:12px;margin-bottom:2px;}
.yt.hid{background:#fef1f1;}.yt.rev{background:#fff7e8;}
.av{width:40px;height:40px;border-radius:50%;color:#fff!important;display:flex;align-items:center;
justify-content:center;font-weight:500;font-size:1.05rem;flex:none;}
.yb{flex:1;min-width:0;line-height:1.45;word-wrap:break-word;}
.yb .hd{display:flex;align-items:center;flex-wrap:wrap;gap:2px;font-size:.8rem;}
.yb .hd b{font-weight:500;color:var(--ink)!important;font-size:.82rem;overflow-wrap:anywhere;}
.yb .hd .sc{color:var(--mute)!important;margin-left:8px;}
.yb .hd .chip{margin:0 0 0 8px;padding:1px 9px;font-size:.72rem;}
.yb .tx{font-size:.95rem;color:var(--ink);margin-top:2px;}
.yb .blur{filter:blur(5px);transition:.2s;user-select:none;}.yb .blur:hover{filter:none;}
.chip{display:inline-block;border-radius:14px;padding:3px 12px;margin-right:6px;font-size:.8rem;font-weight:500;
background:color-mix(in srgb,var(--c) 12%,#fff);color:var(--c)!important;}
.empty{background:#f9f9f9;border-radius:12px;padding:110px 20px;text-align:center;aspect-ratio:16/9;
display:flex;flex-direction:column;align-items:center;justify-content:center;}
.empty h3{font-weight:500;margin:10px 0 4px;}.empty p{color:var(--mute)!important;margin:0;}

/* check-a-comment + queue cards */
.verdict{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--c);
border-radius:12px;padding:18px 22px;margin:14px 0;}
.verdict b{font-size:1.5rem;color:var(--c)!important;}
.verdict span{color:var(--mute)!important;display:block;margin-top:2px;}
.meter{height:10px;border-radius:6px;background:linear-gradient(90deg,var(--ok),var(--review) 50%,var(--hide));position:relative;margin:14px 0 4px;}
.meter i{position:absolute;top:-5px;width:4px;height:20px;background:var(--ink);border-radius:2px;}
.text-box{background:#f9f9f9;border-radius:12px;padding:16px 20px;line-height:1.9;font-size:1.05rem;}
.text-box mark{background:rgba(204,0,0,var(--a));color:inherit;padding:2px 3px;border-radius:4px;}
.cmt{background:#f9f9f9;border-left:5px solid var(--c);border-radius:10px;padding:10px 14px;margin:10px 0 4px;}
.cmt small{color:var(--mute)!important;}

/* ---- v6 polish ---- */
[data-testid="stTextAreaRootElement"],[data-baseweb="textarea"]{border:1px solid #cfcfcf!important;border-radius:12px!important;overflow:hidden;}
.stTextArea textarea{border:none!important;}
.stButton>button[kind="primary"]:disabled,.stButton>button[data-testid="stBaseButton-primary"]:disabled{
background:#f2f2f2!important;border:none!important;color:#9a9a9a!important;}
::-webkit-scrollbar{width:10px;height:10px;}::-webkit-scrollbar-thumb{background:#d4d4d4;border-radius:8px;border:2px solid #fff;}
.ytbar .mid{flex:1;max-width:520px;margin:0 24px;height:36px;border:1px solid #d9d9d9;border-radius:20px;display:flex;
align-items:center;padding:0 16px;font-size:.85rem;color:var(--mute);background:#fafafa;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
.ytbar .mid b{color:var(--ink);font-weight:500;margin-left:6px;}
.ph{margin:6px 0 16px;}.ph h1{font-size:1.6rem;font-weight:700;margin:0;letter-spacing:-.02em;}
.ph p{color:var(--mute)!important;margin:2px 0 0;font-size:.95rem;}
[data-testid="stVideo"],[data-testid="stVideo"] iframe,[data-testid="stVideo"] video{border-radius:14px;overflow:hidden;}
.stButton>button{transition:background .15s,transform .05s;}.stButton>button:active{transform:scale(.97);}
.yt{transition:background .15s;}
.yt:not(.hid):not(.rev):hover{background:#f7f7f7;}
.yb .hd .sc{margin-left:auto;display:flex;align-items:center;gap:7px;color:var(--mute)!important;font-size:.76rem;}
.yb .hd .sc em{font-style:normal;font-variant-numeric:tabular-nums;}
.tm{display:inline-block;width:46px;height:5px;border-radius:3px;background:#e6e6e6;overflow:hidden;}
.tm i{display:block;height:100%;border-radius:3px;}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0 12px;}
.tile{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px 6px;text-align:center;}
.tile b{display:block;font-size:1.25rem;font-weight:700;color:var(--c)!important;line-height:1.2;}
.tile span{font-size:.7rem;color:var(--mute)!important;text-transform:uppercase;letter-spacing:.04em;}
.st-key-modpanel{background:#f9f9f9;border:1px solid #eee;border-radius:18px;padding:16px 16px 10px;}
[class*="st-key-act_"] .stButton>button{height:32px;min-height:32px;font-size:13px;padding:0 .5rem;background:#fff;border:1px solid #d9d9d9;}
[class*="st-key-act_"] [data-testid="stColumn"]:first-child .stButton>button:not(:disabled){color:var(--hide);}
[class*="st-key-act_"] [data-testid="stColumn"]:last-child .stButton>button:not(:disabled){color:var(--ok);}
[class*="st-key-act_"] .stButton>button:not(:disabled):hover{background:#f2f2f2;}
.empty{background:linear-gradient(180deg,#fafafa,#f2f2f2);}
.empty .steps{display:flex;gap:10px;margin-top:18px;flex-wrap:wrap;justify-content:center;}
.empty .steps span{background:#fff;border:1px solid var(--line);border-radius:18px;padding:5px 14px;font-size:.82rem;color:var(--ink);}
.empty p{max-width:420px;}
.yt.hid{border:1px solid #f6d5d5;}.yt.rev{border:1px solid #f5e1bd;}
</style>
""", unsafe_allow_html=True)

# ---------- Small UI helpers ----------
def keyed(name):
    try:
        return st.container(key=name)
    except TypeError:               # older Streamlit without container keys
        return st.container()

def page_head(title, sub):
    st.markdown(f'<div class="ph"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)

def chip_row(options, key, box):
    """A row of YouTube-style chips. The selected chip is black."""
    cur = st.session_state.setdefault(key, options[0])
    with keyed(box):
        cols = st.columns([len(o) + 5 for o in options] + [14])
        for col, o in zip(cols, options):
            if col.button(o, key=f"{key}__{o}", type="primary" if o == cur else "secondary", use_container_width=True):
                st.session_state[key] = o
                st.rerun()                      # repaint so the new chip turns black
    return cur

# ---------- Model ----------
@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    if torch.cuda.is_available():
        mdl = mdl.cuda()
    return mdl, tok

model, tokenizer = load_model()
DEV = next(model.parameters()).device

def clean(text):
    """Light cleanup only, so inference matches how the model was trained."""
    text = re.sub(r"https?://\S+", " ", str(text))
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)      # stuuuupid -> stuupid
    return re.sub(r"\s+", " ", text).strip()

@torch.no_grad()
def predict_batch(texts, bs=32):
    out = []
    texts = [clean(t) or "." for t in texts]
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i+bs], return_tensors="pt", truncation=True,
                        padding=True, max_length=128).to(DEV)
        out += F.softmax(model(**enc).logits, dim=1)[:, 1].cpu().tolist()
    return np.array(out)

# ---------- Thresholds & decisions ----------
_live = bool(st.session_state.get("vid"))
st.markdown(f'<div class="ytbar"><div class="brand"><span class="play"></span><span class="name">Comment Studio</span>'
            f'<span class="sub">Creator</span></div>'
            f'<div class="mid">{"Moderating<b>youtu.be/" + str(st.session_state.get("vid")) + "</b>" if _live else "Paste a video link below to begin"}</div>'
            f'<div class="right">'
            f'<span class="status {"on" if _live else ""}">{"● Moderating" if _live else "Idle"}</span>'
            f'<span class="me">C</span></div></div>', unsafe_allow_html=True)
page = chip_row(["Comments", "Review queue", "Check a comment"], "page", "nav")

st.sidebar.markdown("### Filter strictness")
st.sidebar.caption("Above the first slider a comment is hidden. Above the second it waits for your review. Build v6")
hide_t = st.sidebar.slider("Hide automatically above", 0.50, 0.99, 0.75, 0.01)
review_t = st.sidebar.slider("Send to review above", 0.05, hide_t - 0.01, min(0.35, hide_t - 0.01), 0.01)

def decide(p):
    if p >= hide_t:   return "Hidden", "var(--hide)"
    if p >= review_t: return "Needs review", "var(--review)"
    return "Published", "var(--ok)"

if "queue" not in st.session_state:
    st.session_state.queue = []

def enqueue(rows):
    seen = {q["comment"] for q in st.session_state.queue}
    new = [r for r in rows if r["comment"] not in seen]
    st.session_state.queue += [{**r, "decision": "Pending"} for r in new]
    return len(new)

if "overrides" not in st.session_state:
    st.session_state.overrides = {}          # comment id -> "Hide" | "Approve"  (the creator's own calls)

def status(c):
    """What the viewer sees: the creator's call wins over the model's."""
    o = st.session_state.overrides.get(c.get("id"))
    if o == "Hide":    return "Hidden"
    if o == "Approve": return "Published"
    return decide(c["score"])[0]

def set_decision(c, action):
    cid = c.get("id")
    if cid is not None:
        st.session_state.overrides[cid] = action
    for r in st.session_state.queue:
        if (cid is not None and r.get("id") == cid) or r is c:
            r["decision"] = "Hidden" if action == "Hide" else "Approved"

COLOR = {"Hidden": "var(--hide)", "Needs review": "var(--review)", "Published": "var(--ok)"}

def explain(text, base):
    """Occlusion: drop each word, see how much the score falls."""
    words = text.split()[:40]
    if len(words) < 2:
        return [(w, base) for w in words]
    variants = [" ".join(words[:i] + words[i+1:]) for i in range(len(words))]
    drops = base - predict_batch(variants)
    scale = max(drops.max(), 0.05)
    return [(w, float(np.clip(d / scale, 0, 1))) for w, d in zip(words, drops)]

def card(row):
    label = status(row); color = COLOR[label]
    st.markdown(f'<div class="cmt" style="--c:{color}">{html.escape(row["comment"][:400])}<br>'
                f'<small>{label} · score {row["score"]:.2f}</small></div>', unsafe_allow_html=True)

COLS = ["#E11D48", "#2F5BEA", "#2E8B57", "#C77D00", "#7C3AED", "#0891B2"]

def row(c, reveal=False):
    label = status(c)
    o = st.session_state.overrides.get(c.get("id"))
    au = html.escape(str(c["author"]).lstrip("@")); txt = html.escape(c["comment"][:400])
    pill, cls = "", ""
    if label == "Hidden":
        who = "Hidden by you" if o == "Hide" else "Hidden automatically"
        pill, cls = f'<span class="chip" style="--c:var(--hide)">{who}</span>', "hid"
        txt = txt if reveal else f'<span class="blur">{txt}</span>'
    elif label == "Needs review":
        pill, cls = '<span class="chip" style="--c:var(--review)">Held for review</span>', "rev"
    elif o == "Approve":
        pill = '<span class="chip" style="--c:var(--ok)">Approved by you</span>'
    col = COLS[sum(map(ord, au)) % 6]
    pct = max(3, int(c["score"] * 100))
    meter = f'<span class="tm"><i style="width:{pct}%;background:{COLOR[label]}"></i></span>'
    return (f'<div class="yt {cls}"><div class="av" style="background:{col}">{au[:1].upper() or "?"}</div>'
            f'<div class="yb"><div class="hd"><b>@{au}</b>{pill}<span class="sc">{meter}<em>{c["score"]:.2f}</em></span></div>'
            f'<div class="tx">{txt}</div></div></div>')

# ---------- Page 1 ----------
if False:
    pass
elif page == "Check a comment":
    page_head("Check a comment", "Paste a comment to see the decision and which words caused it.")
    text = st.text_area("Comment", height=130, placeholder="Type or paste a comment…")
    if st.button("Check comment", type="primary", disabled=not text.strip()):
        p = float(predict_batch([text])[0])
        label, color = decide(p)
        st.markdown(f'<div class="verdict" style="--c:{color}"><b>{label}</b>'
                    f'<span>Toxicity score {p:.2f}</span>'
                    f'<div class="meter"><i style="left:calc({p*100:.0f}% - 2px)"></i></div></div>',
                    unsafe_allow_html=True)
        st.markdown("##### Words that drove the score")
        parts = [f'<mark style="--a:{a*0.75:.2f}">{html.escape(w)}</mark>' if a > 0.15 else html.escape(w)
                 for w, a in explain(text, p)]
        st.markdown(f'<div class="text-box">{" ".join(parts)}</div>', unsafe_allow_html=True)
        if label != "Published":
            n = enqueue([{"comment": text, "score": p, "source": "manual"}])
            st.caption("Added to the review queue." if n else "Already in the review queue.")

# ---------- Live stream page ----------
elif page == "Comments":
    S = st.session_state
    S.setdefault("chat", []); S.setdefault("seen", set()); S.setdefault("vid", None)
    S.setdefault("chat_id", None); S.setdefault("mode", None); S.setdefault("loaded", False); S.setdefault("limit", 200); S.setdefault("q", None); S.setdefault("tok", None); S.setdefault("yt", None); S.setdefault("ctok", None); S.setdefault("note", "")
    frag = getattr(st, "fragment", None)

    left, right = st.columns([5, 3], gap="large")

    def add_many(items, older=False):
        new, ids = [], set()
        for au, tx, cid in items:
            if cid not in S.seen and cid not in ids and str(tx).strip():
                ids.add(cid); new.append((au, tx, cid))
        if not new: return
        S.seen |= ids
        scores = predict_batch([t for _, t, _ in new])
        rows = [{"author": au, "comment": tx, "score": float(p), "source": "live", "id": cid}
                for (au, tx, cid), p in zip(new, scores)]
        S.chat = rows + S.chat if older else S.chat + rows
        enqueue([r for r in rows if r["score"] >= review_t])

    def add(author, text, cid):
        add_many([(author, text, cid)])

    def start_chat_thread(link):
        q = queue.Queue()
        def run():
            try:
                from chat_downloader import ChatDownloader
                for m in ChatDownloader().get_chat(link):
                    q.put((m.get("author", {}).get("name", "viewer"), m.get("message", ""),
                           str(m.get("message_id") or time.time())))
            except Exception as e:
                q.put(("system", f"Live chat stopped: {e}", f"err-{time.time()}"))
        threading.Thread(target=run, daemon=True).start()
        return q

    def fetch_free(n):
        """Read up to n newest comments; keep whatever arrived even if YouTube cuts the stream short."""
        from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
        got = []
        try:
            for c in islice(YoutubeCommentDownloader().get_comments_from_url(
                    f"https://www.youtube.com/watch?v={S.vid}", sort_by=SORT_BY_RECENT), n):
                got.append((c["author"], c["text"], c["cid"]))
        except Exception as e:
            S.note = f"YouTube stopped sending comments after {len(got)} ({e})."
        return got

    def fetch_api_pages(want, tok):
        items = []
        while len(items) < want:
            r = S.yt.commentThreads().list(part="snippet", videoId=S.vid, order="time", pageToken=tok,
                                           maxResults=min(100, want - len(items)), textFormat="plainText").execute()
            items += [(i["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"],
                       i["snippet"]["topLevelComment"]["snippet"]["textDisplay"], i["id"]) for i in r.get("items", [])]
            tok = r.get("nextPageToken")
            if not tok: break
        return items, tok

    def load_older(n=300):
        try:
            if S.mode == "api" and not S.chat_id:
                items, S.ctok = fetch_api_pages(n, S.ctok)
                add_many(items[::-1], older=True)
            elif S.mode == "free":
                S.limit += n
                add_many(fetch_free(S.limit)[::-1], older=True)   # seen-ids skip the ones we already have
        except Exception as e:
            st.warning(f"Could not load older comments: {e}")

    def poll():
        first = not S.loaded
        try:
            if S.mode == "api" and S.chat_id:
                r = S.yt.liveChatMessages().list(liveChatId=S.chat_id, part="snippet,authorDetails",
                                                 pageToken=S.tok).execute()
                S.tok = r.get("nextPageToken")
                add_many([(it["authorDetails"]["displayName"], it["snippet"].get("displayMessage", ""), it["id"])
                          for it in r.get("items", [])])
            elif S.mode == "api":
                items, tok = fetch_api_pages(S.limit if first else 20, None)
                if first: S.ctok = tok
                add_many(items[::-1])
            elif S.mode == "free":
                got = fetch_free(S.limit if first else 30)
                if first:
                    S.note = S.note or (f"Asked for {S.limit}, YouTube returned {len(got)}."
                                        if len(got) < S.limit else "")
                add_many(got[::-1])
            elif S.mode == "chat":
                items = []
                while not S.q.empty():
                    items.append(S.q.get())
                add_many(items)
            S.loaded = True
        except Exception as e:
            st.warning(f"Could not fetch comments: {e}")

    with left:
        u1, u2 = st.columns([6, 1])
        url = u1.text_input("Video link", key="opt_url", label_visibility="collapsed",
                            placeholder="Paste a YouTube video link…")
        go = u2.button("Start", type="primary", use_container_width=True, disabled=not url)
        with st.expander("Options: API key, how many comments, live stream"):
            key = st.text_input("YouTube API key (optional)", type="password", key="opt_key",
                                value=os.environ.get("YOUTUBE_API_KEY", ""),
                                help="Without a key the video still plays and comments are read without the API.")
            limit = st.slider("Comments to check when starting", 20, 5000, 500, 20, key="opt_limit")
            live_cb = st.checkbox("This video is a live stream right now", key="opt_live")
        if go:
            m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})", url)
            if not m:
                st.error("That link has no video ID. Paste the full video link.")
            else:
                S.update(vid=m.group(1), chat=[], seen=set(), tok=None, chat_id=None, yt=None,
                         mode=None, q=None, loaded=False, limit=limit)
                if key:
                    try:
                        from googleapiclient.discovery import build
                        S.yt = build("youtube", "v3", developerKey=key)
                        v = S.yt.videos().list(part="liveStreamingDetails", id=S.vid).execute()
                        det = (v.get("items") or [{}])[0].get("liveStreamingDetails", {})
                        S.chat_id = det.get("activeLiveChatId"); S.mode = "api"
                    except Exception:
                        S.yt = None
                if S.mode is None:
                    S.mode = "chat" if live_cb else "free"
                    if live_cb: S.q = start_chat_thread(url)
                st.rerun()
        if S.vid and S.mode:
            st.caption({"api": "Using the YouTube API.", "free": "No API key: checking for new comments every 15 seconds.",
                        "chat": "No API key: reading the live chat as messages are sent."}[S.mode])

        if S.vid:
            st.video(f"https://www.youtube.com/watch?v={S.vid}")
        else:
            st.markdown('<div class="empty"><div class="play" style="width:64px;height:45px;border-radius:13px"></div>'
                        '<h3>Paste a YouTube link to begin</h3>'
                        '<p>Comments are scored as they arrive. Toxic ones are hidden, borderline ones wait for you.</p>'
                        '<div class="steps"><span>1 · Paste the link</span><span>2 · Press Start</span>'
                        '<span>3 · Hide or approve</span></div></div>', unsafe_allow_html=True)
        def comments_view():
            st.markdown(f'<div class="chead"><span class="cnt">{len(S.chat):,} Comments</span>'
                        f'<span class="sort">⇅ Newest first</span></div>', unsafe_allow_html=True)
            if S.note: st.caption(S.note)
            show = chip_row(["All", "Hidden", "Held for review", "Clean"], "filter", "filter")
            want = {"All": None, "Hidden": "Hidden", "Held for review": "Needs review", "Clean": "Published"}[show]
            rows = [c for c in reversed(S.chat) if want is None or status(c) == want]
            with st.container(height=640, border=False):
                if rows:
                    st.markdown("".join(row(c) for c in rows[:1000]), unsafe_allow_html=True)
                else:
                    st.caption("No comments in this view yet.")
            if S.vid and S.mode in ("api", "free") and not S.chat_id:
                if st.button("Load 300 older comments", key="older"):
                    with st.spinner("Fetching and scoring older comments…"):
                        load_older(300)
                    st.rerun()

    def mod_buttons(c, tag):
        """Creator controls under a comment."""
        label = status(c)
        with keyed(f"act_{tag}_{zlib.crc32(str(c['id']).encode())}"):
            a, b = st.columns(2)
            if a.button("🚫 Hide", key=f"{tag}h_{c['id']}", disabled=(label == "Hidden"), use_container_width=True):
                set_decision(c, "Hide"); st.rerun()
            if b.button("✓ Approve", key=f"{tag}a_{c['id']}", disabled=(label == "Published"), use_container_width=True):
                set_decision(c, "Approve"); st.rerun()

    def feed_view():
        if S.vid and S.mode:
            poll()
        labs = [status(c) for c in S.chat]
        n, hid, rev = len(labs), labs.count("Hidden"), labs.count("Needs review")
        st.markdown('<div class="tiles">'
                    f'<div class="tile" style="--c:var(--accent)"><b>{n}</b><span>Checked</span></div>'
                    f'<div class="tile" style="--c:var(--hide)"><b>{hid}</b><span>Hidden</span></div>'
                    f'<div class="tile" style="--c:var(--review)"><b>{rev}</b><span>Review</span></div>'
                    f'<div class="tile" style="--c:var(--ok)"><b>{n-hid-rev}</b><span>Clean</span></div></div>',
                    unsafe_allow_html=True)
        find = st.text_input("Find a comment", key="find", label_visibility="collapsed",
                             placeholder="🔍  Search a word or @author to hide any comment…")
        if find.strip():
            f = find.strip().lower()
            items = [c for c in reversed(S.chat) if f in c["comment"].lower() or f in str(c["author"]).lower()][:15]
            empty = "No comment matches your search."
        else:
            flagged = [c for c in S.chat if c["score"] >= review_t or S.overrides.get(c["id"]) == "Hide"]
            items = sorted(flagged, key=lambda c: (c["id"] in S.overrides, -c["score"]))[:30]   # undecided first
            empty = "Nothing hidden or held yet. Flagged comments will be listed here."
        with st.container(height=470, border=False):
            if items:
                for c in items:
                    st.markdown(row(c, reveal=True), unsafe_allow_html=True)
                    mod_buttons(c, "f")
            else:
                st.caption(empty)

    every = {"free": 15, "api": 5, "chat": 3}.get(S.mode)
    with right:
        with keyed("modpanel"):
            st.markdown("##### Moderation")
            st.caption("Hidden and held comments, worst first. Hide or approve them yourself.")
            _ = frag(run_every=every)(feed_view)() if frag else feed_view()
            msg = st.chat_input("Test the filter: type a comment…")
            if msg:
                add("You", msg, f"local-{len(S.chat)}-{hash(msg)}")
                st.rerun()
    with left:
        _ = frag(run_every=every)(comments_view)() if frag else comments_view()

# ---------- Review queue ----------
else:
    page_head("Review queue", "Comments the filter hid or held. Hide or approve each one; your call updates the comment list.")
    q = st.session_state.queue
    pending = sorted([i for i, r in enumerate(q) if r["decision"] == "Pending"], key=lambda i: -q[i]["score"])
    c = st.columns(3)
    c[0].metric("Waiting", len(pending))
    c[1].metric("Hidden by you", sum(r["decision"] == "Hidden" for r in q))
    c[2].metric("Approved by you", sum(r["decision"] == "Approved" for r in q))
    if not q:
        st.info("Nothing to review yet. Flagged comments from the Comments page show up here.")
    for i in pending[:15]:
        r = q[i]
        r.setdefault("author", "you" if r.get("source") == "manual" else "viewer")
        st.markdown(row(r, reveal=True), unsafe_allow_html=True)
        with keyed(f"act_q_{i}"):
            a, b, _ = st.columns([1, 1, 4])
            if a.button("🚫 Hide", key=f"qh{i}", use_container_width=True):
                set_decision(r, "Hide"); st.rerun()
            if b.button("✓ Approve", key=f"qa{i}", use_container_width=True):
                set_decision(r, "Approve"); st.rerun()
    done = pd.DataFrame([r for r in q if r["decision"] != "Pending"])
    if not done.empty:
        done["label"] = (done["decision"] == "Hidden").astype(int)
        st.download_button("Download my decisions", done.to_csv(index=False).encode(),
                           "creator_decisions.csv", "text/csv")
