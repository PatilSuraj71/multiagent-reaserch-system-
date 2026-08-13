"""
Advanced Streamlit UI for the Multi-Agent Research System
Run with:  streamlit run app.py
"""

import time
from datetime import datetime

import streamlit as st

from Agent import build_reader_agent, build_search_agent, writer_chain, critic_chain


st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = [
    {"key": "search", "label": "Search", "icon": "🔎", "desc": "Finding sources"},
    {"key": "read", "label": "Read", "icon": "📖", "desc": "Scraping content"},
    {"key": "write", "label": "Write", "icon": "✍️", "desc": "Drafting report"},
    {"key": "critique", "label": "Critique", "icon": "🧐", "desc": "Reviewing draft"},
]


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main > div { padding-top: 1.2rem; }
    #MainMenu, footer { visibility: hidden; }

    /* ---------- Hero ---------- */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: radial-gradient(circle at 20% 20%, rgba(99,102,241,0.35), transparent 60%),
                    linear-gradient(135deg, #14162a 0%, #1a1d27 60%, #1e2130 100%);
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute; top: -50%; right: -10%;
        width: 300px; height: 300px; border-radius: 50%;
        background: radial-gradient(circle, rgba(99,102,241,0.25), transparent 70%);
    }
    .hero h1 {
        margin: 0; font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #a5b4fc, #f0abfc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { margin: .5rem 0 0 0; color: #9ca3af; font-size: .95rem; }
    .pill-row { margin-top: .9rem; display: flex; gap: .5rem; flex-wrap: wrap; }
    .pill {
        font-size: .72rem; font-weight: 600; padding: .25rem .7rem;
        border-radius: 999px; background: rgba(99,102,241,0.15);
        color: #c7d2fe; border: 1px solid rgba(99,102,241,0.3);
    }

    /* ---------- Stepper ---------- */
    .stepper { display: flex; align-items: center; margin: 1.4rem 0 1.6rem 0; }
    .step-wrap { display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; width: 110px; }
    .step-circle {
        width: 52px; height: 52px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.35rem; border: 2px solid #374151; background: #1a1d27;
        transition: all .3s ease;
    }
    .step-circle.pending { color: #6b7280; border-color: #374151; }
    .step-circle.active {
        color: #fff; border-color: #6366f1; background: #312e81;
        box-shadow: 0 0 0 6px rgba(99,102,241,0.18);
        animation: pulse 1.4s infinite;
    }
    .step-circle.done { color: #fff; border-color: #22c55e; background: #14532d; }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 4px rgba(99,102,241,0.25); }
        50% { box-shadow: 0 0 0 10px rgba(99,102,241,0.08); }
        100% { box-shadow: 0 0 0 4px rgba(99,102,241,0.25); }
    }
    .step-label { margin-top: .5rem; font-size: .82rem; font-weight: 700; color: #e5e7eb; }
    .step-desc { font-size: .7rem; color: #6b7280; }
    .step-line { flex: 1 1 auto; height: 3px; background: #374151; margin: 0 -6px; margin-bottom: 24px; border-radius: 2px; }
    .step-line.done { background: linear-gradient(90deg, #22c55e, #16a34a); }
    .step-line.active { background: linear-gradient(90deg, #22c55e, #6366f1); }

    /* ---------- Terminal log ---------- */
    .terminal {
        background: #0a0b10; border: 1px solid #262a38; border-radius: 12px;
        padding: .9rem 1.1rem; font-family: 'JetBrains Mono', monospace;
        font-size: .78rem; color: #9ca3af; max-height: 220px; overflow-y: auto;
    }
    .terminal .ok { color: #4ade80; }
    .terminal .ts { color: #6366f1; }

    /* ---------- Cards ---------- */
    .glass-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: .8rem;
    }
    .report-box {
        border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
        padding: 1.4rem 1.6rem; background: rgba(255,255,255,0.02);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: .7rem;
    }
    .badge { display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.7rem; font-weight:700; }
    .badge-green { background:#14532d; color:#4ade80; }
    .badge-gray { background:#374151; color:#9ca3af; }
    </style>
    """,
    unsafe_allow_html=True,
)


defaults = {
    "history": [],
    "state": None,
    "running": False,
    "log_lines": [],
    "topic_input": "",
    "compare_ids": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def render_stepper(current_index: int, total_steps: int = len(STEPS)) -> str:
    """current_index: -1 = nothing started, 0..3 = that step active, 4 = all done"""
    html = ['<div class="stepper">']
    for i, step in enumerate(STEPS):
        if current_index > i:
            circle_state, line_state = "done", "done"
            icon = "✔"
        elif current_index == i:
            circle_state, line_state = "active", "active"
            icon = step["icon"]
        else:
            circle_state, line_state = "pending", "pending"
            icon = step["icon"]

        html.append('<div class="step-wrap">')
        html.append(f'<div class="step-circle {circle_state}">{icon}</div>')
        html.append(f'<div class="step-label">{step["label"]}</div>')
        html.append(f'<div class="step-desc">{step["desc"]}</div>')
        html.append("</div>")

        if i < len(STEPS) - 1:
            html.append(f'<div class="step-line {line_state if current_index > i else ""}"></div>')
    html.append("</div>")
    return "".join(html)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_lines.append(f'<span class="ts">[{ts}]</span> {msg}')


def render_log() -> str:
    if not st.session_state.log_lines:
        return '<div class="terminal">Waiting to start...</div>'
    return '<div class="terminal">' + "<br>".join(st.session_state.log_lines) + "</div>"


def word_count(text: str) -> int:
    return len(str(text).split())


def reading_time(text: str) -> str:
    minutes = max(1, round(word_count(text) / 200))
    return f"~{minutes} min read"


EXAMPLE_TOPICS = [
    "Latest advances in solid-state batteries",
    "Impact of AI on drug discovery",
    "Global semiconductor supply chain 2026",
    "Progress in fusion energy research",
]



with st.sidebar:
    st.markdown("## 🧠 Research Agents")
    st.caption("Multi-agent pipeline · Search → Read → Write → Critique")
    st.divider()

    show_raw = st.toggle("Show raw agent output while running", value=True)
    auto_expand = st.toggle("Auto-expand final report", value=True)

    st.divider()
    st.markdown("**History**")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - 1 - i
            cols = st.columns([5, 1])
            with cols[0]:
                if st.button(f"📄 {h['topic'][:26]}", key=f"hist_{real_idx}", use_container_width=True):
                    st.session_state.state = h
            with cols[1]:
                checked = real_idx in st.session_state.compare_ids
                if st.checkbox("", value=checked, key=f"cmp_{real_idx}"):
                    if real_idx not in st.session_state.compare_ids:
                        st.session_state.compare_ids.append(real_idx)
                elif real_idx in st.session_state.compare_ids:
                    st.session_state.compare_ids.remove(real_idx)

        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.state = None
            st.session_state.compare_ids = []
            st.rerun()
    else:
        st.caption("No past runs yet.")

    st.divider()
    st.caption("Select 2 runs above to compare them side by side.")



st.markdown(
    """
    <div class="hero">
        <h1>Multi-Agent Research System</h1>
        <p>Four specialized agents collaborate to search, read, write, and critique a research report — live.</p>
        <div class="pill-row">
            <span class="pill">🔎 Search Agent</span>
            <span class="pill">📖 Reader Agent</span>
            <span class="pill">✍️ Writer Chain</span>
            <span class="pill">🧐 Critic Chain</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# INPUT
# ==========================================================================
col1, col2 = st.columns([5, 1])
with col1:
    topic = st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="e.g. Latest advances in solid-state batteries",
        label_visibility="collapsed",
        disabled=st.session_state.running,
    )
with col2:
    run_clicked = st.button(
        "🚀 Run",
        use_container_width=True,
        disabled=st.session_state.running or not topic.strip(),
        type="primary",
    )

def _set_topic(value: str):
    st.session_state.topic_input = value


chip_cols = st.columns(len(EXAMPLE_TOPICS))
for c, ex in zip(chip_cols, EXAMPLE_TOPICS):
    with c:
        st.button(
            ex,
            key=f"ex_{ex}",
            use_container_width=True,
            disabled=st.session_state.running,
            on_click=_set_topic,
            args=(ex,),
        )

stepper_placeholder = st.empty()
log_placeholder = st.empty()
cards_placeholder = st.container()

if not st.session_state.running:
    stepper_placeholder.markdown(render_stepper(-1), unsafe_allow_html=True)



def run_pipeline_with_ui(topic: str) -> dict:
    state = {"topic": topic, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
    st.session_state.log_lines = []

    # ---- Step 1: Search ----
    stepper_placeholder.markdown(render_stepper(0), unsafe_allow_html=True)
    log(f"Starting research on <b>{topic}</b>")
    log("Search agent dispatched...")
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)

    search_agent = build_search_agent()
    search_result = search_agent.invoke(
        {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
    )
    state["search_results"] = search_result["messages"][-1].content
    log(f'<span class="ok">✔ Search complete</span> — {word_count(state["search_results"])} words retrieved')
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)

    if show_raw:
        with cards_placeholder.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"**🔎 Search Agent**\n\n{state['search_results'][:600]}"
                        + ("..." if len(state["search_results"]) > 600 else ""))
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- Step 2: Read ----
    stepper_placeholder.markdown(render_stepper(1), unsafe_allow_html=True)
    log("Reader agent scraping top source...")
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Based on the following search results about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{state['search_results'][:800]}",
                )
            ]
        }
    )
    state["scraped_content"] = reader_result["messages"][-1].content
    log(f'<span class="ok">✔ Scraping complete</span> — {word_count(state["scraped_content"])} words extracted')
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)

    if show_raw:
        with cards_placeholder.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"**📖 Reader Agent**\n\n{state['scraped_content'][:600]}"
                        + ("..." if len(state["scraped_content"]) > 600 else ""))
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- Step 3: Write ----
    stepper_placeholder.markdown(render_stepper(2), unsafe_allow_html=True)
    log("Writer drafting the report...")
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({"topic": topic, "research": research_combined})
    log(f'<span class="ok">✔ Draft complete</span> — {word_count(state["report"])} words written')
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)

    if show_raw:
        with cards_placeholder.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**✍️ Writer** — draft report ready for review.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- Step 4: Critique ----
    stepper_placeholder.markdown(render_stepper(3), unsafe_allow_html=True)
    log("Critic reviewing the draft...")
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)

    state["feedback"] = critic_chain.invoke({"report": state["report"]})
    log(f'<span class="ok">✔ Review complete</span>')
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)

    if show_raw:
        with cards_placeholder.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"**🧐 Critic**\n\n" + str(state["feedback"])[:600]
                        + ("..." if len(str(state["feedback"])) > 600 else ""))
            st.markdown("</div>", unsafe_allow_html=True)

    stepper_placeholder.markdown(render_stepper(4), unsafe_allow_html=True)
    log("Pipeline finished ✅")
    log_placeholder.markdown(render_log(), unsafe_allow_html=True)
    return state


if run_clicked:
    st.session_state.running = True
    try:
        result_state = run_pipeline_with_ui(topic)
        result_state["id"] = len(st.session_state.history)
        st.session_state.state = result_state
        st.session_state.history.append(result_state)
    except Exception as e:
        st.error(f"⚠️ Pipeline failed: {e}")
    finally:
        st.session_state.running = False
        time.sleep(0.4)
        st.rerun()



if len(st.session_state.compare_ids) == 2:
    st.divider()
    st.subheader("⚖️ Compare Runs")
    ids = sorted(st.session_state.compare_ids)
    c1, c2 = st.columns(2)
    for col, idx in zip((c1, c2), ids):
        h = st.session_state.history[idx]
        with col:
            st.markdown(f"**{h['topic']}**")
            st.caption(h.get("timestamp", ""))
            m1, m2 = st.columns(2)
            m1.metric("Words", word_count(h.get("report", "")))
            m2.metric("Reading time", reading_time(h.get("report", "")))
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            st.markdown(str(h.get("report", ""))[:1500])
            st.markdown("</div>", unsafe_allow_html=True)
    st.divider()


state = st.session_state.state

if state and not st.session_state.running:
    st.divider()
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.subheader(f"📋 {state['topic']}")
        st.caption(f"Generated {state.get('timestamp', '')}")
    with top_r:
        st.markdown('<span class="badge badge-green">Complete</span>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Search results", f'{word_count(state.get("search_results", ""))} words')
    m2.metric("Scraped content", f'{word_count(state.get("scraped_content", ""))} words')
    m3.metric("Report", f'{word_count(state.get("report", ""))} words')
    m4.metric("Reading time", reading_time(state.get("report", "")))

    tab_report, tab_feedback, tab_sources, tab_analytics = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🗂️ Sources & Data", "📊 Analytics"]
    )

    with tab_report:
        with st.expander("Report", expanded=auto_expand):
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            st.markdown(str(state.get("report", "")))
            st.markdown("</div>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download report (.md)",
                data=str(state.get("report", "")),
                file_name=f"report_{state['topic'][:30].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ Download report (.txt)",
                data=str(state.get("report", "")),
                file_name=f"report_{state['topic'][:30].replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    with tab_feedback:
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown(str(state.get("feedback", "")))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_sources:
        s1, s2 = st.tabs(["Search Results", "Scraped Content"])
        with s1:
            st.text_area("Search results", state.get("search_results", ""), height=320, disabled=True)
        with s2:
            st.text_area("Scraped content", state.get("scraped_content", ""), height=320, disabled=True)

    with tab_analytics:
        a1, a2, a3 = st.columns(3)
        a1.metric("Search → Scraped ratio", f'{word_count(state.get("scraped_content","")) / max(1, word_count(state.get("search_results",""))):.2f}x')
        a2.metric("Report / Research ratio", f'{word_count(state.get("report","")) / max(1, word_count(state.get("search_results","")) + word_count(state.get("scraped_content",""))):.2f}x')
        a3.metric("Total runs so far", len(st.session_state.history))
        st.caption("Ratios are rough signals of how much the writer condensed vs. expanded the raw research.")

elif not state:
    st.info("👆 Enter a topic (or pick an example above) and click **Run** to start the pipeline.")