import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import BertTokenizer, BertForMaskedLM

st.set_page_config(
    page_title="Log Anomaly Detector",
    page_icon="🛡️",
    layout="wide"
)

# ============ TOGGLE THEME ============
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

col_toggle, _ = st.columns([1, 9])
with col_toggle:
    if st.button("🌙 Dark" if not st.session_state.dark_mode else "☀️ Light"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

dark = st.session_state.dark_mode

# ============ COULEURS SELON THÈME ============
if dark:
    bg       = "#0a0e1a"
    card_bg  = "#111827"
    border   = "#1f2937"
    text     = "#e2e8f0"
    subtext  = "#6b7280"
    accent   = "#38bdf8"
    hero_bg  = "linear-gradient(135deg, #0f1729 0%, #1a2744 100%)"
    hero_brd = "#1e3a5f"
    plot_bg  = "#111827"
    tick_col = "#6b7280"
    spine_col= "#1f2937"
    legend_fc= "#1f2937"
    legend_ec= "#374151"
else:
    bg       = "#f8fafc"
    card_bg  = "#ffffff"
    border   = "#e2e8f0"
    text     = "#0f172a"
    subtext  = "#64748b"
    accent   = "#0284c7"
    hero_bg  = "linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%)"
    hero_brd = "#e2e8f0"
    plot_bg  = "#ffffff"
    tick_col = "#94a3b8"
    spine_col= "#e2e8f0"
    legend_fc= "#ffffff"
    legend_ec= "#e2e8f0"

# ============ CSS ============
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {bg};
    color: {text};
}}
.main {{ background-color: {bg}; }}
.block-container {{ padding: 2rem 3rem; }}

.hero {{
    background: {hero_bg};
    border: 1px solid {hero_brd};
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}}

.hero-title {{
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: {text};
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
}}
.hero-title span {{ color: {accent}; }}
.hero-subtitle {{ font-size: 0.95rem; color: {subtext}; margin: 0; font-weight: 300; }}

.badge {{
    display: inline-block;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    color: {accent};
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-right: 0.5rem;
    margin-top: 1rem;
}}

.metric-card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}

.metric-value {{
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0.25rem 0;
}}

.metric-label {{
    font-size: 0.8rem;
    color: {subtext};
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.normal-val   {{ color: {'#22c55e' if dark else '#16a34a'}; }}
.moyen-val    {{ color: {'#eab308' if dark else '#ca8a04'}; }}
.eleve-val    {{ color: {'#f97316' if dark else '#ea580c'}; }}
.critique-val {{ color: {'#ef4444' if dark else '#dc2626'}; }}

.section-title {{
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: {accent};
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid {border};
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}}

.log-row {{
    background: {card_bg};
    border-left: 3px solid #22c55e;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    border-radius: 0 8px 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: {subtext};
}}
.log-row.critique {{ border-left-color: #ef4444; color: {'#fca5a5' if dark else '#dc2626'}; }}
.log-row.eleve    {{ border-left-color: #f97316; color: {'#fdba74' if dark else '#ea580c'}; }}
.log-row.moyen    {{ border-left-color: #eab308; color: {'#fde047' if dark else '#ca8a04'}; }}

.score-pill {{
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    margin-left: 0.5rem;
}}
.score-critique {{ background: rgba(239,68,68,0.2); color: #ef4444; }}
.score-eleve    {{ background: rgba(249,115,22,0.2); color: #f97316; }}
.score-moyen    {{ background: rgba(234,179,8,0.2);  color: #eab308; }}

.stButton > button {{
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.3) !important;
}}
</style>
""", unsafe_allow_html=True)

# ============ MODÈLE ============
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained(
        "seynabouthiandoum/bert-log-anomaly-detection"
    )
    model.eval()
    return tokenizer, model

def preprocess_log(line):
    line = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "[TIMESTAMP]", line)
    line = re.sub(r"\d{1,3}(\.\d{1,3}){3}", "[IP]", line)
    line = re.sub(r"\b\d{5,}\b", "[NUM]", line)
    return re.sub(r"\s+", " ", line).strip()

def compute_score(text, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128, padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()

# ============ HERO ============
st.markdown(f"""
<div class="hero">
    <p class="hero-title">🛡️ Log <span>Anomaly</span> Detector</p>
    <p class="hero-subtitle">Détection intelligente d'anomalies • BERT fine-tuné par Masked Language Modeling</p>
    <span class="badge">BERT-base-uncased</span>
    <span class="badge">ESP/UCAD</span>
    <span class="badge">Seynabou Thiandoum & Fatou Gueye Ndong</span>
</div>
""", unsafe_allow_html=True)

# ============ UPLOAD ============
st.markdown('<p class="section-title">📂 Source de données</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["log","txt"], label_visibility="collapsed")

if uploaded_file:
    lines = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    col1, col2 = st.columns([3, 1])
    with col1:
        max_lines = st.slider("Lignes à analyser", 50, min(500, len(lines)), 200)
    with col2:
        st.metric("Lignes détectées", f"{len(lines):,}")

    lines = lines[:max_lines]

    if st.button("⚡ LANCER L'ANALYSE"):
        with st.spinner("Chargement du modèle BERT..."):
            tokenizer, model = load_model()

        scores = []
        bar = st.progress(0)
        status = st.empty()

        for i, line in enumerate(lines):
            scores.append(compute_score(preprocess_log(line), model, tokenizer))
            if i % 10 == 0:
                bar.progress(i / len(lines))
                status.markdown(f"⏳ `{i}/{len(lines)}` logs analysés...")

        bar.progress(1.0)
        status.markdown("✅ **Analyse terminée !**")

        p90 = np.percentile(scores, 90)
        p95 = np.percentile(scores, 95)
        p99 = np.percentile(scores, 99)

        def categorize(s):
            if s >= p99: return "Critique"
            elif s >= p95: return "Élevé"
            elif s >= p90: return "Moyen"
            return "Normal"

        df = pd.DataFrame({"log": lines, "score": scores})
        df["categorie"] = df["score"].apply(categorize)
        counts = df["categorie"].value_counts()

        # MÉTRIQUES
        st.markdown('<br><p class="section-title">📊 Tableau de bord</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, cat, cls, sub in zip(
            [c1,c2,c3,c4],
            ["Normal","Moyen","Élevé","Critique"],
            ["normal","moyen","eleve","critique"],
            ["logs sains","à surveiller","suspects","anomalies"]
        ):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{cat}</div>
                    <div class="metric-value {cls}-val">{counts.get(cat,0)}</div>
                    <div class="metric-label">{sub}</div>
                </div>""", unsafe_allow_html=True)

        # GRAPHIQUES
        st.markdown("<br>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=plot_bg)
        for ax in axes:
            ax.set_facecolor(plot_bg)
            ax.tick_params(colors=tick_col)
            for spine in ax.spines.values():
                spine.set_edgecolor(spine_col)

        kde_color = "#38bdf8" if dark else "#0284c7"
        sns.kdeplot(scores, ax=axes[0], fill=True, color=kde_color, alpha=0.3, linewidth=2)
        axes[0].axvline(p90, color="#eab308", linestyle="--", linewidth=1.5, label="Seuil 90%")
        axes[0].axvline(p95, color="#f97316", linestyle="--", linewidth=1.5, label="Seuil 95%")
        axes[0].axvline(p99, color="#ef4444", linestyle="--", linewidth=1.5, label="Seuil 99%")
        axes[0].set_title("Distribution des scores", color=text, fontsize=11, pad=12)
        axes[0].set_xlabel("Score", color=subtext, fontsize=9)
        axes[0].set_ylabel("Densité", color=subtext, fontsize=9)
        axes[0].legend(facecolor=legend_fc, edgecolor=legend_ec,
                       labelcolor=text, fontsize=8)

        cats_order = ["Normal","Moyen","Élevé","Critique"]
        bar_colors = ["#22c55e","#eab308","#f97316","#ef4444"]
        vals = [counts.get(c, 0) for c in cats_order]
        bars = axes[1].bar(cats_order, vals, color=bar_colors, width=0.5)
        for b, v in zip(bars, vals):
            axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                         str(v), ha='center', color=text, fontsize=9)
        axes[1].set_title("Répartition par sévérité", color=text, fontsize=11, pad=12)
        axes[1].set_ylabel("Nombre de logs", color=subtext, fontsize=9)

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        # TOP ANOMALIES
        st.markdown('<br><p class="section-title">🚨 Top anomalies</p>', unsafe_allow_html=True)
        for _, row in df.nlargest(10, "score").iterrows():
            cat = row['categorie'].lower().replace('é','e')
            cat_class = cat if cat in ['critique','eleve','moyen'] else ''
            score_class = f"score-{cat_class}" if cat_class else ''
            preview = row['log'][:100] + "..." if len(row['log']) > 100 else row['log']
            st.markdown(f"""<div class="log-row {cat_class}">
                {preview}
                <span class="score-pill {score_class}">{row['score']:.3f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Télécharger CSV",
            df.to_csv(index=False),
            "anomaly_results.csv", "text/csv"
        )
