import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
from transformers import BertTokenizer, BertForMaskedLM

st.set_page_config(
    page_title="Log Anomaly Detector",
    page_icon="🛡️",
    layout="wide"
)

# ============ CSS CUSTOM ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

.main { background-color: #0a0e1a; }
.block-container { padding: 2rem 3rem; }

.hero {
    background: linear-gradient(135deg, #0f1729 0%, #1a2744 50%, #0f1729 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(56, 189, 248, 0.05) 0%, transparent 60%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
}

.hero-title span { color: #38bdf8; }

.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 300;
}

.badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-right: 0.5rem;
    margin-top: 1rem;
}

.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: border-color 0.2s;
}

.metric-card:hover { border-color: #374151; }

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0.25rem 0;
}

.metric-label {
    font-size: 0.8rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.normal-val { color: #22c55e; }
.moyen-val  { color: #eab308; }
.eleve-val  { color: #f97316; }
.critique-val { color: #ef4444; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.log-row {
    background: #111827;
    border-left: 3px solid #22c55e;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    border-radius: 0 8px 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
}

.log-row.critique { border-left-color: #ef4444; color: #fca5a5; }
.log-row.eleve    { border-left-color: #f97316; color: #fdba74; }
.log-row.moyen    { border-left-color: #eab308; color: #fde047; }

.score-pill {
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    margin-left: 0.5rem;
}

.score-critique { background: rgba(239,68,68,0.2); color: #ef4444; }
.score-eleve    { background: rgba(249,115,22,0.2); color: #f97316; }
.score-moyen    { background: rgba(234,179,8,0.2); color: #eab308; }

div[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 2px dashed #1f2937 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.3) !important;
}

.stSlider > div { color: #94a3b8 !important; }
.stProgress > div > div { background: #38bdf8 !important; }
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
st.markdown("""
<div class="hero">
    <p class="hero-title">🛡️ Log <span>Anomaly</span> Detector</p>
    <p class="hero-subtitle">Détection intelligente d'anomalies par fine-tuning BERT • Masked Language Modeling</p>
    <span class="badge">BERT-base-uncased</span>
    <span class="badge">ESP/UCAD</span>
    <span class="badge">Seynabou Thiandoum & Fatou Gueye Ndong</span>
</div>
""", unsafe_allow_html=True)

# ============ UPLOAD ============
st.markdown('<p class="section-title">📂 Source de données</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["log", "txt"], label_visibility="collapsed")

if uploaded_file:
    lines = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    col1, col2 = st.columns([3, 1])
    with col1:
        max_lines = st.slider("Nombre de lignes à analyser", 50, min(500, len(lines)), 200)
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

        # ── MÉTRIQUES ──
        st.markdown('<br><p class="section-title">📊 Tableau de bord</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        counts = df["categorie"].value_counts()

        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Normal</div>
                <div class="metric-value normal-val">{counts.get('Normal',0)}</div>
                <div class="metric-label">logs sains</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Moyen</div>
                <div class="metric-value moyen-val">{counts.get('Moyen',0)}</div>
                <div class="metric-label">à surveiller</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Élevé</div>
                <div class="metric-value eleve-val">{counts.get('Élevé',0)}</div>
                <div class="metric-label">suspects</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Critique</div>
                <div class="metric-value critique-val">{counts.get('Critique',0)}</div>
                <div class="metric-label">anomalies</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GRAPHIQUES ──
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                                  facecolor='#111827')
        for ax in axes:
            ax.set_facecolor('#111827')
            ax.tick_params(colors='#6b7280')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1f2937')

        sns.kdeplot(scores, ax=axes[0], fill=True,
                    color="#38bdf8", alpha=0.3, linewidth=2)
        axes[0].axvline(p90, color="#eab308", linestyle="--",
                        linewidth=1.5, label="Seuil 90%")
        axes[0].axvline(p95, color="#f97316", linestyle="--",
                        linewidth=1.5, label="Seuil 95%")
        axes[0].axvline(p99, color="#ef4444", linestyle="--",
                        linewidth=1.5, label="Seuil 99%")
        axes[0].set_title("Distribution des scores d'anomalie",
                           color="#e2e8f0", fontsize=11, pad=12)
        axes[0].set_xlabel("Score", color="#6b7280", fontsize=9)
        axes[0].set_ylabel("Densité", color="#6b7280", fontsize=9)
        axes[0].legend(facecolor='#1f2937', edgecolor='#374151',
                       labelcolor='#e2e8f0', fontsize=8)

        cats = ["Normal", "Moyen", "Élevé", "Critique"]
        colors = ["#22c55e", "#eab308", "#f97316", "#ef4444"]
        vals = [counts.get(c, 0) for c in cats]
        bars = axes[1].bar(cats, vals, color=colors,
                           width=0.5, edgecolor='none')
        for bar, val in zip(bars, vals):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.5, str(val),
                         ha='center', va='bottom',
                         color='#e2e8f0', fontsize=9,
                         fontfamily='monospace')
        axes[1].set_title("Répartition par sévérité",
                           color="#e2e8f0", fontsize=11, pad=12)
        axes[1].set_ylabel("Nombre de logs", color="#6b7280", fontsize=9)

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        # ── TOP ANOMALIES ──
        st.markdown('<br><p class="section-title">🚨 Top anomalies critiques</p>',
                    unsafe_allow_html=True)
        top = df.nlargest(10, "score")

        for _, row in top.iterrows():
            cat = row['categorie'].lower().replace('é','e')
            score_class = f"score-{cat}" if cat in ['critique','eleve','moyen'] else ''
            cat_class = cat if cat in ['critique','eleve','moyen'] else ''
            log_preview = row['log'][:100] + "..." if len(row['log']) > 100 else row['log']
            st.markdown(f"""
            <div class="log-row {cat_class}">
                {log_preview}
                <span class="score-pill {score_class}">{row['score']:.3f}</span>
            </div>""", unsafe_allow_html=True)

        # ── EXPORT ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Télécharger les résultats (CSV)",
            df.to_csv(index=False),
            "anomaly_results.csv",
            "text/csv"
        )
