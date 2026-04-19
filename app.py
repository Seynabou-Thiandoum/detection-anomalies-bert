import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import BertTokenizer, BertForMaskedLM

st.set_page_config(
    page_title="Détection d'Anomalies - BERT",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("seynabouthiandoum/bert-log-anomaly-detection")
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

st.title("🔍 Détection d\'Anomalies dans les Logs")
st.markdown("**Modèle : BERT-base-uncased fine-tuné par MLM**")
st.markdown("*Seynabou Thiandoum & Fatou Gueye Ndong — ESP/UCAD*")
st.markdown("---")

uploaded_file = st.file_uploader("📂 Uploadez votre fichier .log", type=["log","txt"])

if uploaded_file:
    lines = uploaded_file.read().decode("utf-8", errors="ignore").splitlines()
    lines = [l.strip() for l in lines if l.strip()]
    st.info(f"📄 {len(lines)} lignes chargées")

    max_lines = st.slider("Lignes à analyser", 100, min(500, len(lines)), 200)
    lines = lines[:max_lines]

    if st.button("🚀 Lancer la détection", type="primary"):
        with st.spinner("Chargement du modèle..."):
            tokenizer, model = load_model()

        scores = []
        bar = st.progress(0)
        status = st.empty()

        for i, line in enumerate(lines):
            scores.append(compute_score(preprocess_log(line), model, tokenizer))
            if i % 10 == 0:
                bar.progress(i / len(lines))
                status.text(f"⏳ {i}/{len(lines)} logs analysés...")

        bar.progress(1.0)
        status.text("✅ Analyse terminée !")

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

        st.markdown("---")
        st.subheader("📊 Résultats")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Normal",   df[df.categorie=="Normal"].shape[0])
        c2.metric("🟡 Moyen",    df[df.categorie=="Moyen"].shape[0])
        c3.metric("🟠 Élevé",    df[df.categorie=="Élevé"].shape[0])
        c4.metric("🔴 Critique", df[df.categorie=="Critique"].shape[0])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.kdeplot(scores, ax=axes[0], fill=True, color="steelblue")
        axes[0].axvline(p90, color="orange", linestyle="--", label="90%")
        axes[0].axvline(p95, color="red", linestyle="--", label="95%")
        axes[0].axvline(p99, color="darkred", linestyle="--", label="99%")
        axes[0].set_title("Distribution des scores")
        axes[0].legend()

        counts = df["categorie"].value_counts().reindex(["Normal","Moyen","Élevé","Critique"])
        axes[1].bar(counts.index, counts.values, color=["green","orange","darkorange","red"])
        axes[1].set_title("Anomalies par catégorie")
        st.pyplot(fig)

        st.subheader("🚨 Top 10 Anomalies")
        st.dataframe(df.nlargest(10, "score")[["log","score","categorie"]], use_container_width=True)

        st.download_button("⬇️ Télécharger CSV", df.to_csv(index=False), "resultats.csv", "text/csv")
