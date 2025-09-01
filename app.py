import os
import pickle
import numpy as np
import streamlit as st
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --------------------------
# Caching heavy resources
# --------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    # MiniLM: fast, 384-dim sentence embeddings (great on CPU)
    return SentenceTransformer("all-MiniLM-L6-v2")

import os
import faiss
import pickle
import streamlit as st   # <-- this was missing

BASE_DIR = "C:/Users/nithi/OneDrive/Desktop/Github/DeepLearning/Explainer_AI"

@st.cache_resource(show_spinner=False)
def load_index():
    index_path = os.path.join(BASE_DIR, "models", "policy.index")
    map_path   = os.path.join(BASE_DIR, "models", "policy.id2text.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"id2text mapping not found at {map_path}")

    index = faiss.read_index(index_path)
    with open(map_path, "rb") as f:
        id2text = pickle.load(f)

    return index, id2text

@st.cache_resource(show_spinner=False)
def load_qa_model(model_name: str):
    # FLAN-T5 is instruction-tuned → better for Q&A with context
    return pipeline("text2text-generation", model=model_name, tokenizer=model_name)

# --------------------------
# Helpers
# --------------------------
def normalize(vec):
    # Ensure shape [1, dim] and float32
    if vec.ndim == 1: vec = vec[None, :]
    vec = vec.astype("float32")
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    return vec / norm

def retrieve(query, embedder, index, id2text, k=3):
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    q = normalize(q)
    scores, idxs = index.search(q, k)
    hits = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx == -1:  # in case index is empty/short
            continue
        hits.append({"rank": rank, "score": float(score), "text": id2text[idx], "idx": int(idx)})
    return hits

def build_prompt(query, hits, max_chars=3000):
    # Join top-k chunks as context; keep it readable for model
    context = ""
    for h in hits:
        piece = h["text"].strip()
        if len(context) + len(piece) + 2 > max_chars:
            break
        context += piece + "\n\n"
    # Instruct FLAN-T5 to answer from context (not general knowledge)
    prompt = (
        "Answer the question using ONLY the context. Be concise and clear.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n"
        "Answer:"
    )
    return prompt, context

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="Medicaid & SNAP Policy Assistant", page_icon="📄", layout="centered")
st.title("📄 Medicaid & SNAP Policy Assistant (Local RAG)")

with st.sidebar:
    st.markdown("### Settings")
    model_choice = st.selectbox("FLAN-T5 model", ["google/flan-t5-base", "google/flan-t5-small"])
    top_k = st.slider("Top-K retrieved chunks", 1, 8, 4)
    max_len = st.slider("Max answer length (tokens)", 64, 512, 280, step=16)
    show_chunks = st.checkbox("Show retrieved chunks", value=True)

st.info("Type a policy question, e.g., **Who is eligible for SNAP?** or **What does Medicaid cover for adults?**")

query = st.text_input("Your question")
go = st.button("Get answer")

# Lazy-load resources
embedder = load_embedder()
index, id2text = load_index()
qa = load_qa_model(model_choice)

if go and query.strip():
    with st.spinner("Retrieving and generating..."):
        # 1) Retrieve
        hits = retrieve(query, embedder, index, id2text, k=top_k)

        # 2) Prompt FLAN-T5 with Question + Context
        prompt, context = build_prompt(query, hits, max_chars=3000)

        # 3) Generate
        out = qa(prompt, max_length=max_len, do_sample=False)[0]["generated_text"].strip()

    st.subheader("Answer")
    st.text_area(" ", value=out, height=240)

    if show_chunks:
        st.subheader("Sources (retrieved chunks)")
        for h in hits:
            with st.expander(f"[{h['rank']}] score={h['score']:.3f} • idx={h['idx']}"):
                st.write(h["text"])
