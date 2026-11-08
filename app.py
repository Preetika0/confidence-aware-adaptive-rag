from pathlib import Path
import json, re
import pandas as pd
import streamlit as st
from rag_app.core import get_embeddings, get_llm, ingest, split_docs, make_store, load_store, generate, buffer_memory, summary_memory, entity_memory, context_text

st.set_page_config(page_title="Adaptive RAG Assistant", page_icon="📚", layout="wide")
st.title("Confidence-Aware Adaptive RAG Assistant")
st.caption("Evidence-aware prompt selection for university regulations")
for key, value in {"messages": [], "db": None, "indexed": False, "summary": ""}.items():
    if key not in st.session_state: st.session_state[key] = value

with st.sidebar:
    st.header("RAG configuration")
    backend = st.selectbox("Vector database", ["FAISS", "ChromaDB"])
    chunking = st.selectbox("Chunking", ["Recursive", "Fixed-size", "Semantic"])
    size = st.slider("Chunk size", 300, 1500, 800, 100)
    overlap = st.slider("Chunk overlap", 0, 300, 100, 25)
    k = st.slider("Retrieved chunks (k)", 1, 10, 4)
    strategy = st.selectbox("Prompt strategy", ["Adaptive", "Standard", "Structured"])
    memory = st.selectbox("Conversation memory", ["Buffer", "Summary", "Entity", "Off"])
    planning = st.checkbox("Multi-part query planning", True)
    uploads = st.file_uploader("Upload official PDF/TXT/MD documents", type=["pdf","txt","md"], accept_multiple_files=True)
    if st.button("Build index", disabled=not uploads, type="primary"):
        paths = []
        for f in uploads:
            p = Path("data/documents") / f.name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f.getbuffer())
            paths.append(str(p))
        with st.spinner("Ingesting and indexing documents..."):
            emb = get_embeddings()
            docs = ingest(paths)
            chunks = split_docs(docs, chunking, size, overlap, emb)
            st.session_state.db = make_store(chunks, emb, backend, "outputs/" + backend.lower())
            st.session_state.indexed = True
            st.session_state.index_meta = {"pages":len(docs), "chunks":len(chunks), "chunking":chunking, "vector_db":backend}
        st.success("Index built.")
    if st.button("Load saved index"):
        try:
            st.session_state.db = load_store(get_embeddings(), backend, "outputs/" + backend.lower())
            st.session_state.indexed = True
            st.success("Saved index loaded.")
        except Exception as e: st.error(str(e))
    st.divider()
    st.header("50-question evaluation")
    test_upload = st.file_uploader("Upload JSONL test set", type=["jsonl"])
    if test_upload:
        Path("data/evaluation/testset.jsonl").write_bytes(test_upload.getbuffer())
    if st.button("Run evaluation", disabled=not (st.session_state.indexed and Path("data/evaluation/testset.jsonl").exists())):
        rows = [json.loads(x) for x in Path("data/evaluation/testset.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        model = get_llm()
        results = []
        with st.spinner("Evaluating all questions with LLM-as-Judge..."):
            for row in rows:
                out = generate(st.session_state.db, model, row["question"], strategy, k, query_planning=planning)
                sources = {d.metadata.get("source","").lower() for d in out["docs"]}
                gold = {s.lower() for s in row.get("relevant_sources", [])}
                recall = len(sources & gold)/len(gold) if gold else None
                judge = 'Return ONLY JSON with integer correctness (0-5), faithfulness (0-5), rationale. Question: {} Expected: {} Answer: {} Evidence: {}'.format(row["question"], row["expected_answer"], out["answer"], context_text(out["docs"]))
                raw = model.invoke(judge).content
                correctness = faithfulness = None
                m = re.search(r"\{.*\}", raw, re.S)
                if m:
                    try:
                        j = json.loads(m.group()); correctness = int(j.get("correctness",0)); faithfulness = int(j.get("faithfulness",0))
                    except Exception: pass
                results.append({"id":row["id"],"question":row["question"],"expected_level":row["expected_level"],"predicted_level":out["level"],"sufficiency_correct":out["level"]==row["expected_level"],"recall_at_k_source_proxy":recall,"correctness_0_5":correctness,"faithfulness_0_5":faithfulness,"latency_seconds":out["latency"],"answer":out["answer"]})
        df = pd.DataFrame(results)
        Path("outputs").mkdir(exist_ok=True)
        df.to_csv("outputs/evaluation_results.csv", index=False)
        st.session_state.eval_df = df

if "index_meta" in st.session_state: st.info("Active index: " + str(st.session_state.index_meta))
tab1, tab2, tab3 = st.tabs(["Assistant", "Evaluation", "Methodology"])
with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    q = st.chat_input("Ask about the uploaded university documents...")
    if q:
        if not st.session_state.indexed: st.warning("Upload official documents and build an index first.")
        else:
            st.session_state.messages.append({"role":"user","content":q})
            model = get_llm()
            if memory == "Buffer": history = buffer_memory(st.session_state.messages[:-1])
            elif memory == "Summary": history = st.session_state.summary
            elif memory == "Entity": history = "Entities: " + ", ".join(entity_memory(st.session_state.messages))
            else: history = ""
            with st.spinner("Retrieving evidence and selecting prompt..."):
                out = generate(st.session_state.db, model, q, strategy, k, history, planning)
            st.session_state.messages.append({"role":"assistant","content":out["answer"]})
            if memory == "Summary": st.session_state.summary = summary_memory(model, st.session_state.messages, st.session_state.summary)
            with st.chat_message("assistant"):
                st.markdown(out["answer"])
                st.caption("Evidence: {} | Latency: {:.2f}s".format(out["level"], out["latency"]))
                with st.expander("Inspect retrieval and decision"):
                    st.write("Query plan:", out["plan"]); st.write("Assessment:", out["reason"])
                    for d in out["docs"]:
                        st.markdown("**{} — page {}**".format(d.metadata.get("source"), d.metadata.get("page")))
                        st.write(d.page_content[:1000])
with tab2:
    if "eval_df" in st.session_state:
        df = st.session_state.eval_df
        st.dataframe(df, use_container_width=True)
        st.download_button("Download evaluation CSV", df.to_csv(index=False), "evaluation_results.csv")
        st.metric("Sufficiency classification accuracy", "{:.1%}".format(df.sufficiency_correct.mean()))
        st.metric("Mean latency", "{:.2f}s".format(df.latency_seconds.mean()))
    else: st.info("Upload your JSONL test set and run evaluation from the sidebar.")
with tab3:
    st.markdown("""
    **Prompt A — Standard:** answer using retrieved context.  
    **Prompt B — Structured:** use only evidence, cite sources and flag missing details.  
    **Prompt C — Adaptive:** enough evidence → answer; partial → answer supported parts and identify gaps; insufficient → abstain.

    Compare fixed-size, recursive and semantic splitting, plus FAISS and ChromaDB. Keep corpus, model, retrieval settings and questions identical when comparing prompts.
    """)
    st.warning("The current Recall@k is a source-file overlap proxy. Token usage is not yet captured. Treat LLM-as-Judge as an aid and manually verify scores.")
