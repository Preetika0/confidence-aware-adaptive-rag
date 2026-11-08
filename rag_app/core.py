import os, re, json, time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

load_dotenv()
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def get_llm():
    return ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0.0, top_k=40, top_p=0.9)

def ingest(paths):
    docs = []
    for path in paths:
        p = Path(path)
        loaded = PyPDFLoader(str(p)).load() if p.suffix.lower() == ".pdf" else TextLoader(str(p), encoding="utf-8").load()
        for d in loaded:
            d.metadata["source"] = p.name
            d.metadata["page"] = int(d.metadata.get("page", -1)) + 1 if "page" in d.metadata else "N/A"
        docs.extend(loaded)
    return docs

def split_docs(docs, kind, size, overlap, emb):
    if kind == "Fixed-size":
        splitter = CharacterTextSplitter(separator=" ", chunk_size=size, chunk_overlap=overlap)
    elif kind == "Recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                    separators=["\n\n", "\n", ". ", " ", ""])
    else:
        splitter = SemanticChunker(emb, breakpoint_threshold_type="percentile")
    return splitter.split_documents(docs)

def make_store(chunks, emb, backend, folder):
    Path(folder).mkdir(parents=True, exist_ok=True)
    if backend == "FAISS":
        db = FAISS.from_documents(chunks, emb)
        db.save_local(str(Path(folder) / "faiss"))
    else:
        db = Chroma.from_documents(chunks, emb, persist_directory=str(Path(folder) / "chroma"))
    return db

def load_store(emb, backend, folder):
    if backend == "FAISS":
        return FAISS.load_local(str(Path(folder) / "faiss"), emb, allow_dangerous_deserialization=True)
    return Chroma(persist_directory=str(Path(folder) / "chroma"), embedding_function=emb)

def context_text(docs):
    return "\n\n".join("[Source: {}, page {}]\n{}".format(
        d.metadata.get("source", "unknown"), d.metadata.get("page", "N/A"), d.page_content
    ) for d in docs)

def classify_evidence(model, question, context):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content='Classify if the supplied evidence supports the COMPLETE question. Return ONLY JSON: {"level":"enough|partial|insufficient","reason":"..."}. enough=all parts supported; partial=some supported; insufficient=no useful answer. Never use outside knowledge.'),
        ("human", "Question: {q}\nEvidence: {ctx}")
    ])
    response = model.invoke(prompt.format_messages(q=question, ctx=context))
    content = getattr(response, "content", None)
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        raw = "\n".join(
            item if isinstance(item, str) else item["text"]
            for item in content
            if isinstance(item, str) or (
                isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        )
    elif isinstance(content, dict) and isinstance(content.get("text"), str):
        raw = content["text"]
    else:
        raw = ""

    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            result, _ = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict):
            continue
        level = result.get("level")
        if isinstance(level, str) and level in ("enough", "partial", "insufficient"):
            reason = result.get("reason", "")
            return level, reason if isinstance(reason, str) else ""

    has_evidence = isinstance(context, str) and bool(context.strip())
    return ("partial" if has_evidence else "insufficient",
            "Conservative fallback: invalid classifier output.")

def generate(db, model, question, strategy, k, history="", query_planning=True):
    start = time.perf_counter()
    plan = question
    if query_planning:
        planner = ChatPromptTemplate.from_messages([
            ("system", "Break this request into numbered retrieval sub-questions. Do not answer."),
            ("human", "{q}")
        ])
        plan = model.invoke(planner.format_messages(q=question)).content
    docs = db.similarity_search(question + "\n" + plan, k=k)
    ctx = context_text(docs)
    level, reason = classify_evidence(model, question, ctx)
    if strategy == "Standard":
        instruction = "Answer using only retrieved context. If missing, say so."
    elif strategy == "Structured":
        instruction = "Use only evidence. Cite every fact [Source: filename, page N]. Separate supported facts and missing information. Never guess."
    else:
        instruction = {
            "enough": "Evidence is sufficient. Answer all parts directly and cite each factual claim [Source: filename, page N].",
            "partial": "Evidence is partial. Answer supported parts with citations and clearly identify missing information. Do not infer.",
            "insufficient": "Evidence is insufficient. Do not guess. Say: Not enough information is available in the provided university documents to answer this question."
        }[level]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an academic regulations assistant.\n{instruction}\nConversation memory (not a source of policy facts): {history}\nRetrieved evidence:\n{context}"),
        ("human", "{question}")
    ])
    answer = model.invoke(prompt.format_messages(instruction=instruction, history=history or "None",
                                                   context=ctx, question=question)).content
    return {"answer": answer, "docs": docs, "level": level, "reason": reason,
            "plan": plan, "latency": time.perf_counter()-start}

def buffer_memory(messages, turns=8):
    return "\n".join("{}: {}".format(m["role"], m["content"]) for m in messages[-2*turns:])

def summary_memory(model, messages, previous=""):
    transcript = "\n".join("{}: {}".format(m["role"], m["content"]) for m in messages)
    return model.invoke("Update this concise conversation summary without adding facts. Previous: {}\nConversation: {}".format(previous, transcript)).content

def entity_memory(messages):
    text = " ".join(m["content"] for m in messages)
    return sorted(set(re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b", text)))
