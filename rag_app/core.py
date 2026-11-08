
import os
import re
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your local .env "
            "file or Streamlit Cloud Secrets."
        )

    return ChatGroq(
        model=MODEL,
        api_key=api_key,
        temperature=0
    )


def ingest(paths):
    docs = []

    for path in paths:
        p = Path(path)

        if p.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(p)).load()
        else:
            loaded = TextLoader(
                str(p),
                encoding="utf-8"
            ).load()

        for d in loaded:
            d.metadata["source"] = p.name

            if "page" in d.metadata:
                d.metadata["page"] = int(d.metadata["page"]) + 1
            else:
                d.metadata["page"] = "N/A"

        docs.extend(loaded)

    return docs


def split_docs(docs, kind, size, overlap, emb):

    if kind == "Fixed-size":
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=size,
            chunk_overlap=overlap
        )

    elif kind == "Recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    else:
        splitter = SemanticChunker(
            emb,
            breakpoint_threshold_type="percentile"
        )

    return splitter.split_documents(docs)


def make_store(chunks, emb, backend, folder):
    Path(folder).mkdir(parents=True, exist_ok=True)

    if backend == "FAISS":
        db = FAISS.from_documents(chunks, emb)
        db.save_local(str(Path(folder) / "faiss"))

    else:
        db = Chroma.from_documents(
            chunks,
            emb,
            persist_directory=str(Path(folder) / "chroma")
        )

    return db


def load_store(emb, backend, folder):

    if backend == "FAISS":
        return FAISS.load_local(
            str(Path(folder) / "faiss"),
            emb,
            allow_dangerous_deserialization=True
        )

    return Chroma(
        persist_directory=str(Path(folder) / "chroma"),
        embedding_function=emb
    )


def context_text(docs):
    return "\n\n".join(
        "[Source: {}, page {}]\n{}".format(
            d.metadata.get("source", "unknown"),
            d.metadata.get("page", "N/A"),
            d.page_content
        )
        for d in docs
    )


def classify_evidence(model, question, context):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            'Classify whether the supplied evidence supports the '
            'COMPLETE question. Return ONLY JSON: '
            '{{"level":"enough|partial|insufficient",'
            '"reason":"..."}}. '
            'enough means all parts are supported; '
            'partial means some parts are supported; '
            'insufficient means no useful answer is supported. '
            'Never use outside knowledge.'
        ),
        (
            "human",
            "Question: {q}\nEvidence: {ctx}"
        )
    ])

    response = model.invoke(
        prompt.format_messages(
            q=question,
            ctx=context
        )
    )

    raw = str(response.content).strip()

    try:
        match = re.search(r"\{.*\}", raw, re.S)

        if match:
            result = json.loads(match.group())

            level = result.get("level")
            reason = result.get("reason", "")

            if level in ("enough", "partial", "insufficient"):
                return level, str(reason)

    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    has_evidence = bool(context and context.strip())

    return (
        "partial" if has_evidence else "insufficient",
        "Conservative fallback: invalid classifier output."
    )


def generate(
    db,
    model,
    question,
    strategy,
    k,
    history="",
    query_planning=True
):

    start = time.perf_counter()

    plan = question

    # Multi-part query planning
    if query_planning:

        planner = ChatPromptTemplate.from_messages([
            (
                "system",
                "Break this request into numbered retrieval "
                "sub-questions. Do not answer."
            ),
            ("human", "{q}")
        ])

        plan_response = model.invoke(
            planner.format_messages(q=question)
        )

        plan = str(plan_response.content)

    # Retrieve relevant documents
    docs = db.similarity_search(
        question + "\n" + plan,
        k=k
    )

    ctx = context_text(docs)

    # Evidence sufficiency classification
    level, reason = classify_evidence(
        model,
        question,
        ctx
    )

    # Select prompting strategy
    if strategy == "Standard":

        instruction = (
            "Answer using only retrieved context. "
            "If information is missing, clearly say so."
        )

    elif strategy == "Structured":

        instruction = (
            "Use only retrieved evidence. "
            "Cite every factual claim using "
            "[Source: filename, page N]. "
            "Separate supported facts and missing information. "
            "Never guess."
        )

    else:

        instruction = {
            "enough": (
                "Evidence is sufficient. Answer all parts directly "
                "and cite each factual claim "
                "[Source: filename, page N]."
            ),

            "partial": (
                "Evidence is partial. Answer only supported parts "
                "with citations and clearly identify missing "
                "information. Do not infer."
            ),

            "insufficient": (
                "Evidence is insufficient. Do not guess. Say: "
                "Not enough information is available in the "
                "provided university documents to answer this question."
            )
        }[level]

    # Final answer generation
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an academic regulations assistant.\n"
            "{instruction}\n"
            "Conversation memory (not a source of policy facts): "
            "{history}\n"
            "Retrieved evidence:\n{context}"
        ),
        ("human", "{question}")
    ])

    answer_response = model.invoke(
        prompt.format_messages(
            instruction=instruction,
            history=history or "None",
            context=ctx,
            question=question
        )
    )

    answer = str(answer_response.content)

    return {
        "answer": answer,
        "docs": docs,
        "level": level,
        "reason": reason,
        "plan": plan,
        "latency": time.perf_counter() - start
    }


def buffer_memory(messages, turns=8):
    return "\n".join(
        "{}: {}".format(m["role"], m["content"])
        for m in messages[-2 * turns:]
    )


def summary_memory(model, messages, previous=""):

    transcript = "\n".join(
        "{}: {}".format(m["role"], m["content"])
        for m in messages
    )

    return model.invoke(
        "Update this concise conversation summary without "
        "adding facts. Previous: {}\nConversation: {}".format(
            previous,
            transcript
        )
    ).content


def entity_memory(messages):

    text = " ".join(
        m["content"] for m in messages
    )

    return sorted(
        set(
            re.findall(
                r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b",
                text
            )
        )
    )
