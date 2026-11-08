# Confidence-Aware Adaptive Prompt Engineering for RAG

This implementation follows the user's Review 1 proposal and the relevant PEGAI Jackfruit Problem Statement 3 (Retrieval-Augmented Academic Regulations Assistant), adding evidence-sufficiency-aware prompt selection as the research contribution.

## Implemented
- LangChain PDF/TXT ingestion with source and page metadata
- Fixed-size, recursive and semantic chunking
- FAISS and ChromaDB vector stores
- RAG generation with source citations
- Multi-part query planning
- Buffer, summary and entity memory options
- Standard, Structured and Adaptive prompt strategies
- JSONL evaluation runner with Recall@k source-file proxy, LLM-as-Judge correctness/faithfulness, sufficiency classification accuracy and latency
- Streamlit application

## Windows setup
1. Install Python 3.11/3.12 and Ollama (https://ollama.com).
2. Run `ollama pull llama3.1:8b`.
3. In PowerShell, from this folder:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
   Use the virtual-environment Python for both installation and launch; a globally installed `streamlit` command may use a different Python environment.
4. Optional: copy `.env.example` to `.env`.

## Use
Upload official university regulations/curriculum/exam policy PDFs, build an index, ask questions, and inspect citations and the evidence classification. Upload a JSONL file to run evaluation. The template contains three examples only; create and manually validate the required 50-question set.

Each JSONL row must contain: `id`, `question`, `expected_answer`, `expected_level` (`enough`, `partial`, `insufficient`) and `relevant_sources` (exact filenames).

## Fair comparison
Run Standard, Structured and Adaptive modes on the same corpus, model, chunking strategy, vector DB, retrieval k and 50 questions. Compare chunking methods and vector databases in separate controlled runs. Include answerable, partly answerable, unanswerable/out-of-scope and multi-part questions.

## Important limitations before submission
- Recall@k currently measures source-file overlap, not gold chunk-level Recall@k. Annotate relevant pages/chunks and upgrade this for rigorous evaluation.
- LLM-as-Judge requires manual spot-checking.
- Latency is recorded, but token counts are not. Add tokenizer/provider usage accounting and include the sufficiency-classifier call in total cost.
- Upload real official documents; none are fabricated or included.
- Compare both FAISS and ChromaDB empirically.
- The 50-question dataset must be manually authored and verified against your chosen official documents.
