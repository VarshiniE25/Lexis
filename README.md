# Contract Intelligence Engine

A production-ready system that converts unstructured contract PDFs into structured, machine-readable JSON using RAG (Retrieval-Augmented Generation).

## Project Structure

```
contract_intelligence/
├── README.md
├── requirements.txt
├── .env.example
├── app.py                    # Streamlit UI
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration & constants
│   ├── pdf_parser.py         # PDF parsing with PyMuPDF
│   ├── chunker.py            # Text chunking logic
│   ├── embeddings.py         # Embedding generation & FAISS index
│   ├── retriever.py          # RAG retrieval logic
│   ├── llm_client.py         # LLM API calls (GPT-4o-mini)
│   ├── prompts.py            # All prompts, centralized
│   ├── extractor.py          # Core extraction pipeline
│   ├── validator.py          # Second-pass LLM validation
│   ├── cache.py              # Caching layer
│   └── models.py             # Pydantic data models
├── batch/
│   ├── batch_processor.py    # Batch processing for 500+ contracts
│   └── metrics.py            # Extraction metrics computation
├── logs/
│   └── .gitkeep
└── sample_contracts/
    └── .gitkeep
```

## Setup & Installation

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

### 5. Run batch processing
```bash
python batch/batch_processor.py --input_dir ./sample_contracts --output_dir ./results
```

## Architecture

```
PDF → PyMuPDF Parser → Chunker (500-800 words, 100 overlap)
    → sentence-transformers Embeddings → FAISS Index
    → Per-clause RAG Retrieval (top-k chunks)
    → Parallel LLM Extraction (asyncio.gather)
    → LLM Validation Pass
    → Structured JSON Output
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `LLM_MODEL`: Model name (default: gpt-4o-mini)
- `TOP_K_CHUNKS`: Number of chunks to retrieve (default: 5)
- `CACHE_TTL`: Cache TTL in seconds (default: 3600)
