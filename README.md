# IndusDocs — PDF-first RAG with Tavily web-search fallback

IndusDocs answers from local technical PDFs first. If the PDFs do not provide enough evidence, it searches the web with Tavily. Groq is used for generating fast, concise answers from the retrieved evidence. If Groq is unavailable, the app still displays relevant PDF passages or raw web search snippets and URLs.

## Project structure

```text
indusdocs_pdf_first_rag/
├── app.py
├── requirements.txt
├── .env.example
└── data/
    └── put_your_pdfs_here.pdf
```

## What changed in this version

- **No Sentence Transformers, PyTorch, FAISS, or NumPy.** Local retrieval uses a lightweight BM25-style ranker written with Python's standard library, avoiding the PyTorch Windows Application Control issue and embedding-model downloads.
- **PDFs are searched first.** The app extracts selectable text, chunks it, and preserves filename/page metadata.
- **Separate web fallback.** Tavily Search API is called only when PDF evidence is insufficient.
- **Graceful Groq quota handling.** If Groq cannot answer because of connectivity or limits, the app shows PDF excerpts or the original Tavily search snippets and URLs instead of stopping.
- **Groq API key is optional for basic retrieval.** To get generated answers, configure a working key.
- Python 3.11 or newer (the code avoids heavy ML dependencies).
- A Groq API key for generated answers (free from https://console.groq.com/).
- A Tavily API key for web fallback (optional, but required for web search).
- Internet access for API calls.

## Setup on Windows / VS Code

Open the folder containing `app.py` in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, call its Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Configure `.env`

1. Copy `.env.example` and rename the copy to `.env` in the same folder as `app.py`.
2. Add your Gemini key after `GEMINI_API_KEY=` if you want generated answers.
3. Add your Tavily key after `TAVILY_API_KEY=` to enable web fallback.
4. Never upload `.env` or expose either key on GitHub.

Get a Tavily API key through the official Tavily website: https://tavily.com/
Check the provider's current plan and usage limits before relying on it. This project does not bypass any provider quotas.

Example configuration:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
TAVILY_API_KEY=your_actual_tavily_api_key
```

If you leave `GROQ_API_KEY` blank, the app can still retrieve PDF passages and display web search snippets, but it will not generate a synthesized answer. If `TAVILY_API_KEY` is blank, web fallback is disabled.

## Running the Web UI

To test the assistant in a browser-based chat interface:

```powershell
python ui.py
```

This will automatically open `http://localhost:8000` in your browser. It includes:
- A sidebar showing all indexed PDF manuals.
- Clickable sample question pills.
- Clean markdown formatted chat bubbles.
- Zero extra pip dependencies needed (uses Python's standard library).

---

## Running in Terminal (CLI Mode)

```powershell
python app.py
```

Example questions:

```text
What is the maximum operating pressure stated in the pump manual?
What is the recommended maintenance interval?
What is the difference between model ABC-100 and ABC-200?
```

Type `exit` to stop.

## Workflow

1. Read PDF files under `data/` and extract page text with `pypdf`.
2. Split extracted text into overlapping chunks, preserving PDF filename and page number.
3. Tokenize each chunk and build local term-frequency statistics.
4. Rank chunks for each question with a BM25-style lexical score.
5. If there is relevant PDF evidence and Gemini is available, ask Gemini to answer using only the passages and cite PDF pages.
6. If PDF evidence is insufficient, call Tavily Search.
7. If Gemini can summarize the search results, show the synthesized answer with URLs. If Gemini is out of quota or unavailable, show the original search snippets and URLs.

## Tuning and limitations

- `MIN_BM25_SCORE=1.0` is a starting point, not a universal relevance threshold. Tune it using questions that should and should not be answerable from your manuals.
- Lexical retrieval is lightweight but does not understand synonyms as well as semantic embeddings. Include relevant terminology in your PDFs and questions; consider adding a local semantic model later if your machine and environment support it.
- Scanned/image-only PDFs may have no selectable text and require OCR.
- Search snippets are not a substitute for checking primary technical documentation. Verify safety-related or specification-critical claims against the original source.
- API keys, model availability, and provider quotas are controlled by their respective providers and may change.
