# Book Assistant

A small desktop app (Python + Tkinter) that lets you chat with a PDF textbook using
Retrieval-Augmented Generation (RAG). Ask questions, get a page taught step by step,
look up word meanings, or generate quizzes. Answers cite page numbers.

## Features
- Works with scanned PDFs (automatic OCR with Tesseract, cached after the first run)
- Page-aware chunking: every answer shows which pages were used
- Quick prompts: *Teach this page*, *Quiz me*, *Meaning of word*, *Use in sentences*
- Conversation memory for follow-up questions
- Database is built once and reused (no PDF or OCR needed afterwards)

## Setup
```bash
pip install -r requirements.txt
```
Python 3.9+ (Tkinter is included with the Windows installer).

**OCR (only for scanned PDFs):** install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
(Windows: `winget install UB-Mannheim.TesseractOCR`).

## Usage
1. Put your PDF next to `book_rag_gui.py` and name it `book.pdf`
   (or run `python book_rag_gui.py path/to/other.pdf`).
2. Run:
   ```bash
   python book_rag_gui.py
   ```
3. On first run it asks for your API key (or set the `GAPGPT_API_KEY` environment variable)
   and builds the database. This can take a while for scanned books; it only happens once.

The app uses an OpenAI-compatible API (`base_url` and models are set at the top of the file).

## How it works
PDF -> (OCR if needed) -> sentence-aware chunks with page numbers -> embeddings
-> cosine-similarity search -> top chunks + question sent to the chat model.
Questions that mention a page ("page 22") use that page's text directly.

## Notes
- Do not commit your API key or any copyrighted book files; `.gitignore` already excludes them.
- Page numbers are PDF page indices, which may differ from the numbers printed in the book.
