
import os
import re
import sys
import json
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import numpy as np
from openai import OpenAI


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(BASE_DIR, "book.pdf")
META_FILE = PDF_FILE + ".meta.json"    # texts + pages + build info
VEC_FILE = PDF_FILE + ".vectors.npy"   # embeddings
OCR_CACHE = PDF_FILE + ".ocr.json"      # OCR results (only used when building)
KEY_FILE = os.path.join(BASE_DIR, "api_key.txt")

BASE_URL = "https://api.gapgpt.app/v1"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
CHUNK_SIZE = 1000
OVERLAP = 200
TOP_K = 5
MIN_SCORE = 0.25
BATCH = 64
HISTORY_TURNS = 3

MIN_PAGE_CHARS = 50
OCR_LANG = "eng"
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


SIGNATURE = {"embed_model": EMBED_MODEL, "chunk_size": CHUNK_SIZE, "overlap": OVERLAP, "ocr": 1}

client = None  # created in main()



def with_retry(fn, tries=4):
    for n in range(tries):
        try:
            return fn()
        except Exception as e:
            if n == tries - 1:
                raise
            wait = 2 ** n
            print(f"API error ({type(e).__name__}), retrying in {wait}s...")
            time.sleep(wait)


def embed(texts):
    res = with_retry(lambda: client.embeddings.create(model=EMBED_MODEL, input=texts))
    return [d.embedding for d in res.data]


def get_api_key(root):
    key = os.environ.get("GAPGPT_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    key = simpledialog.askstring("API key", "Enter your GapGPT API key:", show="*", parent=root)
    if not key or not key.strip():
        raise SystemExit("No API key.")
    with open(KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key.strip())
    return key.strip()


# ---------- Build database (only needed the first time) ----------
def save_json(path, data):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def ocr_page(page):
    import io
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise SystemExit("Run: pip install pytesseract pillow")
    if os.path.exists(TESSERACT_EXE):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img, lang=OCR_LANG)
    except pytesseract.TesseractNotFoundError:
        raise SystemExit("Tesseract is not installed. Install it from "
                         "https://github.com/UB-Mannheim/tesseract/wiki")


def read_pdf():
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # pip install pymupdf
    doc = fitz.open(PDF_FILE)
    cache = {}
    if os.path.exists(OCR_CACHE):
        try:
            with open(OCR_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}

    pages, new = [], 0
    for i, page in enumerate(doc):
        text = page.get_text()
        if len(text.strip()) < MIN_PAGE_CHARS:
            key = str(i + 1)
            if key not in cache:
                print(f"OCR page {i + 1}/{len(doc)}")
                cache[key] = ocr_page(page)
                new += 1
                if new % 10 == 0:
                    save_json(OCR_CACHE, cache)
            text = cache[key]
        pages.append((i + 1, text))
    if new:
        save_json(OCR_CACHE, cache)
    return pages


SENT_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n\s*\n")


def split_text(text):
    pieces = []
    for s in SENT_SPLIT.split(text):
        s = " ".join(s.split())
        while len(s) > CHUNK_SIZE:
            pieces.append(s[:CHUNK_SIZE])
            s = s[CHUNK_SIZE:]
        if s:
            pieces.append(s)

    chunks, cur, last = [], "", ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > CHUNK_SIZE:
            chunks.append(cur)
            cur = last if len(last) <= OVERLAP else ""
        cur = (cur + " " + p).strip()
        last = p
    if cur:
        chunks.append(cur)
    return chunks


def build_db():
    pages = read_pdf()
    total_len = sum(len(t) for _, t in pages)
    print("Pages:", len(pages), "| Text length:", total_len)
    if total_len < 100:
        raise SystemExit("PDF has no readable text.")

    texts, page_nums = [], []
    for num, page_text in pages:
        for chunk in split_text(page_text):
            texts.append(chunk)
            page_nums.append(num)
    print("Chunks:", len(texts))

    vectors = []
    for i in range(0, len(texts), BATCH):
        vectors.extend(embed(texts[i:i + BATCH]))
        print(f"{min(i + BATCH, len(texts))}/{len(texts)}")

    with open(VEC_FILE + ".tmp", "wb") as f:
        np.save(f, np.array(vectors, dtype=np.float32))
    os.replace(VEC_FILE + ".tmp", VEC_FILE)

    meta = {"signature": SIGNATURE, "texts": texts, "pages": page_nums}
    save_json(META_FILE, meta)


def db_is_valid():
    """The database is reusable even without the PDF, as long as the settings match."""
    if not (os.path.exists(META_FILE) and os.path.exists(VEC_FILE)):
        return False
    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            sig = json.load(f).get("signature", {})
        return all(sig.get(k) == v for k, v in SIGNATURE.items())
    except (json.JSONDecodeError, OSError):
        return False


def load_db():
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    matrix = np.load(VEC_FILE)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return meta["texts"], meta["pages"], matrix


# ---------- Search + answer ----------
def search(query, texts, pages, matrix):
    q = np.array(embed([query])[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = matrix @ q
    best = np.argsort(scores)[::-1][:TOP_K]
    return [(texts[i], pages[i], float(scores[i])) for i in best if scores[i] >= MIN_SCORE]


SYSTEM_PROMPT = """You are a friendly English study assistant helping a student with a textbook.
Use the textbook context first. If the answer is in the context, answer from it
and mention the page number(s), e.g. (p. 42).
If it is not, answer from your own knowledge and start with: [Not in the textbook]
Never claim something is from the textbook unless it appears in the context.
Reply in the language of the QUESTION itself, never in the language of the context.
Keep answers clear and well organised: short sections, simple wording."""

PAGE_RE = re.compile(r"\b(?:page|p\.?|صفحه)\s*(\d+)", re.IGNORECASE)


def ask(question, history, texts, pages, matrix):
    hits = []
    m = PAGE_RE.search(question)
    if m:  
        n = int(m.group(1))
        hits = [(t, p, 1.0) for t, p in zip(texts, pages) if p == n]
    if not hits:
        
        query = f"{history[-1][0]} {question}" if history and len(question) < 40 else question
        hits = search(query, texts, pages, matrix)

    if hits:
        context = "\n\n".join(f"[Page {p}]\n{t}" for t, p, _ in hits)
    else:
        context = "(no relevant passages found)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in history[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({
        "role": "user",
        "content": f"TEXTBOOK CONTEXT:\n{context}\n\nQUESTION:\n{question}",
    })

    res = with_retry(lambda: client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, messages=messages))
    return res.choices[0].message.content, sorted({p for _, p, _ in hits})


# ---------- Ready-made prompts (English) ----------
def prompt_teach_page(n):
    return (f"Teach me page {n} of the textbook. Go through the vocabulary and ideas on that "
            f"page step by step: for each word or phrase give a simple meaning and one example "
            f"sentence. Finish with a 3-question mini quiz (answers at the end).")


def prompt_quiz_page(n):
    return f"Make a 5-question quiz from page {n} of the textbook. Put the answers at the end."


def prompt_word_meaning(w):
    return (f"What does the word or phrase \"{w}\" mean? If it appears in the textbook, mention the "
            f"page. Give a simple definition, 2 example sentences, and common collocations "
            f"or related words.")


def prompt_word_sentences(w):
    return (f"Write 5 different example sentences using \"{w}\" (easy to hard), "
            f"and say what each sentence shows about its meaning.")


# ---------- Colors ----------
BG = "#f3f4f9"
HEADER = "#4338ca"
BLUE, BLUE_D = "#2563eb", "#1d4ed8"
GREEN, GREEN_D = "#059669", "#047857"
INDIGO, INDIGO_D = "#4f46e5", "#4338ca"
GRAY, GRAY_D = "#6b7280", "#4b5563"


def setup_style(root):
    root.configure(bg=BG)
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure(".", background=BG, font=("Segoe UI", 10))
    st.configure("TLabel", background=BG)
    st.configure("TFrame", background=BG)
    st.configure("TLabelframe", background=BG, bordercolor="#c7d2fe")
    st.configure("TLabelframe.Label", background=BG, foreground=HEADER,
                 font=("Segoe UI", 10, "bold"))
    st.configure("Status.TLabel", background=BG, foreground=GRAY)
    st.configure("TEntry", fieldbackground="white", padding=4)
    for name, base, dark in [("Page", BLUE, BLUE_D), ("Word", GREEN, GREEN_D),
                             ("Send", INDIGO, INDIGO_D), ("Clear", GRAY, GRAY_D)]:
        st.configure(f"{name}.TButton", background=base, foreground="white", borderwidth=0,
                     focusthickness=0, padding=(10, 6), font=("Segoe UI", 10, "bold"))
        st.map(f"{name}.TButton",
               background=[("disabled", "#c4c7d0"), ("active", dark)],
               foreground=[("disabled", "white")])


# ---------- GUI ----------
BG = "#f3f4f8"
CARD = "#ffffff"
ACCENT = "#4f46e5"
TEXT = "#1f2937"
MUTED = "#6b7280"
BORDER = "#e5e7eb"
FONT = "Segoe UI"


def render(widget, text):
    """Insert text; **bold** becomes real bold, markdown headers lose their #."""
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    for i, part in enumerate(text.split("**")):
        widget.insert("end", part, "bold" if i % 2 else "")


def darker(color, f=0.85):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))


def make_button(parent, text, color, command):
    hover = darker(color)
    btn = tk.Button(parent, text=text, command=command, bg=color, fg="white",
                    activebackground=hover, activeforeground="white", relief="flat",
                    bd=0, padx=12, pady=6, cursor="hand2", font=(FONT, 10, "bold"))
    btn.bind("<Enter>", lambda e: btn.config(bg=hover) if str(btn["state"]) == "normal" else None)
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn


def make_entry(parent, **kw):
    return tk.Entry(parent, font=(FONT, 11), relief="flat", bg="#f9fafb", fg=TEXT,
                    insertbackground=TEXT, highlightthickness=1,
                    highlightbackground="#d1d5db", highlightcolor=ACCENT, **kw)


class App:
    def __init__(self, root, texts, pages, matrix):
        self.root = root
        self.texts, self.pages, self.matrix = texts, pages, matrix
        self.history = []
        self.busy = False
        self.q = queue.Queue()

        root.title("Book Assistant")
        root.geometry("780x600")
        root.minsize(560, 440)
        root.configure(bg=BG)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        self._build_header()
        self._build_quick_panel()
        self._build_chat()
        self._build_input()
        self.root.after(100, self._poll)
        self._say("Ready. Pick a quick prompt above or type your own question.\n", "meta")

    # --- layout ---
    def _build_header(self):
        bar = tk.Frame(self.root, bg=ACCENT)
        bar.grid(row=0, column=0, sticky="ew")
        tk.Label(bar, text="Book Assistant", bg=ACCENT, fg="white",
                 font=(FONT, 15, "bold")).pack(side="left", padx=14, pady=8)
        tk.Label(bar, text="your English textbook helper", bg=ACCENT, fg="#c7d2fe",
                 font=(FONT, 10)).pack(side="left", pady=(6, 0))

    def _build_quick_panel(self):
        card = tk.Frame(self.root, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        card.grid(row=1, column=0, sticky="ew", padx=12, pady=(10, 4))
        card.columnconfigure(2, weight=1)
        card.columnconfigure(3, weight=1)

        tk.Label(card, text="QUICK PROMPTS", bg=CARD, fg=MUTED,
                 font=(FONT, 8, "bold")).grid(row=0, column=0, columnspan=4,
                                              sticky="w", padx=10, pady=(8, 2))
        self.page_var = tk.StringVar()
        self.word_var = tk.StringVar()

        tk.Label(card, text="Page", bg=CARD, fg=TEXT, font=(FONT, 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=(10, 8), pady=4)
        make_entry(card, textvariable=self.page_var, width=16).grid(
            row=1, column=1, padx=(0, 8), pady=4, ipady=4)
        make_button(card, "Teach this page", "#4f46e5",
                    lambda: self._page_action(prompt_teach_page, "Teach page")).grid(
            row=1, column=2, sticky="ew", padx=3, pady=4)
        make_button(card, "Quiz me", "#f59e0b",
                    lambda: self._page_action(prompt_quiz_page, "Quiz page")).grid(
            row=1, column=3, sticky="ew", padx=(3, 10), pady=4)

        tk.Label(card, text="Word", bg=CARD, fg=TEXT, font=(FONT, 10, "bold")).grid(
            row=2, column=0, sticky="w", padx=(10, 8), pady=(4, 10))
        make_entry(card, textvariable=self.word_var, width=16).grid(
            row=2, column=1, padx=(0, 8), pady=(4, 10), ipady=4)
        make_button(card, "Meaning of word", "#0d9488",
                    lambda: self._word_action(prompt_word_meaning, "Meaning of")).grid(
            row=2, column=2, sticky="ew", padx=3, pady=(4, 10))
        make_button(card, "Use in sentences", "#db2777",
                    lambda: self._word_action(prompt_word_sentences, "Sentences with")).grid(
            row=2, column=3, sticky="ew", padx=(3, 10), pady=(4, 10))

    def _build_chat(self):
        frame = tk.Frame(self.root, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.chat = tk.Text(frame, wrap="word", font=(FONT, 11), bg=CARD, fg=TEXT,
                            relief="flat", state="disabled", padx=12, pady=10, spacing3=3)
        sb = ttk.Scrollbar(frame, command=self.chat.yview)
        self.chat.configure(yscrollcommand=sb.set)
        self.chat.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self.chat.tag_config("you", foreground="white", background=ACCENT, font=(FONT, 9, "bold"))
        self.chat.tag_config("ask", foreground="#312e81", background="#eef2ff", font=(FONT, 11))
        self.chat.tag_config("bot", foreground="white", background="#059669", font=(FONT, 9, "bold"))
        self.chat.tag_config("bold", foreground="#111827", font=(FONT, 11, "bold"))
        self.chat.tag_config("meta", foreground=MUTED, font=(FONT, 9, "italic"))
        self.chat.tag_config("err", foreground="#b91c1c", background="#fee2e2")

    def _build_input(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 0))
        bar.columnconfigure(0, weight=1)

        self.entry = tk.Entry(bar, font=(FONT, 11), relief="flat", bg=CARD, fg=TEXT,
                              insertbackground=TEXT, highlightthickness=1,
                              highlightbackground="#d1d5db", highlightcolor=ACCENT)
        self.entry.grid(row=0, column=0, sticky="ew", ipady=7)
        self.entry.bind("<Return>", lambda e: self._send_typed())
        self.entry.focus_set()

        self.send_btn = make_button(bar, "Send", ACCENT, self._send_typed)
        self.send_btn.grid(row=0, column=1, padx=(6, 0), sticky="ns")
        make_button(bar, "Clear", "#6b7280", self._clear).grid(
            row=0, column=2, padx=(6, 0), sticky="ns")

        self.status = tk.Label(self.root, text="", bg=BG, fg=ACCENT, font=(FONT, 9, "italic"))
        self.status.grid(row=4, column=0, sticky="w", padx=14, pady=(2, 8))

    # --- actions ---
    def _page_action(self, make_prompt, label):
        n = self.page_var.get().strip()
        if not n.isdigit():
            self.status.config(text="Enter a page number first (digits only).")
            return
        self._send(make_prompt(int(n)), f"{label} {n}")

    def _word_action(self, make_prompt, label):
        w = self.word_var.get().strip()
        if not w:
            self.status.config(text="Enter a word or phrase first.")
            return
        self._send(make_prompt(w), f"{label} \"{w}\"")

    def _send_typed(self):
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            self._send(text, text)

    def _clear(self):
        self.history.clear()
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self._say("Conversation cleared.\n", "meta")

    def _send(self, prompt, shown):
        if self.busy:
            return
        self.busy = True
        self.send_btn.config(state="disabled")
        self.status.config(text="Thinking...")
        self._say("\n")
        self._say(" YOU ", "you")
        self._say("  " + shown + "\n", "ask")
        threading.Thread(target=self._worker, args=(prompt,), daemon=True).start()

    def _worker(self, prompt):
        try:
            answer, used = ask(prompt, self.history, self.texts, self.pages, self.matrix)
            self.q.put(("ok", prompt, answer, used))
        except Exception as e:
            self.q.put(("err", f"{type(e).__name__}: {e}"))

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "ok":
                    _, prompt, answer, used = item
                    self.history.append((prompt, answer))
                    self._say("\n")
                    self._say(" ASSISTANT ", "bot")
                    self._say("\n")
                    self.chat.config(state="normal")
                    render(self.chat, answer)
                    self.chat.insert("end", "\n")
                    self.chat.config(state="disabled")
                    if used:
                        self._say("Searched pages: " + ", ".join(map(str, used)) + "\n", "meta")
                else:
                    self._say(f"\n Error: {item[1]} \n", "err")
                self.busy = False
                self.send_btn.config(state="normal")
                self.status.config(text="")
                self.chat.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _say(self, text, tag=""):
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.config(state="disabled")
        self.chat.see("end")


# ---------- Main ----------
def main():
    global client
    try: 
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    root.withdraw()
    client = OpenAI(base_url=BASE_URL, api_key=get_api_key(root))

    if not db_is_valid():
        if not os.path.exists(PDF_FILE):
            messagebox.showerror("Missing files",
                                 f"Database not found and PDF is missing:\n{PDF_FILE}")
            return
        print("Building database (first run) - this can take a while...")
        build_db()

    texts, pages, matrix = load_db()
    root.deiconify()
    App(root, texts, pages, matrix)
    root.mainloop()


if __name__ == "__main__":
    main()
