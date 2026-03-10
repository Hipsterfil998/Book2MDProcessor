# Codebase Documentation: Book Conversion Pipeline

## Overview

The pipeline converts books in PDF and EPUB format into structured Markdown, extracts bibliographic and genre metadata, evaluates conversion quality, and produces dependency parsing annotations. Everything is orchestrated by a single central class (`ConverterPipeline`) and configured through a single `config.py` file.

---

## Project Structure

```
.
├── config.py                        # Central configuration
├── book_converter.py                # Pipeline orchestrator
├── utils.py                         # Shared utilities
│
├── converters/
│   ├── text_extraction.py           # Rule-based conversion (no LLM)
│   ├── pdf2md_LLM.py                # PDF to Markdown via Qwen2.5-VL
│   └── epub2md_LLM.py               # EPUB to Markdown via Qwen2.5
│
├── metadata/
│   └── metadata_extractor.py        # Author/title/year/genre extraction
│
├── quality_evaluation/
│   └── evaluator.py                 # LLM-as-judge faithfulness scoring
│
├── dependency_parsing/
│   └── dependency_parsing.py        # Linguistic annotation with Stanza
│
├── books/                           # Input: original PDF and EPUB files
├── output/                          # Output: Markdown + eval pages/chunks
├── scores/                          # Per-book score JSON files
│
└── tests/
    ├── test_book_converter.py
    ├── test_metadata.py
    └── test_utils.py
```

---

## Pipeline Flow

```
books/                     output/{stem}/
  *.pdf  --> PDFConverter --> {stem}.md
                          --> eval_pages/{i}.png + {i}.md
                          --> images/
  *.epub --> EPUBConverter --> {stem}.md
                           --> eval_chunks/{i}.html + {i}.md

output/                    scores/
  **/eval_pages/  --> QualityEvaluator --> {book}_scores.json
  **/eval_chunks/

output/                    metadata/metadata.csv
  **/eval_pages/  --> MetadataExtractor --> author, title, year, genre
  **/eval_chunks/

output/
  **/*.md --> DependencyParser --> parsed_output/*.conllu / *.json
```

---

## `config.py`: Central Configuration

All parametric values (models, prompts, paths, token limits) are defined here. There are no magic constants scattered across the codebase.

**Models used:**
- `PDF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"`: vision-language model for PDF (reads page images)
- `TEXT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"`: text-only model for EPUB and metadata

**Why Qwen2.5?** It is open-weight, available on HuggingFace, and has strong support for Italian and German (the corpus languages). The VL (Vision-Language) variant is required for PDF because the pipeline rasterizes pages to PNG rather than using direct text extraction, which preserves layout, formulas, and images.

**`ENABLE_PREFIX_CACHING = True`:** enables vLLM Automatic Prefix Caching (APC). Since all books in a batch share the same system prompt, vLLM can cache it in the KV states and avoid recomputing it for every request. The system prompts (`BIBLIO_PROMPT`, `GENRE_PROMPT`, `PDF_PROMPT`) are kept as constants specifically to maximize the cache hit rate.

**`_StderrFilter`:** filters non-fatal protobuf/grpc error messages that vLLM emits on stderr, which would otherwise pollute production logs.

---

## `utils.py`: Shared Utilities

### `pil_to_data_url(img)`
Encodes a PIL image as a base64 PNG data URL (`data:image/png;base64,...`). Used to pass PNG pages to the vision-language model through vLLM, which accepts the OpenAI-compatible multimodal format.

### `sample_indices(total, n=20)`
Selects `n` page indices for evaluation using a hybrid strategy:

1. **Guaranteed:** the first `min(10, n)` pages are always included, because they contain the title page, colophon, and metadata.
2. **Stratified:** the remaining slots are filled by random sampling from the body (~75%) and the back (~25%) of the document.

**Why stratified instead of uniform?** Pure uniform sampling over-represents the body and ignores the final pages (indexes, colophons). The 75/25 body/back ratio empirically covers typographic variability better.

### `suppress_worker_stderr()`
A context manager that intercepts file descriptor 2 at the OS level (not just `sys.stderr`) to filter protobuf noise. vLLM worker processes are forked and inherit fd 2 directly, so a Python-level wrapper would not be sufficient.

---

## `book_converter.py`: Pipeline Orchestrator

`ConverterPipeline` exposes three conversion methods:

| Method | Converter used | When to use |
|--------|---------------|-------------|
| `run_simple()` | `DocumentProcessor` (rule-based) | Quick draft, no GPU needed |
| `run_pdf_llm()` | `PDFToMarkdownConverter` | PDFs with complex layout |
| `run_epub_llm()` | `EpubToMarkdownConverter` | EPUB files |

**Resume and idempotency:** `_already_converted(stem)` checks whether `output/{stem}/{stem}.md` already exists. If it does, the book is skipped. This allows the pipeline to be interrupted and resumed without reprocessing already-converted books, which is essential when working with a corpus of hundreds of books and LLM inference is expensive.

---

## `converters/text_extraction.py`: Rule-Based Conversion

`DocumentProcessor` converts documents without an LLM using deterministic rules.

**PDF:** uses PyMuPDF (`fitz`) to extract text in `rawdict` mode, which exposes font size and flags for every span. The mapping rules are:
- font size >= 22: `# H1`, >= 18: `## H2`, >= 14: `### H3`, >= 12 + bold: `#### H4`
- flags & 16 = bold: `**text**`, flags & 2 = italic: `*text*`
- font size < 9 and starts with `\d*†‡§`: footnote `> [^fn]: ...`
- matches `^(figura|fig|tabella|tab)`: caption `*text*`
- images extracted to `images/` and inserted as `![caption](images/filename)`

**EPUB:** uses `ebooklib` to extract HTML chapters and a recursive HTML-to-Markdown walker that handles: headings, bold/italic, code blocks, ordered/unordered lists, GFM tables, footnotes (`epub:type="footnote"`), figures with captions, blockquotes, and callout boxes (detected by CSS class: `callout|note|warning|tip`).

**Limitation:** this rule-based approach fails on scanned PDFs, complex LaTeX formulas, and multi-column layouts, which motivates the LLM-based converters.

---

## `converters/pdf2md_LLM.py`: PDF to Markdown via LLM

`PDFToMarkdownConverter` uses Qwen2.5-VL (vision-language):

1. Rasterizes each PDF page to PNG at `PDF_DPI=300` dpi
2. Encodes each PNG as a base64 data URL
3. Builds a batch of multimodal messages: `[{"type": "image_url", ...}, {"type": "text", "text": PDF_PROMPT}]`
4. Runs batch inference with `LLM.chat()`, processing all pages of a book in a single call
5. Saves the resulting Markdown
6. Saves sampled pages to `eval_pages/{i}.png` and `eval_pages/{i}.md` for quality evaluation and metadata extraction

**Why rasterize instead of extracting text?** Direct text extraction loses visual structure (columns, tables, equations). Rasterization lets the model see the page exactly as a human reader would.

**Batching:** the whole pipeline benefits from vLLM's native batching, which runs parallel inference on all pages in a single forward pass (continuous batching). This is far more efficient than processing pages one at a time.

---

## `converters/epub2md_LLM.py`: EPUB to Markdown via LLM

`EpubToMarkdownConverter` uses Qwen2.5 (text-only):

1. Converts the EPUB to HTML using `pypandoc`
2. Extracts images to `images/` and rewrites `<img>` `src` attributes to local paths
3. Splits the HTML into chunks of at most `EPUB_MAX_CHUNK_CHARS=8000` characters, splitting at top-level block tags (`section`, `article`, `div`)
4. Runs batch inference on all chunks with `EPUB_PROMPT`
5. Joins chunks with `\n\n---\n\n`
6. Saves sampled chunks to `eval_chunks/{i}.html` and `eval_chunks/{i}.md`

**Why chunking?** Models have a limited context window. Splitting at top-level tags ensures each chunk is semantically coherent (a complete HTML block, never cut mid-sentence).

**Difference from `run_simple`:** the LLM handles complex elements better, such as irregular tables, nested lists, and footnotes with cross-references.

---

## `metadata/metadata_extractor.py`: Metadata Extraction

`MetadataExtractor` extracts five fields: author, title, year, and genre, using zero-shot prompting on the same text model.

### `collect_samples(output_dir)`
For each book, it looks for `eval_pages/` (preferred) or `eval_chunks/` and selects:
- `front_files = guaranteed[:5]`: pages 0-4 (title page, colophon, preface) which contain author, title, and year
- `body_files = guaranteed[5:][-3:]`: pages 7-9 of the first 10, which are past the front matter but still early in the book, giving a representative sample for genre classification

**Why these pages?** The first 5 pages of a book almost always contain bibliographic information. Pages 7-9 are past the front matter and represent the tone and style of the text without being overly specialized. Since all of them fall within the guaranteed first-10 set, they are always available regardless of book length.

### `run(output_dir, output_csv)`
Runs two separate batch inference calls:
1. `BIBLIO_PROMPT` + `"Book filename: {name}\n\n{front_text}"` produces JSON `{author, title, year}`
2. `GENRE_PROMPT` + `body_text` produces JSON `{genre}`

**CSV resume:** if `output_csv` already exists, the existing records are loaded, already-processed books are filtered out, and only new records are appended. If there is nothing new to process, the method returns early without rewriting the file (preserving `mtime`).

**Why two separate batches?** The two prompts are very different (bibliographic vs genre) and require different input texts. Keeping them separate maximizes the prefix cache hit rate: within each batch, all messages share the same constant system prompt.

**`_parse_json(raw)`:** handles imperfect model output. It first attempts a direct `json.loads()`. On failure, it attempts recovery by taking the text before the first blank line (models sometimes add text after the JSON) and appending a missing `}` if needed.

---

## `quality_evaluation/evaluator.py`: Quality Evaluation

`QualityEvaluator` implements the **LLM-as-judge** pattern: a language model evaluates the output of another model.

### `evaluate_pdf(eval_pages_dir, scores_dir)`
For each `{i}.png` and `{i}.md` pair:
- Passes the original page image and the generated Markdown to the judge model
- The judge scores faithfulness on three dimensions: `text` (1-5), `structure` (1-5), `math` (1-5)
- Output: JSON with per-page scores and averages

### `evaluate_epub(eval_chunks_dir, scores_dir)`
For each `{i}.html` and `{i}.md` pair:
- Scores faithfulness on two dimensions: `text` and `structure` (no `math` since HTML does not contain formulas)

**Advantage of the eval-per-book design:** evaluation does not require access to the original files. It only uses the source/output pairs saved during conversion, which decouples the evaluation stage from the conversion stage.

**`PDF_JUDGE_PROMPT`:** instructs the judge to evaluate only *faithfulness* (is every element in the Markdown traceable to the original?) rather than general Markdown quality, avoiding bias toward specific formatting styles.

---

## `dependency_parsing/dependency_parsing.py`: Linguistic Annotation

`DependencyParser` applies morphosyntactic dependency analysis to the main Markdown file of each converted book.

**Flow:**
1. `run()` finds one `.md` file per book by matching `output/{stem}/{stem}.md` (the pattern `p.stem == p.parent.name`), ignoring eval pages and eval chunks
2. Markdown is converted to plain text (using the `markdown` library and BeautifulSoup to strip markup)
3. Language is detected automatically from the first 3000 characters of the text using `langdetect`; if the detected language is not among the configured ones, it falls back to the first configured language
4. Only the matching Stanza pipeline runs: tokenize + MWT + POS + lemma + depparse
5. Output is written as a single `{stem}.conllu` and/or `{stem}.json` file, without any language suffix

**Why automatic language detection?** Books in the corpus are either Italian or German. Running both pipelines on every book would double the computation and produce redundant output. Detecting the language per book and routing to the correct pipeline keeps the output clean: one file per book, annotated in the right language.

**`DetectorFactory.seed = 0`:** `langdetect` is non-deterministic by default (it uses a random seed internally). Setting a fixed seed makes language detection reproducible across runs.

**Lazy loading:** Stanza pipelines are loaded only on the first `run()` call, not in `__init__`. All configured languages are loaded upfront so the correct pipeline is ready regardless of the detection result. If a model has not been downloaded yet, it is downloaded automatically.

**Role in the project:** parsing is the final stage of the pipeline. It produces structured linguistic data (CoNLL-U) that can be used for corpus analysis, NLP model training, or linguistic research on the Italian and German text corpus.

---

## Key Architectural Decisions

| Decision | Discarded alternative | Rationale |
|----------|-----------------------|-----------|
| vLLM batching for the entire book | Page-by-page inference | Much higher throughput via continuous batching |
| PDF rasterization to PNG | PyMuPDF text extraction | Preserves visual layout, formulas, and complex tables |
| EPUB chunking by HTML tag | Fixed-size text windows | Semantically coherent chunks, no text split mid-sentence |
| Constant system prompts | Prompts with inline variables | Maximizes prefix cache hit rate with vLLM APC |
| Eval pages saved during conversion | Re-reading original files for eval | Decouples stages; evaluation works without the originals |
| Resume based on `.md` file existence | Flags in a database or JSON | Zero overhead; survives crashes and interruptions |
| Two separate LLMs (VL + text) | A single multimodal model | The text model is faster and uses less VRAM for EPUB and metadata |
| Automatic language detection per book | Running all pipelines on every book | Avoids redundant computation; produces one correctly-annotated file per book |

---

## Tests

Tests use pytest's `tmp_path` fixture for complete isolation (no global state). LLM models are replaced by stubs (`FakeLLM`) that return fixed JSON responses, so all tests run without a GPU.

- **`test_book_converter.py`:** tests the resume logic (`_already_converted`) and correct skipping across all three run methods
- **`test_metadata.py`:** tests `_parse_json` (robust parsing), `collect_samples` (page selection logic), and `run` (CSV append/skip behavior)
- **`test_utils.py`:** tests `sample_indices` (first-10 guarantee, stratification) and `pil_to_data_url` (PNG round-trip)
