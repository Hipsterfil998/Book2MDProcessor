# Book Conversion & Metadata Extraction Pipeline

Pipeline for converting Italian and German books (PDF/EPUB) to structured Markdown, extracting bibliographic metadata, and evaluating conversion quality — all via vLLM inference.

---

## Project Structure

```
├── book_converter.py              # Main pipeline orchestrator
├── converters/
│   ├── text_extraction.py         # Rule-based PDF/EPUB → Markdown (no LLM)
│   ├── pdf2md_LLM.py              # PDF → Markdown via Qwen2.5-VL (vision LLM)
│   └── epub2md_LLM.py             # EPUB → Markdown via Qwen2.5 (text LLM)
├── metadata/
│   └── metadata_extractor.py      # Author/title/year/genre extraction
├── quality_evaluation/
│   └── evaluator.py               # LLM-as-judge faithfulness evaluation
├── dependency_parsing/
│   └── dependency_parsing.py      # Stanza-based dependency parsing
└── requirements.txt
```

---

## Setup on Google Colab

### 1. System dependencies

```bash
!apt-get install -y poppler-utils   # required by pdf2image (PDF rasterization)
!apt-get install -y pandoc          # required by pypandoc (EPUB conversion)
```

### 2. Python dependencies

```bash
!pip install -r requirements.txt
!pip install qwen-vl-utils          # required for Qwen2.5-VL image processing
```

### 3. GPU requirements

| Model | VRAM (bfloat16) | Recommended |
|---|---|---|
| Qwen2.5-VL-7B-Instruct (PDF) | ~16 GB | A100 / L4 |
| Qwen2.5-7B-Instruct (EPUB + metadata) | ~14 GB | T4 (tight) |

Both models are never loaded simultaneously — the pipeline loads one at a time.

---

## Usage

### Convert books (PDF and/or EPUB)

```python
from book_converter import ConverterPipeline

pipeline = ConverterPipeline(
    input_dir="books/",
    output_dir="output/",
    # pdf_model_id="Qwen/Qwen2.5-VL-7B-Instruct",   # default
    # text_model_id="Qwen/Qwen2.5-7B-Instruct",      # default
)

pipeline.run_pdf_llm()    # converts all .pdf files
pipeline.run_epub_llm()   # converts all .epub files
# pipeline.run_simple()   # rule-based fallback, no LLM
```

### Extract metadata

```python
from metadata.metadata_extractor import MetadataExtractor

extractor = MetadataExtractor()
extractor.run(
    output_dir="output/",          # folder with converted book subfolders
    output_csv="metadata/metadata.csv"
)
```

Extracts per book:
- **Author, title, year** — from front pages (title page, TOC, preface)
- **Genre** — from body pages (main content)

Genres: `Journalistic`, `Functional/Gebrauchstexte`, `Factual/Non fiction/Wissenschaft`, `Fiction/Belletristik`

### Evaluate conversion quality

```python
from quality_evaluation.evaluator import QualityEvaluator

evaluator = QualityEvaluator(judge_model_id="Qwen/Qwen2.5-VL-7B-Instruct")

# PDF
evaluator.evaluate_pdf(
    eval_pages_dir="output/book_name/eval_pages/",
    scores_dir="scores/"
)

# EPUB
evaluator.evaluate_epub(
    eval_chunks_dir="output/book_name/eval_chunks/",
    scores_dir="scores/"
)
```

The judge receives the original page image (PDF) or HTML chunk (EPUB) paired with the generated Markdown, and rates **faithfulness** — not general quality. Any VL model can be used for PDF evaluation; a text model suffices for EPUB.

---

## Output Structure

After conversion, each book produces a self-contained subfolder:

```
output/
└── book_name/
    ├── book_name.md          # full Markdown output
    ├── images/               # embedded images extracted from the source
    ├── eval_pages/           # PDF only: sampled page pairs for evaluation
    │   ├── 0.png             #   original page image
    │   ├── 0.md              #   corresponding generated Markdown
    │   ├── 12.png
    │   ├── 12.md
    │   └── ...
    └── eval_chunks/          # EPUB only: sampled chunk pairs for evaluation
        ├── 0.html            #   original HTML chunk
        ├── 0.md              #   corresponding generated Markdown
        └── ...
```

Evaluation scores are saved separately:

```
scores/
└── book_name_scores.json     # {"average": {"text": 4.2, "structure": 3.8, ...}, "pages": {...}}
```

Metadata output:

```
metadata/
└── metadata.csv              # columns: book, author, title, year, genre
```

### Dependency parsing

```python
from dependency_parsing.dependency_parsing import DependencyParser

parser = DependencyParser(langs=["it", "de"], output_format="conllu")
parser.run(input_dir="output/", output_dir="parsed/")
```

Runs full NLP annotation on all `.md` files in the output folder:
- tokenization, POS tagging, lemmatization, dependency parsing (via Stanza)
- Markdown is stripped to plain text before parsing
- Supports Italian (`it`) and German (`de`) simultaneously
- Output formats: `conllu` (CoNLL-U), `json`, or `both`

Each token in the output carries: `id`, `text`, `lemma`, `upos`, `xpos`, `feats`, `head`, `deprel`.

Stanza models are downloaded automatically on first run (~500 MB per language).

---

## Markdown Format

- **PDF**: pages separated by `\n\n---\n\n`, each with a `<!-- Page N -->` header
- **EPUB**: chunks separated by `\n\n---\n\n`

---

## Evaluation Details

During conversion, `eval_n=20` pages/chunks are sampled using stratified sampling:

| Zone | Pages sampled | Content |
|---|---|---|
| Front (first 10%) | ~3 | Title page, TOC, preface |
| Body (middle 80%) | ~14 | Main text, tables, formulas |
| Back (last 10%) | ~3 | Bibliography, index, appendices |

The same sampled pages are reused for both metadata extraction (front → biblio, body → genre) and quality evaluation — no reprocessing of original files required.
