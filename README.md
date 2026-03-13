<div align="center">

<img src="docs/logo.svg" alt="Book2MD Converter logo" width="320"/>

# Book2MD Converter

</div>

Pipeline for converting Italian and German books (PDF/EPUB) to structured Markdown, extracting bibliographic metadata, evaluating conversion quality, and annotating linguistic structure — all via vLLM inference.

> **Recommended GPU:** NVIDIA A100 or equivalent (Google Colab G4).

**[📖 Full Documentation](https://hipsterfil998.github.io/Book2MD-Converter/)**

---

## Project Structure

```
book2md/
├── __init__.py                    # Public API: from book2md import ConverterPipeline, ...
├── base.py                        # Abstract PipelineStep base class
├── config.py                      # Typed configuration dataclasses
├── utils.py                       # Shared utilities (image encoding, sampling, repetition filter)
├── pipeline.py                    # ConverterPipeline orchestrator
├── cli.py                         # CLI entry point
├── converters/
│   ├── pdf.py                     # PDF → Markdown via Qwen3-VL (vision LLM)
│   ├── epub.py                    # EPUB → Markdown via Qwen3 (text LLM)
│   └── text.py                    # Rule-based PDF/EPUB → Markdown (no LLM)
├── metadata/
│   └── extractor.py               # Author / title / year / genre extraction
├── parsing/
│   └── parser.py                  # Dependency parsing with Stanza
└── evaluation/
    └── evaluator.py               # Quality evaluation (NED, BLEU, MarkdownStructureF1)
pyproject.toml
requirements.txt
setup.sh
```

---

## Setup

Clone the repository once, then install in editable mode — `git pull` is enough for future updates, no reinstall needed.

### Local / server

```bash
git clone https://github.com/Hipsterfil998/Book2MD-Converter.git
cd Book2MD-Converter

# Install system + Python dependencies
bash setup.sh

# Also install evaluation libraries and clone Page2MDBench
bash setup.sh --with-eval

# Install the package (makes `book2md` available as a command)
pip install -e .
```

### Google Colab

```python
!git clone https://github.com/Hipsterfil998/Book2MD-Converter.git
%cd Book2MD-Converter

!bash setup.sh            # conversion only
!bash setup.sh --with-eval  # + evaluation

!pip install -e .
```

`setup.sh` installs `poppler-utils` and all Python dependencies. The `--with-eval` flag additionally clones [Page2MDBench](https://github.com/Hipsterfil998/Page2MDBench) and installs the evaluation libraries (`rapidfuzz`, `sacrebleu`, `mistune`, `bert-score`).

### GPU requirements (conversion only)

| Model | VRAM (bfloat16) | Recommended |
|---|---|---|
| Qwen3-VL-4B-Instruct (PDF) | ~10 GB | A100 / G4 |
| Qwen3-4B (EPUB + metadata) | ~8 GB | A100 / G4 |

Both models are never loaded simultaneously. Quality evaluation does **not** require a GPU.

---

## Configuration

Parameters are organised in typed dataclasses in `book2md/config.py`. There are three ways to customise them — no source code editing required for common use cases.

### Level 1 — CLI flags (directories, models, languages)

```bash
book2md --input /my/books/ --output /results/ convert --pdf
book2md --output /results/ --scores /scores/ evaluate
book2md parse --langs it de --format conllu
```

Run `book2md --help` or `book2md <subcommand> --help` for the full option list.

### Level 2 — Python / Colab (constructor arguments)

```python
from book2md import ConverterPipeline

pipeline = ConverterPipeline(
    input_dir="books/",
    output_dir="output/",
    pdf_model_id="Qwen/Qwen3-VL-7B-Instruct",   # override model
    text_model_id="Qwen/Qwen3-8B",
)
pipeline.run_pdf_llm()
```

### Level 3 — Config override (advanced: DPI, token limits, prompts)

```python
from book2md.config import pdf_config, epub_config

pdf_config.dpi = 150
pdf_config.max_new_tokens = 2048
epub_config.repetition_penalty = 1.2
```

Call this before instantiating any class; defaults apply otherwise.

---

## Usage

### Convert books

**CLI:**
```bash
book2md convert --pdf       # PDF → Markdown via Qwen3-VL
book2md convert --epub      # EPUB → Markdown via Qwen3
book2md convert --simple    # rule-based fallback, no LLM
```

**Python:**
```python
from book2md import ConverterPipeline

pipeline = ConverterPipeline()
pipeline.run_pdf_llm()
pipeline.run_epub_llm()
# pipeline.run_simple()  # no GPU required
```

**Resume support:** if `output/{book_name}/{book_name}.md` already exists, the book is automatically skipped. Re-running after an interruption resumes from where it left off.

---

### Extract metadata

**CLI:**
```bash
book2md metadata
book2md --csv results/metadata.csv metadata   # custom output path
```

**Python:**
```python
from book2md import MetadataExtractor

MetadataExtractor().run(output_dir="output/", output_csv="metadata/metadata.csv")
```

Extracts per book: **author, title, year** (from front pages) and **genre** (from body pages).

Genres: `Journalistic`, `Functional/Gebrauchstexte`, `Factual/Non fiction/Wissenschaft`, `Fiction/Belletristik`

**Resume support:** existing records in the CSV are never overwritten; only new books are appended.

---

### Evaluate conversion quality

Evaluation uses reference-based metrics from [Page2MDBench](https://github.com/Hipsterfil998/Page2MDBench). No LLM or GPU required.

**CLI:**
```bash
book2md evaluate
book2md evaluate --bertscore   # also compute BERTScore (slower)
```

**Python:**
```python
from book2md import QualityEvaluator

QualityEvaluator(use_bertscore=False).evaluate_all(
    output_dir="output/",
    scores_dir="scores/"
)
```

| Metric | Direction | Description |
|---|---|---|
| NED | lower is better | Normalised Edit Distance |
| BLEU | higher is better | n-gram precision |
| MarkdownStructureF1 | higher is better | Structural element overlap |
| BERTScore | higher is better | Semantic similarity (optional, slow on CPU) |

---

### Dependency parsing

**CLI:**
```bash
book2md parse
book2md parse --langs it de --format conllu
```

**Python:**
```python
from book2md import DependencyParser

DependencyParser(langs=["it", "de"], output_format="conllu").run(
    input_dir="output/",
    output_dir="parsed/"
)
```

Runs full NLP annotation (tokenize, POS, lemma, depparse) on each book's main Markdown file. Language is detected automatically. Stanza models are downloaded on first run (~500 MB per language).

---

## Output Structure

```
output/
└── book_name/
    ├── book_name.md          # full Markdown output
    ├── images/               # embedded images
    ├── eval_pages/           # PDF: sampled page pairs for evaluation
    │   ├── 0.md              #   LLM-generated Markdown
    │   ├── 0.ref.md          #   PyMuPDF text-layer reference
    │   └── ...
    └── eval_chunks/          # EPUB: sampled chunk pairs for evaluation

scores/
└── book_name_scores.json     # {"average": {"ned": 0.12, "bleu": 68.4, ...}, "pages": {...}}

metadata/
└── metadata.csv              # columns: book, author, title, year, genre

parsed/
└── book_name.conllu          # or .json
```

---

## Markdown Format

- **PDF**: pages separated by `\n\n`, each with a `<!-- Page N -->` header; blank pages are skipped
- **EPUB**: chunks separated by `\n\n`; TOC/nav chapters are skipped

**Repetition filtering** runs two passes after generation:
1. **Line-level**: a line of 25+ chars reappearing within 6 lines triggers truncation.
2. **Inline**: a phrase of 40+ chars reappearing within 400 chars of running text triggers truncation — catches loops inside a single paragraph, common with scanned PDFs.
