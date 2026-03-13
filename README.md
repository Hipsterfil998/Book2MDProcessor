<div align="center">

<img src="docs/logo.svg" alt="BookConverter logo" width="320"/>

# Book2MD Converter

</div>

Pipeline for converting Italian and German books (PDF/EPUB) to structured Markdown, extracting bibliographic metadata, and evaluating conversion quality, all via vLLM inference.

> **Recommended GPU:** NVIDIA A100 or equivalent (Google Colab G4).

**[📖 Full Documentation](https://hipsterfil998.github.io/Book2MDProcessor/)**

---

## Project Structure

```
├── config.py                      # Central configuration: models, paths, parameters, prompts
├── utils.py                       # Shared utilities (image encoding, stratified sampling)
├── book_converter.py              # Main pipeline orchestrator (with resume support)
├── converters/
│   ├── text_extraction.py         # Rule-based PDF/EPUB -> Markdown (no LLM)
│   ├── pdf2md_LLM.py              # PDF -> Markdown via Qwen2.5-VL (vision LLM)
│   └── epub2md_LLM.py             # EPUB -> Markdown via Qwen2.5 (text LLM)
├── metadata/
│   └── metadata_extractor.py      # Author/title/year/genre extraction (with CSV resume)
├── quality_evaluation/
│   └── evaluator.py               # Reference-based quality evaluation (NED, BLEU, MarkdownStructureF1)
├── dependency_parsing/
│   └── dependency_parsing.py      # Stanza-based dependency parsing
├── tests/                         # pytest test suite
└── requirements.txt
```

---

## Configuration

All parameters and prompts are centralised in `config.py`. Edit this file before running; no need to touch individual modules.

```python
# Models
PDF_MODEL_ID  = "Qwen/Qwen3-VL-4B-Instruct"  # vision-language (PDF -> Markdown)
TEXT_MODEL_ID = "Qwen/Qwen3-4B"              # text-only (EPUB -> Markdown, metadata)

# Generation
PDF_MAX_NEW_TOKENS        = 4096
EPUB_MAX_NEW_TOKENS       = 2_048
METADATA_MAX_NEW_TOKENS   = 128
PDF_REPETITION_PENALTY    = 1.15   # prevents looping repetitions
EPUB_REPETITION_PENALTY   = 1.15

# Evaluation sampling
EVAL_N = 20  # pages/chunks sampled per book

# Paths
INPUT_DIR    = "books/"
OUTPUT_DIR   = "output/"
SCORES_DIR   = "scores/"
METADATA_CSV = "metadata/metadata.csv"

# Prompts (PDF_PROMPT, EPUB_PROMPT, BIBLIO_PROMPT, GENRE_PROMPT)
```

Every constructor still accepts the same parameters explicitly, so individual overrides remain possible without editing the config.

---

## Setup on Google Colab

```python
# Conversion only
!bash setup.sh

# Conversion + evaluation
!bash setup.sh --with-eval
```

The script installs `poppler-utils` and all Python dependencies. The `--with-eval` flag also clones [Page2MDBench](https://github.com/Hipsterfil998/Page2MDBench) and installs the evaluation libraries.

### 3. GPU requirements (conversion only)

| Model | VRAM (bfloat16) | Recommended |
|---|---|---|
| Qwen3-VL-4B-Instruct (PDF) | ~10 GB | A100 / L4 / T4 |
| Qwen3-4B (EPUB + metadata) | ~8 GB | T4 |

Both models are never loaded simultaneously; the pipeline loads one at a time.

Quality evaluation does **not** require a GPU or any LLM. NED, BLEU, and MarkdownStructureF1 run on CPU. BERTScore (optional) automatically uses GPU if available, otherwise CPU.

---

## Usage

Both a Python interface and a command-line interface are available. They are fully equivalent.

All CLI flags default to the values in `config.py`. Directory overrides (`--input`, `--output`, etc.) go **before** the subcommand:

```bash
python cli.py --input /miei/libri/ --output /risultati/ convert --pdf
python cli.py --output /risultati/ --scores /punteggi/ evaluate
```

Run `python cli.py --help` or `python cli.py <subcommand> --help` for the full option list.

### Convert books (PDF and/or EPUB)

**Python API:**
```python
from book_converter import ConverterPipeline

pipeline = ConverterPipeline(
    input_dir="books/",
    output_dir="output/",
    # pdf_model_id="Qwen/Qwen3-VL-4B-Instruct",   # default
    # text_model_id="Qwen/Qwen3-4B",              # default
)

pipeline.run_pdf_llm()    # converts all .pdf files
pipeline.run_epub_llm()   # converts all .epub files
# pipeline.run_simple()   # rule-based fallback, no LLM
```

**CLI:**
```bash
python cli.py convert --pdf
python cli.py convert --epub
python cli.py convert --simple
```

**Resume support:** if `output/` already contains converted books (i.e. `output/{book_name}/{book_name}.md` exists), those books are automatically skipped. Re-running the pipeline after an interruption will pick up from where it left off.

### Extract metadata

**Python API:**
```python
from metadata.metadata_extractor import MetadataExtractor

extractor = MetadataExtractor()
extractor.run(
    output_dir="output/",              # folder with converted book subfolders
    output_csv="metadata/metadata.csv"
)
```

**CLI:**
```bash
python cli.py metadata
```

Extracts per book:
- **Author, title, year**: from front pages (title page, TOC, preface)
- **Genre**: from body pages (main content)

Genres: `Journalistic`, `Functional/Gebrauchstexte`, `Factual/Non fiction/Wissenschaft`, `Fiction/Belletristik`

**Resume support:** if `metadata.csv` already exists, books already listed in it are skipped and only new entries are appended. The existing rows are never overwritten.

### Evaluate conversion quality

Evaluation uses reference-based metrics from [Page2MDBench](https://github.com/Hipsterfil998/Page2MDBench). No LLM or GPU required.

**Python API:**
```python
from quality_evaluation.evaluator import QualityEvaluator

evaluator = QualityEvaluator(use_bertscore=False)  # set True to also compute BERTScore
evaluator.evaluate_all(output_dir="output/", scores_dir="scores/")
```

**CLI:**
```bash
python cli.py evaluate
python cli.py evaluate --bertscore   # also compute BERTScore
```

`evaluate_all` detects the conversion type of each book automatically (`eval_pages/` = PDF, `eval_chunks/` = EPUB) and calls the right method.

Individual books can also be evaluated directly:

```python
evaluator.evaluate_pdf(eval_pages_dir="output/book_name/eval_pages/", scores_dir="scores/")
evaluator.evaluate_epub(eval_chunks_dir="output/book_name/eval_chunks/", scores_dir="scores/")
```

**Metrics:**

| Metric | Direction | Description |
|---|---|---|
| NED | lower is better | Normalised Edit Distance |
| BLEU | higher is better | n-gram precision |
| MarkdownStructureF1 | higher is better | Structural element overlap |
| BERTScore | higher is better | Semantic similarity (optional, slow on CPU) |

NED, BLEU, and MarkdownStructureF1 run on CPU with no GPU needed. BERTScore automatically uses GPU if available (`cuda`), otherwise falls back to CPU — which works but is significantly slower.

**References:** PDF evaluation compares `{i}.ref.md` (rule-based Markdown saved during conversion) against `{i}.md` (LLM output). EPUB evaluation converts `{i}.html` to Markdown via DocumentProcessor and compares it against `{i}.md`.

---

## Output Structure

After conversion, each book produces a self-contained subfolder:

```
output/
└── book_name/
    ├── book_name.md          # full Markdown output
    ├── images/               # embedded images extracted from the source
    ├── eval_pages/           # PDF only: sampled Markdown pairs for evaluation
    │   ├── 0.md              #   LLM-generated Markdown for page 0
    │   ├── 0.ref.md          #   PyMuPDF text-layer reference for page 0
    │   ├── 12.md
    │   ├── 12.ref.md
    │   └── ...
    └── eval_chunks/          # EPUB only: sampled Markdown pairs for evaluation
        ├── 0.md              #   LLM-generated Markdown for chunk 0
        ├── 0.ref.md          #   HTML-to-Markdown reference for chunk 0
        ├── 5.md
        ├── 5.ref.md
        └── ...
```

Evaluation scores are saved separately:

```
scores/
└── book_name_scores.json     # {"average": {"ned": 0.12, "bleu": 68.4, "structure_f1": 0.91}, "pages": {...}}
```

Metadata output:

```
metadata/
└── metadata.csv              # columns: book, author, title, year, genre
```

### Dependency parsing

**Python API:**
```python
from dependency_parsing.dependency_parsing import DependencyParser

parser = DependencyParser(langs=["it", "de"], output_format="conllu")
parser.run(input_dir="output/", output_dir="parsed/")
```

**CLI:**
```bash
python cli.py parse
python cli.py parse --langs it de --format conllu
```

Runs full NLP annotation on the main Markdown file of each book (`output/{stem}/{stem}.md`):
- tokenization, POS tagging, lemmatization, dependency parsing (via Stanza)
- Markdown is stripped to plain text before parsing
- Language is detected automatically per book using `langdetect`; only the matching pipeline is used
- Output formats: `conllu` (CoNLL-U), `json`, or `both`
- One output file per book: `{stem}.conllu` / `{stem}.json` (no language suffix)

Each token in the output carries: `id`, `text`, `lemma`, `upos`, `xpos`, `feats`, `head`, `deprel`.

Stanza models are downloaded automatically on first run (~500 MB per language).

---

## Markdown Format

- **PDF**: pages separated by `\n\n`, each with a `<!-- Page N -->` header; blank pages are automatically skipped
- **EPUB**: chunks separated by `\n\n`; TOC/nav chapters are automatically skipped

**Output cleaning (both formats):** after generation, each chunk/page goes through `_clean()` (strips code fences and prompt echoes) then `truncate_repetitions()`, which runs two passes:
1. **Line-level**: a line of 25+ chars reappearing within 6 lines triggers truncation.
2. **Inline**: a phrase of 40+ chars reappearing within 400 chars of running text triggers truncation — catches loops inside a single paragraph, common with scanned PDFs.

---

## Evaluation Details

During conversion, `eval_n=20` pages/chunks are sampled using stratified sampling:

| Zone | Pages sampled | Content |
|---|---|---|
| Front (first 10%) | ~3 | Title page, TOC, preface |
| Body (middle 80%) | ~14 | Main text, tables, formulas |
| Back (last 10%) | ~3 | Bibliography, index, appendices |

The same sampled pages are reused for both metadata extraction (front -> biblio, body -> genre) and quality evaluation: no reprocessing of original files required.
