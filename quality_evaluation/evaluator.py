"""evaluator.py — Quality evaluator for PDF/EPUB to Markdown conversions.

Uses reference-based metrics from Page2MDBench:
  - NED              (Normalised Edit Distance, lower is better)
  - BLEU             (higher is better)
  - MarkdownStructureF1 (higher is better)
  - BERTScore        (optional, higher is better)

PDF:  compares {i}.ref.md (DocumentProcessor rule-based Markdown saved during
      conversion) against {i}.md (LLM-generated Markdown).

EPUB: converts {i}.html to Markdown via DocumentProcessor, then compares
      against {i}.md (LLM-generated Markdown).

Requires Page2MDBench to be cloned at <project_root>/Page2MDBench/:
  git clone https://github.com/Hipsterfil998/Page2MDBench.git
"""

import json
import sys
from pathlib import Path

def _import_metrics():
    try:
        from metrics import NED, BLEU, MarkdownStructureF1, BERTScore
        return NED, BLEU, MarkdownStructureF1, BERTScore
    except ImportError:
        bench = Path(__file__).parent.parent / "Page2MDBench"
        if bench.exists():
            sys.path.insert(0, str(bench))
            from metrics import NED, BLEU, MarkdownStructureF1, BERTScore
            return NED, BLEU, MarkdownStructureF1, BERTScore
        raise ImportError(
            "Page2MDBench not found. Clone it into the project root:\n"
            "  git clone https://github.com/Hipsterfil998/Page2MDBench.git"
        )


NED, BLEU, MarkdownStructureF1, BERTScore = _import_metrics()


class QualityEvaluator:
    """Evaluates PDF/EPUB to Markdown quality using reference-based metrics.

    PDF:  reference = rule-based Markdown ({i}.ref.md) saved during conversion.
    EPUB: reference = {i}.html converted to Markdown via DocumentProcessor.

    No LLM or GPU required.
    """

    def __init__(self, use_bertscore: bool = False):
        self.ned = NED()
        self.bleu = BLEU()
        self.structure_f1 = MarkdownStructureF1()
        self.use_bertscore = use_bertscore
        self.bert = BERTScore() if use_bertscore else None

    def _score_pair(self, reference: str, prediction: str) -> dict:
        result = {
            "ned":          round(self.ned.score(reference, prediction), 4),
            "bleu":         round(self.bleu.score(reference, prediction), 2),
            "structure_f1": round(self.structure_f1.score(reference, prediction), 4),
        }
        if self.use_bertscore:
            result["bertscore"] = round(self.bert.score(reference, prediction), 4)
        return result

    def evaluate_pdf(self, eval_pages_dir: str | Path, scores_dir: str | Path) -> dict:
        """Evaluate a PDF to Markdown conversion.

        Reads {i}.ref.md (rule-based reference) + {i}.md (LLM prediction) pairs.
        {i}.ref.md is generated automatically by PDFToMarkdownConverter.
        """
        eval_pages_dir, scores_dir = Path(eval_pages_dir), Path(scores_dir)
        scores_dir.mkdir(parents=True, exist_ok=True)

        ref_files = sorted(
            eval_pages_dir.glob("*.ref.md"),
            key=lambda p: int(p.stem.split(".")[0])
        )
        if not ref_files:
            print(f"  No .ref.md files in {eval_pages_dir}. Re-run conversion to generate references.")
            return {}

        scores = {}
        skipped = 0
        for rf in ref_files:
            idx = int(rf.stem.split(".")[0])
            pred = eval_pages_dir / f"{idx}.md"
            if not pred.exists():
                continue
            ref_text = rf.read_text(encoding="utf-8").strip()
            if not ref_text:
                skipped += 1
                continue  # scanned page: no embedded text, reference-based metrics not applicable
            scores[idx] = self._score_pair(ref_text, pred.read_text(encoding="utf-8"))
        if skipped:
            print(f"  Skipped {skipped}/{len(ref_files)} page(s): scanned/image-only (no embedded text for reference).")

        result = self._build_result(scores, self._dims(), key="pages")
        (scores_dir / (eval_pages_dir.parent.name + "_scores.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    def evaluate_epub(self, eval_chunks_dir: str | Path, scores_dir: str | Path) -> dict:
        """Evaluate an EPUB to Markdown conversion.

        Reads {i}.ref.md (HTML-to-Markdown reference saved during conversion)
        and compares against {i}.md (LLM prediction).
        """
        eval_chunks_dir, scores_dir = Path(eval_chunks_dir), Path(scores_dir)
        scores_dir.mkdir(parents=True, exist_ok=True)

        ref_files = sorted(
            eval_chunks_dir.glob("*.ref.md"),
            key=lambda p: int(p.stem.split(".")[0])
        )
        if not ref_files:
            print(f"  No .ref.md files in {eval_chunks_dir}. Re-run conversion to generate references.")
            return {}

        scores = {}
        skipped = 0
        for rf in ref_files:
            idx = int(rf.stem.split(".")[0])
            pred = eval_chunks_dir / f"{idx}.md"
            if not pred.exists():
                continue
            ref_text = rf.read_text(encoding="utf-8").strip()
            if not ref_text:
                skipped += 1
                continue
            scores[idx] = self._score_pair(ref_text, pred.read_text(encoding="utf-8"))
        if skipped:
            print(f"  Skipped {skipped}/{len(ref_files)} chunk(s) with empty reference.")

        result = self._build_result(scores, self._dims(), key="chunks")
        (scores_dir / (eval_chunks_dir.parent.name + "_scores.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    def evaluate_all(self, output_dir: str | Path, scores_dir: str | Path) -> dict:
        """Evaluate all converted books found in output_dir.

        Detects conversion type per book (eval_pages/ = PDF, eval_chunks/ = EPUB)
        and calls the matching method. Returns a dict mapping book name to scores.
        """
        output_dir, scores_dir = Path(output_dir), Path(scores_dir)
        results = {}

        book_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
        pdf_dirs  = [d for d in book_dirs if (d / "eval_pages").exists()]
        epub_dirs = [d for d in book_dirs if (d / "eval_chunks").exists()]
        skipped   = [d for d in book_dirs if d not in pdf_dirs and d not in epub_dirs]

        for d in skipped:
            print(f"Skipping (no eval data): {d.name}")
        for d in pdf_dirs:
            print(f"Evaluating PDF: {d.name}")
            results[d.name] = self.evaluate_pdf(d / "eval_pages", scores_dir)
        for d in epub_dirs:
            print(f"Evaluating EPUB: {d.name}")
            results[d.name] = self.evaluate_epub(d / "eval_chunks", scores_dir)

        return results

    def _dims(self) -> tuple:
        return ("ned", "bleu", "structure_f1") + (("bertscore",) if self.use_bertscore else ())

    @staticmethod
    def _build_result(scores: dict, dims: tuple, key: str) -> dict:
        avg = {}
        for k in dims:
            vals = [s[k] for s in scores.values() if isinstance(s.get(k), (int, float))]
            avg[k] = round(sum(vals) / len(vals), 4) if vals else None
        return {"average": avg, key: {str(j): scores[j] for j in sorted(scores)}}
