"""
evaluator.py — Standalone quality evaluator for PDF/EPUB → Markdown conversions.

Reads pre-saved eval_pages/ and eval_chunks/ produced during conversion.
Scores are saved separately from the generated Markdown files.
"""

import json
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams
from utils import pil_to_data_url


_PDF_JUDGE_PROMPT = (
    "You are a faithfulness verifier for document conversion.\n\n"
    "SOURCE DOCUMENT: the page image above.\n"
    "GENERATED MARKDOWN: the text below.\n\n"
    "Your task: verify that every piece of content in the Markdown can be "
    "traced back to the source page. Do not evaluate markdown quality in general — "
    "only check faithfulness to the original.\n\n"
    "Rate from 1 (unfaithful) to 5 (perfectly faithful):\n"
    "- text: all text from the source is present and nothing is invented\n"
    "- structure: headings, tables, lists match the visual layout of the source\n"
    "- math: formulas match the source (rate 5 if no math is present)\n\n"
    'Return ONLY valid JSON: {"text": N, "structure": N, "math": N}\n\n'
    "GENERATED MARKDOWN:\n"
)

_EPUB_JUDGE_PROMPT = (
    "You are a faithfulness verifier for document conversion.\n\n"
    "SOURCE DOCUMENT:\n{html}\n\n"
    "GENERATED MARKDOWN:\n{markdown}\n\n"
    "Your task: verify that every piece of content in the Markdown can be "
    "traced back to the source HTML. Do not evaluate markdown quality in general — "
    "only check faithfulness to the original.\n\n"
    "Rate from 1 (unfaithful) to 5 (perfectly faithful):\n"
    "- text: all text from the source is present and nothing is invented\n"
    "- structure: headings, tables, lists match the source structure\n\n"
    'Return ONLY valid JSON: {"text": N, "structure": N}'
)


class QualityEvaluator:
    """
    Evaluates PDF/EPUB → Markdown conversion quality using an LLM-as-judge.

    Reads pre-saved eval_pages/ (PNG) and eval_chunks/ (HTML) folders
    produced during conversion — no access to original files required.

    Args:
        judge_model_id: HuggingFace model ID to use as judge.
    """

    def __init__(self, judge_model_id: str):
        self.llm = LLM(model=judge_model_id, dtype="bfloat16")

    def evaluate_pdf(
        self,
        eval_pages_dir: str | Path,
        scores_dir: str | Path,
    ) -> dict:
        """
        Evaluate a PDF → Markdown conversion.

        Args:
            eval_pages_dir: Folder with pre-saved {idx}.png and {idx}.md pairs.
            scores_dir: Directory where the scores JSON will be saved.
        Returns:
            Dict with average and per-page scores.
        """
        eval_pages_dir, scores_dir = Path(eval_pages_dir), Path(scores_dir)
        scores_dir.mkdir(parents=True, exist_ok=True)

        page_files = sorted(eval_pages_dir.glob("*.png"), key=lambda p: int(p.stem))

        messages = [
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": pil_to_data_url(Image.open(pf))}},
                {"type": "text", "text": _PDF_JUDGE_PROMPT + (pf.with_suffix(".md")).read_text(encoding="utf-8")},
            ]}]
            for pf in page_files
        ]
        outputs = self.llm.chat(messages, SamplingParams(max_tokens=64, temperature=0.0))

        scores = {}
        for pf, out in zip(page_files, outputs):
            j = int(pf.stem)
            try:
                scores[j] = json.loads(out.outputs[0].text.strip())
            except Exception:
                scores[j] = {"text": None, "structure": None, "math": None}

        result = self._build_result(scores, ("text", "structure", "math"), key="pages")
        (scores_dir / (eval_pages_dir.parent.name + "_scores.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    def evaluate_epub(
        self,
        eval_chunks_dir: str | Path,
        scores_dir: str | Path,
    ) -> dict:
        """
        Evaluate an EPUB → Markdown conversion.

        Args:
            eval_chunks_dir: Folder with pre-saved {idx}.html and {idx}.md pairs.
            scores_dir: Directory where the scores JSON will be saved.
        Returns:
            Dict with average and per-chunk scores.
        """
        eval_chunks_dir, scores_dir = Path(eval_chunks_dir), Path(scores_dir)
        scores_dir.mkdir(parents=True, exist_ok=True)

        chunk_files = sorted(eval_chunks_dir.glob("*.html"), key=lambda p: int(p.stem))

        messages = [
            [{"role": "user", "content": _EPUB_JUDGE_PROMPT.format(
                html=cf.read_text(encoding="utf-8"),
                markdown=cf.with_suffix(".md").read_text(encoding="utf-8"),
            )}]
            for cf in chunk_files
        ]
        outputs = self.llm.chat(messages, SamplingParams(max_tokens=64, temperature=0.0))

        scores = {}
        for cf, out in zip(chunk_files, outputs):
            j = int(cf.stem)
            try:
                scores[j] = json.loads(out.outputs[0].text.strip())
            except Exception:
                scores[j] = {"text": None, "structure": None}

        result = self._build_result(scores, ("text", "structure"), key="chunks")
        (scores_dir / (eval_chunks_dir.parent.name + "_scores.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    @staticmethod
    def _build_result(scores: dict, dims: tuple, key: str) -> dict:
        avg = {
            k: round(
                sum(s[k] for s in scores.values() if isinstance(s.get(k), (int, float)))
                / len(scores), 2
            )
            for k in dims
        }
        return {"average": avg, key: {str(j): scores[j] for j in sorted(scores)}}
