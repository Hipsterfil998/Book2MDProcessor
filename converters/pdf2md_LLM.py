"""
pdf_to_markdown.py — PDF → Markdown converter using Qwen2.5-VL via vLLM.
"""

import logging
from pathlib import Path
import fitz
from pdf2image import convert_from_path
from config import PDF_MODEL_ID, PDF_DPI, PDF_MAX_NEW_TOKENS, EVAL_N, PDF_PROMPT, ENABLE_PREFIX_CACHING
from vllm import LLM, SamplingParams
from utils import pil_to_data_url, sample_indices, suppress_worker_stderr

logger = logging.getLogger(__name__)


class PDFToMarkdownConverter:
    """Converts PDF documents to Markdown using Qwen2.5-VL via vLLM."""

    def __init__(
        self,
        model_id: str = PDF_MODEL_ID,
        dpi: int = PDF_DPI,
        max_new_tokens: int = PDF_MAX_NEW_TOKENS,
        eval_n: int = EVAL_N,
    ) -> None:
        self.dpi = dpi
        self.max_new_tokens = max_new_tokens
        self.eval_n = eval_n
        logger.info("Loading %s...", model_id)
        with suppress_worker_stderr():
            self.llm = LLM(model=model_id, dtype="bfloat16", enforce_eager=True,
                           enable_prefix_caching=ENABLE_PREFIX_CACHING)

    def convert(self, pdf_path: str | Path, output_dir: str | Path) -> Path:
        """Convert a PDF to Markdown. Returns path to the generated .md file.
        Sampled pages for evaluation are saved to output_dir/eval_pages/."""
        pdf_path, output_dir = Path(pdf_path), Path(output_dir)
        img_dir = output_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        pages = convert_from_path(str(pdf_path), dpi=self.dpi)
        page_images = self._extract_images(pdf_path, img_dir)

        def make_prompt(fnames: list[str]) -> str:
            if not fnames:
                return PDF_PROMPT
            placeholders = "\n".join(f"[IMAGE_{i + 1}]" for i in range(len(fnames)))
            return PDF_PROMPT + f"\n\nImages in order:\n{placeholders}"

        messages = [
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": pil_to_data_url(img)}},
                {"type": "text", "text": make_prompt(page_images.get(j, []))},
            ]}]
            for j, img in enumerate(pages)
        ]

        sampling_params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        outputs = self.llm.chat(messages, sampling_params=sampling_params)

        markdown_pages = []
        raw_texts = []
        for j, out in enumerate(outputs):
            raw = self._clean(out.outputs[0].text)
            text = self._resolve_image_refs(raw, page_images.get(j, []))
            raw_texts.append(text)
            markdown_pages.append(f"<!-- Page {j + 1} -->\n{text}")

        out_file = output_dir / (pdf_path.stem + ".md")
        out_file.write_text("\n\n---\n\n".join(markdown_pages), encoding="utf-8")
        logger.info("Saved → %s", out_file)

        # Save sampled Markdown pairs for later evaluation.
        # {j}.ref.md: PDF text layer via PyMuPDF Markdown export (no image, no LLM).
        # {j}.md:     LLM-generated Markdown for the same page.
        eval_dir = output_dir / "eval_pages"
        eval_dir.mkdir(exist_ok=True)
        doc_ref = fitz.open(str(pdf_path))
        for j in sample_indices(len(pages), self.eval_n):
            (eval_dir / f"{j}.md").write_text(raw_texts[j], encoding="utf-8")
            try:
                ref_md = doc_ref[j].get_text("markdown").strip()
            except (AssertionError, ValueError):
                ref_md = doc_ref[j].get_text("text").strip()  # PyMuPDF < 1.24
            (eval_dir / f"{j}.ref.md").write_text(ref_md, encoding="utf-8")
        doc_ref.close()
        logger.info("Eval pages → %s", eval_dir)

        return out_file

    def _extract_images(self, pdf_path: Path, img_dir: Path) -> dict[int, list[str]]:
        """Extract unique content images from PDF.

        Deduplicates by xref and skips images that appear on > 30% of pages
        (headers, footers, watermarks). Returns page_idx → [filenames].
        """
        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)

        # Count how many distinct pages each xref appears on
        xref_page_count: dict[int, int] = {}
        for page in doc:
            for xref in {img[0] for img in page.get_images(full=True)}:
                xref_page_count[xref] = xref_page_count.get(xref, 0) + 1

        # Exclude recurring decorative elements (headers / footers / watermarks)
        max_pages = max(3, int(n_pages * 0.3))
        content_xrefs = {xref for xref, c in xref_page_count.items() if c <= max_pages}

        # Save each unique content image once
        xref_to_fname: dict[int, str] = {}
        img_counter = 0
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                if xref not in content_xrefs or xref in xref_to_fname:
                    continue
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.colorspace and pix.colorspace.n > 3:  # CMYK or similar → RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    fname = f"img{img_counter + 1}.png"
                    pix.save(str(img_dir / fname))
                except Exception:
                    continue  # skip images with unsupported colorspace
                img_counter += 1
                xref_to_fname[xref] = fname

        page_images: dict[int, list[str]] = {
            page_idx: [
                xref_to_fname[img[0]]
                for img in page.get_images(full=True)
                if img[0] in xref_to_fname
            ]
            for page_idx, page in enumerate(doc)
        }
        doc.close()
        return page_images

    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown code fences the model sometimes wraps output in."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[1:end]).strip()
        return text

    @staticmethod
    def _resolve_image_refs(text: str, fnames: list[str]) -> str:
        """Replace [IMAGE_N] placeholders; append any unplaced images at the end."""
        for i, fname in enumerate(fnames):
            ref = f"![image_{i + 1}](images/{fname})"
            placeholder = f"[IMAGE_{i + 1}]"
            text = text.replace(placeholder, ref) if placeholder in text else text + f"\n\n{ref}"
        return text
