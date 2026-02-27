"""
pdf_to_markdown.py — PDF → Markdown converter using Qwen2.5-VL via vLLM.
"""

import logging
from pathlib import Path
import fitz
from pdf2image import convert_from_path
from vllm import LLM, SamplingParams
from config import PDF_MODEL_ID, PDF_DPI, PDF_MAX_NEW_TOKENS, EVAL_N
from utils import pil_to_data_url, sample_indices

logger = logging.getLogger(__name__)


class PDFToMarkdownConverter:
    """Converts PDF documents to Markdown using Qwen2.5-VL via vLLM."""

    _PROMPT = """Convert this PDF page to Markdown.
                Rules:
                - Preserve heading hierarchy (# ## ###)
                - Convert tables to Markdown tables
                - Use $...$ for inline math, $$...$$ for block math
                - Preserve lists and indentation
                - Replace [IMAGE_N] placeholders with ![image_N](images/image_N.png)
                - Output ONLY the Markdown, no commentary"""

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
        self.llm = LLM(model=model_id, dtype="bfloat16")

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
                return self._PROMPT
            placeholders = "\n".join(f"[IMAGE_{i + 1}]" for i in range(len(fnames)))
            return self._PROMPT + f"\n\nImages in order:\n{placeholders}"

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
            text = self._resolve_image_refs(out.outputs[0].text, page_images.get(j, []))
            raw_texts.append(text)
            markdown_pages.append(f"<!-- Page {j + 1} -->\n{text}")

        out_file = output_dir / (pdf_path.stem + ".md")
        out_file.write_text("\n\n---\n\n".join(markdown_pages), encoding="utf-8")
        logger.info("Saved → %s", out_file)

        # Save sampled page image + markdown pairs for later evaluation
        eval_dir = output_dir / "eval_pages"
        eval_dir.mkdir(exist_ok=True)
        for j in sample_indices(len(pages), self.eval_n):
            pages[j].save(str(eval_dir / f"{j}.png"))
            (eval_dir / f"{j}.md").write_text(raw_texts[j], encoding="utf-8")
        logger.info("Eval pages → %s", eval_dir)

        return out_file

    def _extract_images(self, pdf_path: Path, img_dir: Path) -> dict[int, list[str]]:
        """Extract embedded images from PDF. Returns page_idx → [filenames]."""
        page_images: dict[int, list[str]] = {}
        doc = fitz.open(str(pdf_path))
        for page_idx, page in enumerate(doc):
            page_images[page_idx] = []
            for img_idx, img in enumerate(page.get_images(full=True)):
                pix = fitz.Pixmap(doc, img[0])
                if pix.n > 4:  # CMYK → RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                fname = f"page{page_idx + 1}_img{img_idx + 1}.png"
                pix.save(str(img_dir / fname))
                page_images[page_idx].append(fname)
        doc.close()
        return page_images

    @staticmethod
    def _resolve_image_refs(text: str, fnames: list[str]) -> str:
        """Replace [IMAGE_N] placeholders; append any unplaced images at the end."""
        for i, fname in enumerate(fnames):
            ref = f"![image_{i + 1}](images/{fname})"
            placeholder = f"[IMAGE_{i + 1}]"
            text = text.replace(placeholder, ref) if placeholder in text else text + f"\n\n{ref}"
        return text
