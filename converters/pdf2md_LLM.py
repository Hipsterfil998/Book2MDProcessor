"""
pdf_to_markdown.py — PDF → Markdown converter using Qwen2.5-VL via vLLM.
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
import fitz
from pdf2image import convert_from_path
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def _pil_to_data_url(img) -> str:
    """Encode a PIL image as a base64 PNG data URL for vLLM multimodal input."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        dpi: int = 300,
        max_new_tokens: int = 4096,
    ) -> None:
        self.dpi = dpi
        self.max_new_tokens = max_new_tokens
        logger.info("Loading %s...", model_id)
        self.llm = LLM(model=model_id, dtype="bfloat16")

    def convert(self, pdf_path: str | Path, output_dir: str | Path) -> Path:
        """Convert a PDF to Markdown. Returns path to the generated .md file."""
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
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}},
                {"type": "text", "text": make_prompt(page_images.get(j, []))},
            ]}]
            for j, img in enumerate(pages)
        ]

        sampling_params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        outputs = self.llm.chat(messages, sampling_params=sampling_params)

        markdown_pages = []
        for j, out in enumerate(outputs):
            text = self._resolve_image_refs(out.outputs[0].text, page_images.get(j, []))
            markdown_pages.append(f"<!-- Page {j + 1} -->\n{text}")

        out_file = output_dir / (pdf_path.stem + ".md")
        out_file.write_text("\n\n---\n\n".join(markdown_pages), encoding="utf-8")
        logger.info("Saved → %s", out_file)
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
