"""epub_converter.py — EPUB → Markdown via ebooklib + vLLM."""

from pathlib import Path
import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from ebooklib.epub import EpubImage
from config import TEXT_MODEL_ID, EPUB_MAX_CHUNK_CHARS, EPUB_MAX_NEW_TOKENS, EVAL_N, EPUB_PROMPT, ENABLE_PREFIX_CACHING
from vllm import LLM, SamplingParams
from utils import sample_indices, suppress_worker_stderr


class EpubToMarkdownConverter:
    def __init__(
        self,
        model_id: str = TEXT_MODEL_ID,
        max_chunk_chars: int = EPUB_MAX_CHUNK_CHARS,
        max_new_tokens: int = EPUB_MAX_NEW_TOKENS,
        eval_n: int = EVAL_N,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.max_new_tokens = max_new_tokens
        self.eval_n = eval_n
        with suppress_worker_stderr():
            self.llm = LLM(model=model_id, dtype="bfloat16",
                           enable_prefix_caching=ENABLE_PREFIX_CACHING)

    def _chunk(self, html: str) -> list[str]:
        """Split a chapter HTML into chunks by block-level tags inside <body>."""
        soup = BeautifulSoup(html, "html.parser")
        root = soup.find("body") or soup
        blocks = root.find_all(["section", "article", "div"], recursive=False)
        if not blocks:
            blocks = root.find_all(True, recursive=False)

        chunks, current = [], ""
        for block in blocks:
            text = str(block)
            if len(current) + len(text) > self.max_chunk_chars:
                if current:
                    chunks.append(current)
                current = text
            else:
                current += text
        if current:
            chunks.append(current)
        return chunks or [str(root)]

    def _extract_images(self, epub_path: str, images_dir: Path) -> dict[str, Path]:
        """Extract all images from EPUB to images_dir. Returns {epub_src: local_path}."""
        images_dir.mkdir(parents=True, exist_ok=True)
        book = epub.read_epub(epub_path)
        mapping = {}
        for item in book.get_items():
            if isinstance(item, EpubImage):
                dest = images_dir / Path(item.file_name).name
                dest.write_bytes(item.content)
                mapping[item.file_name] = dest
        return mapping

    def _rewrite_img_srcs(self, html: str, mapping: dict[str, Path]) -> str:
        """Replace img src attributes in HTML to point to extracted local files."""
        soup = BeautifulSoup(html, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src", "")
            match = next((v for k, v in mapping.items() if Path(k).name == Path(src).name), None)
            if match:
                img["src"] = str(match)
        return str(soup)

    def convert(self, epub_path: str, output_path: str | None = None) -> str:
        """Convert EPUB to Markdown. Images are saved alongside output_path.
        Chunks are separated by \\n\\n---\\n\\n; sampled chunks saved to eval_chunks/."""
        output = Path(output_path) if output_path else Path(epub_path).with_suffix(".md")
        images_dir = output.parent / "images"

        image_map = self._extract_images(epub_path, images_dir)

        # Iterate chapters in spine order using ebooklib directly.
        # This avoids collapsing the whole EPUB into one giant HTML string
        # (which caused pypandoc's output to be chunked as a single oversized block).
        book = epub.read_epub(epub_path)
        spine_ids = [item_id for item_id, _ in book.spine]
        chunks = []
        for item_id in spine_ids:
            item = book.get_item_with_id(item_id)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            chapter_html = item.get_content().decode("utf-8", errors="replace")
            chapter_html = self._rewrite_img_srcs(chapter_html, image_map)
            chunks.extend(self._chunk(chapter_html))
        messages = [[{"role": "user", "content": EPUB_PROMPT.format(html=c)}] for c in chunks]

        sampling_params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        outputs = self.llm.chat(messages, sampling_params=sampling_params,
                                chat_template_kwargs={"enable_thinking": False})
        raw_texts = [out.outputs[0].text for out in outputs]
        markdown = "\n\n".join(raw_texts)

        output.write_text(markdown, encoding="utf-8")

        # Save sampled Markdown pairs for later evaluation.
        # {j}.ref.md: HTML chunk converted to Markdown (reference, no LLM).
        # {j}.md:     LLM-generated Markdown for the same chunk.
        from converters.text_extraction import DocumentProcessor
        _ref_proc = DocumentProcessor()
        eval_dir = output.parent / "eval_chunks"
        eval_dir.mkdir(exist_ok=True)
        for j in sample_indices(len(chunks), self.eval_n):
            (eval_dir / f"{j}.md").write_text(raw_texts[j], encoding="utf-8")
            ref_md = _ref_proc._epub_html_to_markdown(chunks[j])
            (eval_dir / f"{j}.ref.md").write_text(ref_md, encoding="utf-8")

        return markdown

