"""epub_converter.py — EPUB → Markdown via pypandoc + vLLM."""

from pathlib import Path
import pypandoc
from bs4 import BeautifulSoup
from ebooklib import epub
from ebooklib.epub import EpubImage
from vllm import LLM, SamplingParams

PROMPT = """Convert the following HTML to clean Markdown.
Rules:
- Preserve headings, lists, tables and code blocks.
- when there are images convert them to markdown format ![](images/fig.png)
- put the image flag exactly where it is located in the HTML
- Output only the Markdown.

HTML:
{html}
"""


class EpubToMarkdownConverter:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        max_chunk_chars: int = 8_000,
        max_new_tokens: int = 2_048,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.max_new_tokens = max_new_tokens
        self.llm = LLM(model=model_id, dtype="bfloat16")

    def _chunk(self, html: str) -> list[str]:
        """Split HTML into chunks by top-level block tags."""
        soup = BeautifulSoup(html, "html.parser")
        blocks = soup.find_all(["section", "article", "div"], recursive=False)
        if not blocks:
            blocks = soup.find_all(True, recursive=False)

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
        return chunks

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
        """Convert EPUB to Markdown. Images are saved alongside output_path."""
        output = Path(output_path) if output_path else Path(epub_path).with_suffix(".md")
        images_dir = output.parent / "images"

        image_map = self._extract_images(epub_path, images_dir)
        html = pypandoc.convert_file(epub_path, "html")
        html = self._rewrite_img_srcs(html, image_map)

        chunks = self._chunk(html)
        messages = [[{"role": "user", "content": PROMPT.format(html=c)}] for c in chunks]

        sampling_params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        markdown = "\n\n".join(out.outputs[0].text for out in outputs)

        output.write_text(markdown, encoding="utf-8")
        return markdown
