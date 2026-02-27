"""epub_converter.py — EPUB → Markdown via pypandoc + vLLM."""

import random
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
        eval_n: int = 20,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.max_new_tokens = max_new_tokens
        self.eval_n = eval_n
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
        """Convert EPUB to Markdown. Images are saved alongside output_path.
        Chunks are separated by \\n\\n---\\n\\n; sampled chunks saved to eval_chunks/."""
        output = Path(output_path) if output_path else Path(epub_path).with_suffix(".md")
        images_dir = output.parent / "images"

        image_map = self._extract_images(epub_path, images_dir)
        html = pypandoc.convert_file(epub_path, "html")
        html = self._rewrite_img_srcs(html, image_map)

        chunks = self._chunk(html)
        messages = [[{"role": "user", "content": PROMPT.format(html=c)}] for c in chunks]

        sampling_params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        raw_texts = [out.outputs[0].text for out in outputs]
        markdown = "\n\n---\n\n".join(raw_texts)

        output.write_text(markdown, encoding="utf-8")

        # Save sampled HTML chunk + markdown pairs for later evaluation
        eval_dir = output.parent / "eval_chunks"
        eval_dir.mkdir(exist_ok=True)
        for j in self._sample_indices(len(chunks), self.eval_n):
            (eval_dir / f"{j}.html").write_text(chunks[j], encoding="utf-8")
            (eval_dir / f"{j}.md").write_text(raw_texts[j], encoding="utf-8")

        return markdown

    @staticmethod
    def _sample_indices(total: int, n: int = 20) -> list[int]:
        """Stratified sampling across front / body / back of the document."""
        if total <= n:
            return list(range(total))
        front = list(range(0, max(1, total // 10)))
        back  = list(range(total - max(1, total // 10), total))
        body  = list(range(len(front), total - len(back)))
        n_front = max(1, n // 7)
        n_back  = max(1, n // 7)
        n_body  = n - n_front - n_back
        return sorted(
            random.sample(front, min(n_front, len(front))) +
            random.sample(body,  min(n_body,  len(body)))  +
            random.sample(back,  min(n_back,  len(back)))
        )
