
import re
from pathlib import Path
import fitz
from ebooklib import epub
from ebooklib.epub import EpubHtml
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm

_CAPTION_RE = re.compile(r"^(figura|figure|fig\.?|tabella|table|tab\.?)\s*\d*", re.I)


class DocumentProcessor:
    """Process PDF and EPUB documents to structure-preserving Markdown."""

    # --- PDF ------------------------------------------------------------------

    def process_pdf(self, pdf_path: str, output_dir: str = "output", min_img_size: int = 50) -> list[dict]:
        """
        Extract markdown from PDF preserving inline image positions,
        captions, footnotes, and heading hierarchy.

        Args:
            pdf_path:     Path to the source PDF.
            output_dir:   Directory for markdown and images output.
            min_img_size: Minimum image dimension (px) to extract.

        Returns:
            List of per-page dicts with keys: page, blocks.
        """
        output = Path(output_dir)
        img_dir = output / "images"
        output.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(exist_ok=True)

        filename = Path(pdf_path).stem
        doc = fitz.open(pdf_path)
        pages = []

        for page_num, page in tqdm(enumerate(doc, 1), desc=filename, unit="page", total=len(doc)):
            blocks = self._extract_pdf_page_blocks(doc, page, page_num, img_dir, min_img_size)
            pages.append({"page": page_num, "blocks": blocks})

        doc.close()

        md_lines = []
        for p in pages:
            md_lines.extend(p["blocks"])
            md_lines.append("")  # page break separator

        (output / f"{filename}.md").write_text("\n".join(md_lines), encoding="utf-8")
        return pages

    def _extract_pdf_page_blocks(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_num: int,
        img_dir: Path,
        min_img_size: int,
    ) -> list[str]:
        """
        Collect text and image blocks sorted by vertical position (y0)
        to preserve reading order.
        """
        raw_dict = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        # (y0, markdown_string)
        text_blocks: list[tuple[float, str]] = []
        for block in raw_dict["blocks"]:
            if block["type"] != 0:
                continue
            md = self._pdf_block_to_markdown(block)
            if md.strip():
                text_blocks.append((block["bbox"][1], md))

        img_blocks: list[tuple[float, str]] = []
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                img = doc.extract_image(xref)
                if not img.get("image"):
                    print(f"[warn] p{page_num} xref={xref}: empty image data, skipped")
                    continue
                if img["width"] < min_img_size or img["height"] < min_img_size:
                    continue

                rects = page.get_image_rects(xref)
                y0 = rects[0].y0 if rects else float("inf")

                caption = self._find_pdf_caption(text_blocks, y0, img["height"])
                if caption:
                    text_blocks = [b for b in text_blocks if caption not in b[1]]

                img_filename = f"p{page_num:04d}_i{img_idx + 1}.{img['ext']}"
                (img_dir / img_filename).write_bytes(img["image"])
                img_blocks.append((y0, f"![{caption or ''}](images/{img_filename})\n"))

            except Exception as e:
                print(f"[warn] p{page_num} xref={xref}: {e}")

        all_blocks = sorted(text_blocks + img_blocks, key=lambda b: b[0])
        return [b[1] for b in all_blocks]

    def _pdf_block_to_markdown(self, block: dict) -> str:
        """
        Convert a rawdict text block to markdown.
        Detects headings (font size), bold, italic, captions, footnotes.
        """
        lines = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                text = span.get("text", "")
                flags = span.get("flags", 0)
                is_bold = bool(flags & 16)
                is_italic = bool(flags & 2)
                if is_bold and is_italic:
                    text = f"***{text}***"
                elif is_bold:
                    text = f"**{text}**"
                elif is_italic:
                    text = f"*{text}*"
                line_text += text
            if line_text.strip():
                lines.append(line_text)

        if not lines:
            return ""

        full_text = " ".join(lines).strip()
        sizes = [s.get("size", 0) for l in block["lines"] for s in l["spans"]]
        max_size = max(sizes) if sizes else 0

        # Footnote: small font starting with number or symbol
        if max_size < 9 and re.match(r"^[\d\*†‡§]", full_text):
            return f"> [^fn]: {full_text}"

        # Caption: starts with figure/table keywords
        if _CAPTION_RE.match(full_text):
            return f"*{full_text}*"

        # Headings by font size (thresholds are tunable)
        if max_size >= 22:
            return f"# {full_text}"
        if max_size >= 18:
            return f"## {full_text}"
        if max_size >= 14:
            return f"### {full_text}"
        if max_size >= 12 and any(bool(s.get("flags", 0) & 16) for l in block["lines"] for s in l["spans"]):
            return f"#### {full_text}"

        return full_text

    @staticmethod
    def _find_pdf_caption(text_blocks: list[tuple], img_y0: float, img_height: float) -> str | None:
        """Return caption text if a text block sits immediately below the image."""
        threshold = img_y0 + img_height + 30  # 30pt vertical tolerance
        for y0, text in text_blocks:
            if img_y0 < y0 < threshold and _CAPTION_RE.match(text.strip()):
                return re.sub(r"[#*>`\[\]]", "", text).strip()
        return None

    # --- EPUB -----------------------------------------------------------------

    def process_epub(self, epub_path: str, output_dir: str = "output") -> list[str]:
        """
        Extract markdown from EPUB preserving headings, figures, captions,
        footnotes, tables, callout boxes, and inline formatting.

        Args:
            epub_path:   Path to the source EPUB.
            output_dir:  Directory for markdown output.

        Returns:
            List of per-chapter markdown strings.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        filename = Path(epub_path).stem
        book = epub.read_epub(epub_path)
        chapters = []

        for item in book.get_items():
            if not isinstance(item, EpubHtml):
                continue
            html = item.get_content().decode("utf-8", errors="ignore")
            md = self._epub_html_to_markdown(html)
            if md.strip():
                chapters.append(md)

        (output / f"{filename}.md").write_text("\n\n---\n\n".join(chapters), encoding="utf-8")
        return chapters

    def _epub_html_to_markdown(self, html: str) -> str:
        """Convert EPUB HTML to markdown with full structural preservation."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "meta", "link"]):
            tag.decompose()
        return DocumentProcessor._node_to_markdown(soup.body or soup).strip()

    @staticmethod
    def _node_to_markdown(node: Tag) -> str:
        """Recursively convert an HTML node tree to markdown."""
        if not isinstance(node, Tag):
            return str(node) if node else ""
        return "".join(DocumentProcessor._element_to_markdown(child) for child in node.children)

    @staticmethod
    def _element_to_markdown(el) -> str:  # noqa: C901
        """Convert a single HTML element to its markdown representation."""
        if isinstance(el, NavigableString):
            return str(el)
        if not isinstance(el, Tag):
            return ""

        tag = el.name
        inner = DocumentProcessor._node_to_markdown(el)
        css_class = " ".join(el.get("class", []))

        # Headings
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return f"\n\n{'#' * int(tag[1])} {inner.strip()}\n\n"

        # Paragraphs
        if tag == "p":
            return f"\n\n{inner.strip()}\n\n" if inner.strip() else ""

        # Inline formatting
        if tag in ("strong", "b"):
            return f"**{inner}**"
        if tag in ("em", "i"):
            return f"*{inner}*"
        if tag == "code" and el.parent.name != "pre":
            return f"`{inner}`"
        if tag == "sup":
            return f"^{inner}^"
        if tag == "sub":
            return f"~{inner}~"

        # Code blocks
        if tag == "pre":
            lang = ""
            code_el = el.find("code")
            if code_el:
                lang = next(
                    (c.replace("language-", "") for c in code_el.get("class", []) if c.startswith("language-")),
                    "",
                )
                inner = code_el.get_text()
            return f"\n\n```{lang}\n{inner}\n```\n\n"

        # Lists
        if tag == "ul":
            items = [f"- {DocumentProcessor._node_to_markdown(li).strip()}" for li in el.find_all("li", recursive=False)]
            return "\n\n" + "\n".join(items) + "\n\n"
        if tag == "ol":
            items = [f"{i + 1}. {DocumentProcessor._node_to_markdown(li).strip()}" for i, li in enumerate(el.find_all("li", recursive=False))]
            return "\n\n" + "\n".join(items) + "\n\n"

        # Blockquotes and callout boxes
        if tag == "blockquote" or re.search(r"callout|note|warning|tip|aside", css_class, re.I):
            quoted = "\n".join(f"> {line}" for line in inner.strip().splitlines())
            return f"\n\n{quoted}\n\n"

        # Figures and captions
        if tag == "figure":
            md = ""
            img_el = el.find("img")
            caption_el = el.find("figcaption")
            caption_text = caption_el.get_text(strip=True) if caption_el else ""
            if img_el:
                md += f"\n\n![{img_el.get('alt', caption_text)}]({img_el.get('src', '')})\n\n"
            if caption_text:
                md += f"*{caption_text}*\n\n"
            return md
        if tag == "figcaption":
            return f"*{inner.strip()}*\n\n"
        if tag == "img":
            return f"\n\n![{el.get('alt', '')}]({el.get('src', '')})\n\n"

        # Tables
        if tag == "table":
            return DocumentProcessor._table_to_markdown(el)

        # Footnotes
        epub_type = el.get("epub:type", "")
        if epub_type == "footnote" or re.search(r"footnote|endnote", css_class, re.I):
            fn_id = el.get("id", "fn")
            return f"\n\n[^{fn_id}]: {inner.strip()}\n\n"
        if tag == "a" and (epub_type == "noteref" or re.search(r"noteref|fn-ref", css_class, re.I)):
            return f"[^{el.get('href', '').lstrip('#')}]"

        # Misc
        if tag == "hr":
            return "\n\n---\n\n"
        if tag == "br":
            return "  \n"
        if tag == "a":
            href = el.get("href", "")
            return f"[{inner}]({href})" if href else inner

        # Generic containers (div, section, span, article, main, ...)
        return inner

    @staticmethod
    def _table_to_markdown(table_el: Tag) -> str:
        """Convert an HTML <table> to a GFM markdown table."""
        rows = [
            [td.get_text(separator=" ", strip=True) for td in row.find_all(["th", "td"])]
            for row in table_el.find_all("tr")
        ]
        if not rows:
            return ""
        header = "| " + " | ".join(rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
        body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
        return f"\n\n{header}\n{separator}\n{body}\n\n"