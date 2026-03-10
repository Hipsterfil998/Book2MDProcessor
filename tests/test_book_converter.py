"""Tests for ConverterPipeline file discovery (no GPU needed)."""
from pathlib import Path
from unittest.mock import patch

import pytest

from book_converter import ConverterPipeline


def make_pipeline(input_dir, tmp_path):
    return ConverterPipeline(input_dir=str(input_dir), output_dir=str(tmp_path / "out"))


# ── run_pdf_llm ───────────────────────────────────────────────────────────────

class TestRunPdfLlm:
    class _FakePDFConverter:
        def __init__(self, model_id): pass
        def convert(self, path, output_dir): pass

    def _run(self, tmp_path, filenames: list[str]) -> list:
        for name in filenames:
            (tmp_path / name).write_bytes(b"fake")
        calls = []

        class Converter(self._FakePDFConverter):
            def convert(self, path, output_dir): calls.append(path)

        with patch("book_converter.PDFToMarkdownConverter", Converter):
            make_pipeline(tmp_path, tmp_path).run_pdf_llm()
        return calls

    def test_finds_lowercase_pdf(self, tmp_path):
        assert len(self._run(tmp_path, ["test.pdf"])) == 1

    def test_finds_uppercase_pdf(self, tmp_path):
        assert len(self._run(tmp_path, ["test.PDF"])) == 1

    def test_finds_mixed_case_pdf(self, tmp_path):
        assert len(self._run(tmp_path, ["test.Pdf"])) == 1

    def test_ignores_epub(self, tmp_path):
        assert len(self._run(tmp_path, ["book.epub"])) == 0

    def test_ignores_txt(self, tmp_path):
        assert len(self._run(tmp_path, ["notes.txt"])) == 0

    def test_finds_pdf_in_subdirectory(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        calls = self._run(subdir.parent, [])
        # Place the file manually
        (subdir / "nested.pdf").write_bytes(b"fake")
        calls2 = []

        class Converter(self._FakePDFConverter):
            def convert(self, path, output_dir): calls2.append(path)

        with patch("book_converter.PDFToMarkdownConverter", Converter):
            make_pipeline(tmp_path, tmp_path).run_pdf_llm()
        assert len(calls2) == 1

    def test_finds_multiple_pdfs(self, tmp_path):
        calls = self._run(tmp_path, ["a.pdf", "b.PDF", "c.Pdf"])
        assert len(calls) == 3


# ── run_epub_llm ──────────────────────────────────────────────────────────────

class TestRunEpubLlm:
    class _FakeEPUBConverter:
        def __init__(self, model_id): pass
        def convert(self, epub_path, output_path): pass

    def _run(self, tmp_path, filenames: list[str]) -> list:
        for name in filenames:
            (tmp_path / name).write_bytes(b"fake")
        calls = []

        class Converter(self._FakeEPUBConverter):
            def convert(self, epub_path, output_path): calls.append(epub_path)

        with patch("book_converter.EpubToMarkdownConverter", Converter):
            make_pipeline(tmp_path, tmp_path).run_epub_llm()
        return calls

    def test_finds_lowercase_epub(self, tmp_path):
        assert len(self._run(tmp_path, ["book.epub"])) == 1

    def test_finds_uppercase_epub(self, tmp_path):
        assert len(self._run(tmp_path, ["book.EPUB"])) == 1

    def test_ignores_pdf(self, tmp_path):
        assert len(self._run(tmp_path, ["doc.pdf"])) == 0


# ── run_simple ────────────────────────────────────────────────────────────────

class TestRunSimple:
    def test_finds_pdf_and_epub(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"pdf")
        (tmp_path / "b.epub").write_bytes(b"epub")
        pdf_calls, epub_calls = [], []

        class FakeProcessor:
            def process_pdf(self, path, output_dir): pdf_calls.append(path)
            def process_epub(self, path, output_dir): epub_calls.append(path)

        with patch("book_converter.DocumentProcessor", FakeProcessor):
            make_pipeline(tmp_path, tmp_path).run_simple()

        assert len(pdf_calls) == 1
        assert len(epub_calls) == 1

    def test_finds_uppercase_extensions(self, tmp_path):
        (tmp_path / "A.PDF").write_bytes(b"pdf")
        (tmp_path / "B.EPUB").write_bytes(b"epub")
        pdf_calls, epub_calls = [], []

        class FakeProcessor:
            def process_pdf(self, path, output_dir): pdf_calls.append(path)
            def process_epub(self, path, output_dir): epub_calls.append(path)

        with patch("book_converter.DocumentProcessor", FakeProcessor):
            make_pipeline(tmp_path, tmp_path).run_simple()

        assert len(pdf_calls) == 1
        assert len(epub_calls) == 1

    def test_ignores_other_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("text")
        (tmp_path / "image.png").write_bytes(b"img")
        pdf_calls, epub_calls = [], []

        class FakeProcessor:
            def process_pdf(self, path, output_dir): pdf_calls.append(path)
            def process_epub(self, path, output_dir): epub_calls.append(path)

        with patch("book_converter.DocumentProcessor", FakeProcessor):
            make_pipeline(tmp_path, tmp_path).run_simple()

        assert pdf_calls == []
        assert epub_calls == []


# ── resume / skip already-converted ──────────────────────────────────────────

def _make_md(out_dir, stem: str) -> None:
    """Simulate a completed conversion by creating the expected .md file."""
    md = out_dir / stem / f"{stem}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("done")


class TestSkipAlreadyConverted:
    # ── run_pdf_llm ───────────────────────────────────────────────────────────

    def test_pdf_already_converted_is_skipped(self, tmp_path):
        (tmp_path / "book.pdf").write_bytes(b"fake")
        out_dir = tmp_path / "out"
        _make_md(out_dir, "book")
        calls = []

        class Converter:
            def __init__(self, model_id): pass
            def convert(self, path, output_dir): calls.append(path)

        with patch("book_converter.PDFToMarkdownConverter", Converter):
            ConverterPipeline(input_dir=str(tmp_path), output_dir=str(out_dir)).run_pdf_llm()

        assert calls == []

    def test_pdf_not_yet_converted_is_processed(self, tmp_path):
        (tmp_path / "book.pdf").write_bytes(b"fake")
        out_dir = tmp_path / "out"
        calls = []

        class Converter:
            def __init__(self, model_id): pass
            def convert(self, path, output_dir): calls.append(path)

        with patch("book_converter.PDFToMarkdownConverter", Converter):
            ConverterPipeline(input_dir=str(tmp_path), output_dir=str(out_dir)).run_pdf_llm()

        assert len(calls) == 1

    def test_pdf_partial_resume(self, tmp_path):
        """2 PDFs: one already converted → only the other is processed."""
        (tmp_path / "a.pdf").write_bytes(b"fake")
        (tmp_path / "b.pdf").write_bytes(b"fake")
        out_dir = tmp_path / "out"
        _make_md(out_dir, "a")
        calls = []

        class Converter:
            def __init__(self, model_id): pass
            def convert(self, path, output_dir): calls.append(path.stem)

        with patch("book_converter.PDFToMarkdownConverter", Converter):
            ConverterPipeline(input_dir=str(tmp_path), output_dir=str(out_dir)).run_pdf_llm()

        assert calls == ["b"]

    # ── run_epub_llm ──────────────────────────────────────────────────────────

    def test_epub_already_converted_is_skipped(self, tmp_path):
        (tmp_path / "book.epub").write_bytes(b"fake")
        out_dir = tmp_path / "out"
        _make_md(out_dir, "book")
        calls = []

        class Converter:
            def __init__(self, model_id): pass
            def convert(self, epub_path, output_path): calls.append(epub_path)

        with patch("book_converter.EpubToMarkdownConverter", Converter):
            ConverterPipeline(input_dir=str(tmp_path), output_dir=str(out_dir)).run_epub_llm()

        assert calls == []

    def test_epub_partial_resume(self, tmp_path):
        (tmp_path / "a.epub").write_bytes(b"fake")
        (tmp_path / "b.epub").write_bytes(b"fake")
        out_dir = tmp_path / "out"
        _make_md(out_dir, "a")
        calls = []

        class Converter:
            def __init__(self, model_id): pass
            def convert(self, epub_path, output_path): calls.append(Path(epub_path).stem)

        with patch("book_converter.EpubToMarkdownConverter", Converter):
            ConverterPipeline(input_dir=str(tmp_path), output_dir=str(out_dir)).run_epub_llm()

        assert calls == ["b"]

    # ── run_simple ────────────────────────────────────────────────────────────

    def test_simple_already_converted_pdf_is_skipped(self, tmp_path):
        (tmp_path / "book.pdf").write_bytes(b"fake")
        out_dir = tmp_path / "out"
        _make_md(out_dir, "book")
        calls = []

        class FakeProcessor:
            def process_pdf(self, path, output_dir): calls.append(path)
            def process_epub(self, path, output_dir): calls.append(path)

        with patch("book_converter.DocumentProcessor", FakeProcessor):
            ConverterPipeline(input_dir=str(tmp_path), output_dir=str(out_dir)).run_simple()

        assert calls == []

    def test_simple_partial_resume(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"fake")
        (tmp_path / "b.epub").write_bytes(b"fake")
        out_dir = tmp_path / "out"
        _make_md(out_dir, "a")
        calls = []

        class FakeProcessor:
            def process_pdf(self, path, output_dir): calls.append(Path(path).stem)
            def process_epub(self, path, output_dir): calls.append(Path(path).stem)

        with patch("book_converter.DocumentProcessor", FakeProcessor):
            ConverterPipeline(input_dir=str(tmp_path), output_dir=str(out_dir)).run_simple()

        assert calls == ["b"]
