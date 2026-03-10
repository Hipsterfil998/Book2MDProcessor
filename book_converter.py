from config import PDF_MODEL_ID, TEXT_MODEL_ID, INPUT_DIR, OUTPUT_DIR
from pathlib import Path
from tqdm import tqdm
from converters.text_extraction import DocumentProcessor
from converters.pdf2md_LLM import PDFToMarkdownConverter
from converters.epub2md_LLM import EpubToMarkdownConverter


class ConverterPipeline:

    def __init__(
        self,
        input_dir: str = INPUT_DIR,
        output_dir: str = OUTPUT_DIR,
        pdf_model_id: str = PDF_MODEL_ID,
        text_model_id: str = TEXT_MODEL_ID,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pdf_model_id = pdf_model_id
        self.text_model_id = text_model_id

    def _already_converted(self, stem: str) -> bool:
        """Return True if the output .md file for this book already exists."""
        return (Path(self.output_dir) / stem / f"{stem}.md").exists()

    def run_simple(self) -> None:
        """Convert all PDFs and EPUBs using DocumentProcessor (no LLM)."""
        processor = DocumentProcessor()
        documents = [
            p for p in Path(self.input_dir).rglob("*")
            if p.suffix.lower() in (".pdf", ".epub")
        ]

        for filepath in tqdm(documents, desc="Converting", unit="file"):
            if self._already_converted(filepath.stem):
                tqdm.write(f"Skipping (already converted): {filepath.name}")
                continue
            doc_output = Path(self.output_dir) / filepath.stem
            if filepath.suffix.lower() == ".pdf":
                processor.process_pdf(str(filepath), output_dir=doc_output)
            else:
                processor.process_epub(str(filepath), output_dir=doc_output)

    def run_pdf_llm(self) -> None:
        """Convert all PDFs using PDFToMarkdownConverter (LLM)."""
        converter = PDFToMarkdownConverter(model_id=self.pdf_model_id)

        pdf_files = [p for p in Path(self.input_dir).rglob("*") if p.suffix.lower() == ".pdf"]
        for pdf_path in tqdm(pdf_files, desc="Converting PDFs (LLM)", unit="file"):
            if self._already_converted(pdf_path.stem):
                tqdm.write(f"Skipping (already converted): {pdf_path.name}")
                continue
            doc_output = Path(self.output_dir) / pdf_path.stem
            doc_output.mkdir(parents=True, exist_ok=True)
            converter.convert(pdf_path, output_dir=doc_output)

    def run_epub_llm(self) -> None:
        """Convert all EPUBs using EpubToMarkdownConverter (LLM)."""
        converter = EpubToMarkdownConverter(model_id=self.text_model_id)

        epub_files = [p for p in Path(self.input_dir).rglob("*") if p.suffix.lower() == ".epub"]
        for epub_path in tqdm(epub_files, desc="Converting EPUBs (LLM)", unit="file"):
            if self._already_converted(epub_path.stem):
                tqdm.write(f"Skipping (already converted): {epub_path.name}")
                continue
            doc_output_dir = Path(self.output_dir) / epub_path.stem
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            doc_output_md = doc_output_dir / (epub_path.stem + ".md")
            converter.convert(epub_path, output_path=str(doc_output_md))
