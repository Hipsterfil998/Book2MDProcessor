import glob
from pathlib import Path
from tqdm import tqdm
from converters.text_extraction import DocumentProcessor
from converters.pdf2md_LLM import PDFToMarkdownConverter
from converters.epub2md_LLM import EpubToMarkdownConverter


class ConverterPipeline:

    def __init__(self, input_dir: str, output_dir: str, model_id: str = "google/gemma-3-4b-it", device: str = "cuda"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_id = model_id
        self.device = device

    def run_simple(self) -> None:
        """Convert all PDFs and EPUBs using DocumentProcessor (no LLM)."""
        processor = DocumentProcessor()
        documents = [f for f in glob.glob(f"{self.input_dir}/**", recursive=True)
                     if f.endswith(('.pdf', '.epub'))]

        for filepath in tqdm(documents, desc="Converting", unit="file"):
            doc_output = Path(self.output_dir) / Path(filepath).stem
            if filepath.endswith('.pdf'):
                processor.process_pdf(filepath, output_dir=doc_output)
            else:
                processor.process_epub(filepath, output_dir=doc_output)

    def run_pdf_llm(self) -> None:
        """Convert all PDFs using PDFToMarkdownConverter (LLM)."""
        converter = PDFToMarkdownConverter(model_id=self.model_id, device=self.device)

        for pdf_path in tqdm(glob.glob(f"{self.input_dir}/**/*.pdf", recursive=True), desc="Converting PDFs (LLM)", unit="file"):
            doc_output = Path(self.output_dir) / Path(pdf_path).stem
            doc_output.mkdir(parents=True, exist_ok=True)
            converter.convert(pdf_path, output_dir=doc_output)

    def run_epub_llm(self) -> None:
        """Convert all EPUBs using EpubToMarkdownConverter (LLM)."""
        converter = EpubToMarkdownConverter(model_id=self.model_id, device=self.device)

        for epub_path in tqdm(glob.glob(f"{self.input_dir}/**/*.epub", recursive=True), desc="Converting EPUBs (LLM)", unit="file"):
            doc_output = Path(self.output_dir) / (Path(epub_path).stem + ".md")
            converter.convert(epub_path, output_path=str(doc_output))
